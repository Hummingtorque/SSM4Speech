import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score

from event_ssm.layers import TorchS5
from event_ssm.seq_model import ClassificationModel


def build_model_from_config(cfg, num_classes: int, device: torch.device) -> ClassificationModel:
    ssm_cfg = cfg.model.ssm
    ssm_init = cfg.model.ssm_init
    audio_cfg = cfg.model.get("audio_encoder", {})
    step_rescale = float(ssm_cfg.get("step_rescale", 1.0))
    pooling_every_n_layers = int(ssm_cfg.get("pooling_every_n_layers", 1))

    input_gate = bool(ssm_cfg.get("input_gate", False))
    input_gate_rank = int(ssm_cfg.get("input_gate_rank", 0))
    input_gate_mode = str(ssm_cfg.get("input_gate_mode", "sigmoid"))
    input_gate_energy_scale_init = float(ssm_cfg.get("input_gate_energy_scale_init", 0.0))
    input_gate_bias_init = float(ssm_cfg.get("input_gate_bias_init", 0.0))
    input_gate_min = float(ssm_cfg.get("input_gate_min", 0.0))

    def s5_factory(d_model_in, d_model_out, d_ssm, block_size, discretization, step_rescale_layer, stride, pooling_mode):
        return TorchS5(
            H_in=d_model_in,
            H_out=d_model_out,
            P=d_ssm,
            block_size=block_size,
            discretization=discretization,
            dt_min=float(ssm_init.dt_min),
            dt_max=float(ssm_init.dt_max),
            step_rescale=float(step_rescale_layer),
            stride=int(stride),
            pooling_mode=str(pooling_mode),
            input_gate=input_gate,
            input_gate_rank=input_gate_rank,
            input_gate_mode=input_gate_mode,
            input_gate_energy_scale_init=input_gate_energy_scale_init,
            input_gate_bias_init=input_gate_bias_init,
            input_gate_min=input_gate_min,
        )

    mel_bins = int(getattr(cfg.training, "mel_bins", 64))
    input_is_mel = bool(audio_cfg.get("use", True))
    audio_encoder_type = str(audio_cfg.get("type", "conv1d"))
    audio_in_channels = int(audio_cfg.get("in_channels", 1))
    audio_freq_stride = int(audio_cfg.get("freq_stride", 1))
    conv_kernel = int(audio_cfg.get("kernel_size", 3))
    conv_stride = int(audio_cfg.get("stride", 1))

    model = ClassificationModel(
        ssm=s5_factory,
        discretization=cfg.model.ssm.discretization,
        num_classes=num_classes,
        d_model=cfg.model.ssm.d_model,
        d_ssm=cfg.model.ssm.d_ssm,
        ssm_block_size=cfg.model.ssm.ssm_block_size,
        num_stages=cfg.model.ssm.num_stages,
        num_layers_per_stage=cfg.model.ssm.num_layers_per_stage,
        num_embeddings=0,
        input_is_mel=input_is_mel,
        mel_bins=mel_bins,
        audio_encoder_type=audio_encoder_type,
        audio_in_channels=audio_in_channels,
        audio_freq_stride=audio_freq_stride,
        conv_kernel=conv_kernel,
        conv_stride=conv_stride,
        dropout=cfg.model.ssm.dropout,
        classification_mode=cfg.model.ssm.classification_mode,
        prenorm=cfg.model.ssm.prenorm,
        batchnorm=cfg.model.ssm.batchnorm,
        bn_momentum=cfg.model.ssm.bn_momentum,
        step_rescale=step_rescale,
        pooling_stride=cfg.model.ssm.pooling_stride,
        pooling_every_n_layers=pooling_every_n_layers,
        pooling_mode=cfg.model.ssm.pooling_mode,
        state_expansion_factor=cfg.model.ssm.state_expansion_factor,
    ).to(device)

    return model


class GateFeatureCollector:
    """
    Collect scalar statistics per gated SSM layer:
    [mean_u, std_u, mean_logit, std_logit, mean_gate, std_gate]
    """

    def __init__(self, model: nn.Module, feature_mode: str = "full"):
        self.layer_names: List[str] = []
        self._batch_feats: Dict[str, torch.Tensor] = {}
        self.handles = []
        self.feature_mode = str(feature_mode)
        for name, module in model.named_modules():
            if isinstance(module, TorchS5) and getattr(module, "input_gate", False):
                gate_net = getattr(module, "input_gate_net", None)
                if gate_net is None:
                    continue
                self.layer_names.append(name)

                def _hook(_mod, inp, out, lname=name):
                    u = inp[0].detach()  # [B,L,H]
                    logits = out.detach()  # [B,L,H]
                    gate = torch.sigmoid(logits)
                    mu_u = u.mean(dim=(1, 2))
                    sd_u = u.std(dim=(1, 2), unbiased=False)
                    mu_l = logits.mean(dim=(1, 2))
                    sd_l = logits.std(dim=(1, 2), unbiased=False)
                    mu_g = gate.mean(dim=(1, 2))
                    sd_g = gate.std(dim=(1, 2), unbiased=False)
                    if self.feature_mode == "mean_gate":
                        feat = mu_g.unsqueeze(1)  # [B,1]
                    else:
                        feat = torch.stack([mu_u, sd_u, mu_l, sd_l, mu_g, sd_g], dim=1)  # [B,6]
                    self._batch_feats[lname] = feat

                self.handles.append(gate_net.register_forward_hook(_hook))

        if len(self.layer_names) == 0:
            raise RuntimeError("No gated TorchS5 layers found. Ensure input_gate=true in the checkpoint config.")

    def pop_batch_features(self) -> torch.Tensor:
        ordered = []
        for lname in self.layer_names:
            if lname not in self._batch_feats:
                raise RuntimeError(f"Missing hook feature for layer: {lname}")
            ordered.append(self._batch_feats[lname])
        self._batch_feats = {}
        return torch.cat(ordered, dim=1)  # [B, num_layers*6]

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def build_mel_fn(cfg):
    sr = int(getattr(cfg.training, "sr", 16000))
    n_fft = int(getattr(cfg.training, "n_fft", 1024))
    hop_length = int(getattr(cfg.training, "hop_length", 160))
    win_length = int(getattr(cfg.training, "win_length", 400))
    mel_bins = int(getattr(cfg.training, "mel_bins", 64))
    top_db = float(getattr(cfg.training, "top_db", 80.0))
    use_log_mel = bool(getattr(cfg.training, "use_log_mel", True))
    dt_ms = float(hop_length) / float(sr) * 1000.0

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=mel_bins,
        power=2.0,
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=float(top_db))

    def wav_to_mel(wav_path: Path) -> np.ndarray:
        y, orig_sr = torchaudio.load(str(wav_path))
        if y.dim() == 2 and y.shape[0] > 1:
            y = y.mean(dim=0, keepdim=True)
        if int(orig_sr) != int(sr):
            y = torchaudio.functional.resample(y, int(orig_sr), int(sr))
        with torch.no_grad():
            mel_power = mel_transform(y)
            if use_log_mel:
                mel_db = amp_to_db(mel_power)
                mel_db = mel_db - mel_db.amax(dim=(-2, -1), keepdim=True)
                mel_db = mel_db.clamp(min=-float(top_db), max=0.0)
                mel = (mel_db + float(top_db)) / float(top_db)
            else:
                max_val = mel_power.amax()
                if not torch.isfinite(max_val) or float(max_val) <= 0.0:
                    mel = torch.zeros_like(mel_power)
                else:
                    mel = mel_power / (max_val + 1e-8)
        return mel.squeeze(0).transpose(0, 1).cpu().numpy().astype(np.float32)  # [L, mel_bins]

    return wav_to_mel, dt_ms, mel_bins


def pad_batch(mels: List[np.ndarray], dt_ms: float, device: torch.device):
    lengths = np.array([m.shape[0] for m in mels], dtype=np.int64)
    max_len = int(max(1, lengths.max()))
    mel_bins = int(mels[0].shape[1])
    x = np.zeros((len(mels), max_len, mel_bins), dtype=np.float32)
    t = np.zeros((len(mels), max_len), dtype=np.float32)
    for i, m in enumerate(mels):
        li = m.shape[0]
        x[i, :li, :] = m
        t[i, :li] = dt_ms
    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(t).to(device),
        torch.from_numpy(lengths).to(device),
    )


class ProbeMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    parser = argparse.ArgumentParser(description="Train a probe MLP on B-gating-stage features (keyword vs background).")
    parser.add_argument("--run-dir", type=str, required=True, help="Training run dir with config.yaml + checkpoint")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint override")
    parser.add_argument("--max-samples", type=int, default=4096, help="Max keyword and background samples each")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--probe-epochs", type=int, default=20)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--feature-mode", type=str, default="full", choices=["full", "mean_gate"])
    parser.add_argument("--out-json", type=str, default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    run_dir = Path(args.run_dir).resolve()
    cfg = OmegaConf.load(str(run_dir / "config.yaml"))
    ckpt_path = Path(args.checkpoint).resolve() if args.checkpoint else (run_dir / "checkpoints" / "model.pt")
    state = torch.load(str(ckpt_path), map_location="cpu")["state_dict"]
    num_classes = int(state["decoder.weight"].shape[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(cfg, num_classes=num_classes, device=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    wav_to_mel, dt_ms, _mel_bins = build_mel_fn(cfg)
    root = Path(str(cfg.training.root))
    bg_dir = root / "_background_noise_"
    test_list = set([x.strip() for x in (root / "testing_list.txt").read_text().splitlines() if x.strip()])
    keywords = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

    keyword_wavs = []
    for lab in keywords:
        d = root / lab
        if not d.exists():
            continue
        for wav in d.glob("*.wav"):
            rel = f"{lab}/{wav.name}"
            if rel in test_list:
                keyword_wavs.append(wav)
    if len(keyword_wavs) == 0:
        raise RuntimeError("No keyword wavs found. Check dataset root and testing_list.txt.")

    bg_wavs = sorted(bg_dir.glob("*.wav"))
    if len(bg_wavs) == 0:
        raise RuntimeError(f"No background wavs found in {bg_dir}")

    n = min(args.max_samples, len(keyword_wavs))
    kw_pick = list(rng.choice(keyword_wavs, size=n, replace=False))
    kw_mels = [wav_to_mel(p) for p in kw_pick]
    kw_lens = [m.shape[0] for m in kw_mels]

    # Build length-matched background segments.
    bg_full = [wav_to_mel(p) for p in bg_wavs]
    bg_mels = []
    for L in kw_lens:
        src = bg_full[int(rng.integers(0, len(bg_full)))]
        if src.shape[0] >= L:
            st = int(rng.integers(0, src.shape[0] - L + 1))
            seg = src[st:st + L, :]
        else:
            seg = np.zeros((L, src.shape[1]), dtype=np.float32)
            seg[:src.shape[0], :] = src
        bg_mels.append(seg.astype(np.float32))

    collector = GateFeatureCollector(model, feature_mode=args.feature_mode)

    def extract_feats(mels: List[np.ndarray]) -> np.ndarray:
        feats = []
        with torch.no_grad():
            for i in range(0, len(mels), args.batch_size):
                batch = mels[i:i + args.batch_size]
                x, t, l = pad_batch(batch, dt_ms=dt_ms, device=device)
                _ = model(x, t, l, train=False)
                f = collector.pop_batch_features().cpu().numpy()
                feats.append(f)
        return np.concatenate(feats, axis=0)

    X_kw = extract_feats(kw_mels)
    X_bg = extract_feats(bg_mels)
    collector.close()

    X = np.concatenate([X_kw, X_bg], axis=0).astype(np.float32)
    y = np.concatenate([np.ones((X_kw.shape[0],), dtype=np.float32), np.zeros((X_bg.shape[0],), dtype=np.float32)], axis=0)

    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    X = X[idx]
    y = y[idx]

    n_train = int(0.8 * X.shape[0])
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    # Normalize on train split
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-6
    X_train = (X_train - mu) / sd
    X_val = (X_val - mu) / sd

    Xt = torch.from_numpy(X_train).to(device)
    yt = torch.from_numpy(y_train).to(device)
    Xv = torch.from_numpy(X_val).to(device)
    yv = torch.from_numpy(y_val).to(device)

    probe = ProbeMLP(input_dim=X.shape[1]).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.probe_lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    best = {"val_auc": -1.0, "val_acc": 0.0, "epoch": -1}
    for ep in range(1, args.probe_epochs + 1):
        probe.train()
        order = torch.randperm(Xt.shape[0], device=device)
        loss_sum = 0.0
        for i in range(0, Xt.shape[0], args.batch_size):
            ids = order[i:i + args.batch_size]
            logits = probe(Xt[ids])
            loss = bce(logits, yt[ids])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * ids.shape[0]

        probe.eval()
        with torch.no_grad():
            lv = probe(Xv)
            pv = torch.sigmoid(lv)
            pred = (pv >= 0.5).float()
            val_acc = float((pred == yv).float().mean().item())
            val_auc = float(roc_auc_score(yv.detach().cpu().numpy(), pv.detach().cpu().numpy()))
        if val_auc > best["val_auc"]:
            best = {"val_auc": val_auc, "val_acc": val_acc, "epoch": ep}
        train_loss = loss_sum / max(1, Xt.shape[0])
        print(f"[probe] epoch {ep:02d} | train_loss {train_loss:.4f} | val_acc {val_acc:.4f} | val_auc {val_auc:.4f}")

    result = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "device": str(device),
        "num_keyword_samples": int(X_kw.shape[0]),
        "num_background_samples": int(X_bg.shape[0]),
        "feature_dim": int(X.shape[1]),
        "probe_epochs": int(args.probe_epochs),
        "feature_mode": str(args.feature_mode),
        "best": best,
    }

    out_path = Path(args.out_json).resolve() if args.out_json else (run_dir / "gate_mlp_probe_result.json")
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[*] Saved probe result: {out_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
