import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from omegaconf import OmegaConf

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


def parse_snr_list(s: str) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for item in s.split(","):
        t = item.strip().lower()
        if t in ("clean", "none", "inf"):
            out.append(None)
        else:
            out.append(float(t))
    return out


def build_test_items(root: Path) -> Tuple[List[Tuple[Path, int]], List[str]]:
    test_list_path = root / "testing_list.txt"
    test_list = [ln.strip() for ln in test_list_path.read_text().splitlines() if ln.strip()]
    labels = sorted([d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("_")])
    label_to_idx = {l: i for i, l in enumerate(labels)}
    items: List[Tuple[Path, int]] = []
    for rel in test_list:
        p = Path(rel)
        if len(p.parts) < 2:
            continue
        lab = p.parts[0]
        if lab not in label_to_idx:
            continue
        wavp = root / rel
        if wavp.exists():
            items.append((wavp, label_to_idx[lab]))
    return items, labels


def load_background_pool(bg_dir: Path, target_sr: int) -> List[torch.Tensor]:
    bg_wavs = sorted(bg_dir.glob("*.wav"))
    if not bg_wavs:
        raise RuntimeError(f"No background wavs found in {bg_dir}")
    pool: List[torch.Tensor] = []
    for p in bg_wavs:
        y, sr = torchaudio.load(str(p))
        if y.dim() == 2 and y.shape[0] > 1:
            y = y.mean(dim=0, keepdim=True)
        if int(sr) != int(target_sr):
            y = torchaudio.functional.resample(y, int(sr), int(target_sr))
        pool.append(y.squeeze(0).contiguous())
    return pool


def snr_token(snr_db: Optional[float]) -> int:
    if snr_db is None:
        return 0
    return int(round((snr_db + 100.0) * 1000))


def pick_noise_segment(bg_pool: List[torch.Tensor], length: int, sample_seed: int) -> torch.Tensor:
    rng = np.random.default_rng(sample_seed)
    bg = bg_pool[int(rng.integers(0, len(bg_pool)))]
    if bg.numel() >= length:
        max_start = int(bg.numel() - length)
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        return bg[start:start + length]
    reps = (length + int(bg.numel()) - 1) // int(bg.numel())
    return bg.repeat(reps)[:length]


def mix_at_snr(signal: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    ps = float(signal.pow(2).mean().item())
    pn = float(noise.pow(2).mean().item())
    if pn <= 1e-12:
        return signal
    target_ratio = 10.0 ** (float(snr_db) / 10.0)
    scale = np.sqrt(max(ps, 1e-12) / (pn * target_ratio))
    return (signal + float(scale) * noise).clamp(-1.0, 1.0)


def build_mel_pipeline(cfg):
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

    def wav_tensor_to_mel(y: torch.Tensor) -> np.ndarray:
        y = y.unsqueeze(0)
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
        return mel.squeeze(0).transpose(0, 1).cpu().numpy().astype(np.float32)

    return sr, dt_ms, wav_tensor_to_mel


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


def evaluate_and_collect_gates(
    model: ClassificationModel,
    gate_layers: List[Tuple[str, TorchS5]],
    items: List[Tuple[Path, int]],
    bg_pool: List[torch.Tensor],
    wav_to_mel,
    sample_rate: int,
    dt_ms: float,
    snr_db: Optional[float],
    batch_size: int,
    seed: int,
    device: torch.device,
):
    sums = {name: 0.0 for name, _ in gate_layers}
    sqs = {name: 0.0 for name, _ in gate_layers}
    cnts = {name: 0 for name, _ in gate_layers}
    correct = 0
    total = 0
    skey = snr_token(snr_db)

    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            mels: List[np.ndarray] = []
            ys: List[int] = []
            for j, (wav_path, y_idx) in enumerate(chunk):
                idx = start + j
                y, orig_sr = torchaudio.load(str(wav_path))
                if y.dim() == 2 and y.shape[0] > 1:
                    y = y.mean(dim=0, keepdim=True)
                y = y.squeeze(0)
                if int(orig_sr) != int(sample_rate):
                    y = torchaudio.functional.resample(y.unsqueeze(0), int(orig_sr), int(sample_rate)).squeeze(0)
                if snr_db is not None:
                    noise = pick_noise_segment(bg_pool, int(y.numel()), int(seed + 7919 * idx + skey))
                    y = mix_at_snr(y, noise, snr_db=float(snr_db))
                mels.append(wav_to_mel(y))
                ys.append(int(y_idx))

            x, t, l = pad_batch(mels, dt_ms=dt_ms, device=device)
            logits = model(x, t, l, train=False)
            pred = logits.argmax(dim=-1).cpu().numpy()
            y_true = np.array(ys, dtype=np.int64)
            correct += int((pred == y_true).sum())
            total += int(y_true.shape[0])

            for name, layer in gate_layers:
                g = getattr(layer, "vad_gate", None)
                if g is None:
                    continue
                gv = g.detach().float()
                sums[name] += float(gv.sum().item())
                sqs[name] += float((gv * gv).sum().item())
                cnts[name] += int(gv.numel())

    stats = {}
    for name, _ in gate_layers:
        n = max(1, cnts[name])
        mean = sums[name] / n
        var = max(0.0, sqs[name] / n - mean * mean)
        stats[name] = {"mean": mean, "std": float(np.sqrt(var)), "count": cnts[name]}
    acc = 100.0 * float(correct) / max(1, total)
    return acc, stats


def main():
    parser = argparse.ArgumentParser(description="Diagnose gate statistics vs SNR on a fixed checkpoint.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--snrs", type=str, default="clean,20,10,5,0,-5")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-json", type=str, default="")
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(args.run_dir).resolve()
    cfg = OmegaConf.load(str(run_dir / "config.yaml"))
    ckpt = Path(args.checkpoint).resolve() if args.checkpoint else (run_dir / "checkpoints" / "model.pt")
    state = torch.load(str(ckpt), map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    num_classes = int(state["decoder.weight"].shape[0])

    model = build_model_from_config(cfg, num_classes=num_classes, device=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    gate_layers = [(name, m) for name, m in model.named_modules() if isinstance(m, TorchS5) and getattr(m, "input_gate", False)]
    if not gate_layers:
        raise RuntimeError("No TorchS5 input_gate layers found in this model.")

    root = Path(str(cfg.training.root)).resolve()
    items, labels = build_test_items(root)
    if args.max_test_samples > 0:
        items = items[: int(args.max_test_samples)]
    sr, dt_ms, wav_to_mel = build_mel_pipeline(cfg)
    bg_pool = load_background_pool(root / "_background_noise_", target_sr=sr)
    snrs = parse_snr_list(args.snrs)

    print(f"[*] Device: {device}")
    print(f"[*] Run: {run_dir}")
    print(f"[*] Checkpoint: {ckpt}")
    print(f"[*] Test samples: {len(items)} | classes: {len(labels)}")
    print(f"[*] Gated layers: {[n for n, _ in gate_layers]}")
    print(f"[*] SNRs: {snrs}")

    results = []
    for snr in snrs:
        acc, per_layer = evaluate_and_collect_gates(
            model=model,
            gate_layers=gate_layers,
            items=items,
            bg_pool=bg_pool,
            wav_to_mel=wav_to_mel,
            sample_rate=sr,
            dt_ms=dt_ms,
            snr_db=snr,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
        )
        layer_means = [v["mean"] for v in per_layer.values() if v["count"] > 0]
        global_mean = float(np.mean(layer_means)) if layer_means else float("nan")
        row = {
            "snr_db": "clean" if snr is None else float(snr),
            "accuracy": acc,
            "global_gate_mean": global_mean,
            "per_layer_gate": per_layer,
        }
        results.append(row)
        print(f"[snr={row['snr_db']}] acc={acc:.3f}% | global_gate_mean={global_mean:.6f}")

    out = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt),
        "snrs": ["clean" if s is None else float(s) for s in snrs],
        "num_test_samples": len(items),
        "results": results,
    }
    out_path = Path(args.out_json).resolve() if args.out_json else (run_dir / f"gate_vs_snr_{args.seed}.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[*] Saved: {out_path}")


if __name__ == "__main__":
    main()
