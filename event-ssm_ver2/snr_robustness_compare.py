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


def load_run(run_dir: Path, checkpoint: Optional[Path], device: torch.device):
    cfg = OmegaConf.load(str(run_dir / "config.yaml"))
    ckpt_path = checkpoint if checkpoint is not None else (run_dir / "checkpoints" / "model.pt")
    state = torch.load(str(ckpt_path), map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    num_classes = int(state["decoder.weight"].shape[0])
    model = build_model_from_config(cfg, num_classes=num_classes, device=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return cfg, model, ckpt_path


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
        rel_path = Path(rel)
        if len(rel_path.parts) < 2:
            continue
        lab = rel_path.parts[0]
        if lab not in label_to_idx:
            continue
        wav_path = root / rel
        if wav_path.exists():
            items.append((wav_path, label_to_idx[lab]))
    return items, labels


def load_background_pool(bg_dir: Path, target_sr: int) -> List[torch.Tensor]:
    bg_wavs = sorted(bg_dir.glob("*.wav"))
    if len(bg_wavs) == 0:
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


def pick_noise_segment(
    bg_pool: List[torch.Tensor],
    length: int,
    sample_seed: int,
) -> torch.Tensor:
    rng = np.random.default_rng(sample_seed)
    bg = bg_pool[int(rng.integers(0, len(bg_pool)))]
    if bg.numel() >= length:
        max_start = int(bg.numel() - length)
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        return bg[start:start + length]
    reps = (length + int(bg.numel()) - 1) // int(bg.numel())
    tiled = bg.repeat(reps)
    return tiled[:length]


def mix_at_snr(signal: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    ps = float(signal.pow(2).mean().item())
    pn = float(noise.pow(2).mean().item())
    if pn <= 1e-12:
        return signal
    target_ratio = 10.0 ** (float(snr_db) / 10.0)
    scale = np.sqrt(max(ps, 1e-12) / (pn * target_ratio))
    mixed = signal + float(scale) * noise
    return mixed.clamp(-1.0, 1.0)


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


def evaluate_model_at_snr(
    model: ClassificationModel,
    items: List[Tuple[Path, int]],
    bg_pool: List[torch.Tensor],
    wav_to_mel,
    sample_rate: int,
    dt_ms: float,
    snr_db: Optional[float],
    batch_size: int,
    seed: int,
    device: torch.device,
) -> float:
    correct = 0
    total = 0
    snr_key = snr_token(snr_db)

    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            mels: List[np.ndarray] = []
            ys: List[int] = []
            for j, (wav_path, y_idx) in enumerate(chunk):
                global_idx = start + j
                y, orig_sr = torchaudio.load(str(wav_path))
                if y.dim() == 2 and y.shape[0] > 1:
                    y = y.mean(dim=0, keepdim=True)
                y = y.squeeze(0)
                if int(orig_sr) != int(sample_rate):
                    y = torchaudio.functional.resample(y.unsqueeze(0), int(orig_sr), int(sample_rate)).squeeze(0)

                if snr_db is not None:
                    noise = pick_noise_segment(
                        bg_pool=bg_pool,
                        length=int(y.numel()),
                        sample_seed=int(seed + 7919 * global_idx + snr_key),
                    )
                    y = mix_at_snr(y, noise, snr_db=float(snr_db))

                mel = wav_to_mel(y)
                mels.append(mel)
                ys.append(int(y_idx))

            x, t, l = pad_batch(mels, dt_ms=dt_ms, device=device)
            logits = model(x, t, l, train=False)
            pred = logits.argmax(dim=-1).cpu().numpy()
            y_true = np.array(ys, dtype=np.int64)
            correct += int((pred == y_true).sum())
            total += int(y_true.shape[0])

    return 100.0 * float(correct) / max(1, total)


def main():
    parser = argparse.ArgumentParser(description="Compare two checkpoints under SNR noise on GSC test set.")
    parser.add_argument("--run-a", type=str, required=True, help="Run dir A (e.g., no gate)")
    parser.add_argument("--run-b", type=str, required=True, help="Run dir B (e.g., with gate)")
    parser.add_argument("--name-a", type=str, default="no_gate")
    parser.add_argument("--name-b", type=str, default="with_gate")
    parser.add_argument("--checkpoint-a", type=str, default="")
    parser.add_argument("--checkpoint-b", type=str, default="")
    parser.add_argument("--snrs", type=str, default="clean,20,10,5,0,-5")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-test-samples", type=int, default=0, help="0 means full test set")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-json", type=str, default="")
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_a = Path(args.run_a).resolve()
    run_b = Path(args.run_b).resolve()
    ckpt_a = Path(args.checkpoint_a).resolve() if args.checkpoint_a else None
    ckpt_b = Path(args.checkpoint_b).resolve() if args.checkpoint_b else None

    cfg_a, model_a, loaded_ckpt_a = load_run(run_a, ckpt_a, device)
    cfg_b, model_b, loaded_ckpt_b = load_run(run_b, ckpt_b, device)

    root_a = Path(str(cfg_a.training.root)).resolve()
    root_b = Path(str(cfg_b.training.root)).resolve()
    if root_a != root_b:
        raise RuntimeError(f"Two runs use different dataset roots: {root_a} vs {root_b}")
    data_root = root_a
    bg_dir = data_root / "_background_noise_"

    sr_a, dt_ms_a, wav_to_mel_a = build_mel_pipeline(cfg_a)
    sr_b, dt_ms_b, wav_to_mel_b = build_mel_pipeline(cfg_b)
    if int(sr_a) != int(sr_b):
        raise RuntimeError(f"Sample rate mismatch: {sr_a} vs {sr_b}")
    if abs(float(dt_ms_a) - float(dt_ms_b)) > 1e-9:
        raise RuntimeError(f"Hop timestep mismatch: {dt_ms_a} vs {dt_ms_b}")

    items, labels = build_test_items(data_root)
    if args.max_test_samples > 0:
        items = items[: int(args.max_test_samples)]
    if len(items) == 0:
        raise RuntimeError("No valid test items found from testing_list.txt")

    bg_pool = load_background_pool(bg_dir=bg_dir, target_sr=sr_a)
    snr_list = parse_snr_list(args.snrs)

    print(f"[*] Device: {device}")
    print(f"[*] Test samples: {len(items)} | classes: {len(labels)}")
    print(f"[*] Run A: {run_a} | ckpt: {loaded_ckpt_a}")
    print(f"[*] Run B: {run_b} | ckpt: {loaded_ckpt_b}")
    print(f"[*] SNRs: {snr_list}")

    results: List[Dict] = []
    for snr in snr_list:
        acc_a = evaluate_model_at_snr(
            model=model_a,
            items=items,
            bg_pool=bg_pool,
            wav_to_mel=wav_to_mel_a,
            sample_rate=sr_a,
            dt_ms=dt_ms_a,
            snr_db=snr,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
        )
        acc_b = evaluate_model_at_snr(
            model=model_b,
            items=items,
            bg_pool=bg_pool,
            wav_to_mel=wav_to_mel_b,
            sample_rate=sr_b,
            dt_ms=dt_ms_b,
            snr_db=snr,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
        )
        row = {
            "snr_db": "clean" if snr is None else float(snr),
            args.name_a: acc_a,
            args.name_b: acc_b,
            "delta_acc_points": acc_b - acc_a,
        }
        results.append(row)
        print(f"[snr={row['snr_db']}] {args.name_a}: {acc_a:.3f}% | {args.name_b}: {acc_b:.3f}% | delta: {acc_b - acc_a:+.3f}")

    clean_a = None
    clean_b = None
    for row in results:
        if row["snr_db"] == "clean":
            clean_a = row[args.name_a]
            clean_b = row[args.name_b]
            break
    if clean_a is not None and clean_b is not None:
        for row in results:
            row[f"{args.name_a}_drop_from_clean"] = clean_a - row[args.name_a]
            row[f"{args.name_b}_drop_from_clean"] = clean_b - row[args.name_b]

    out = {
        "run_a": str(run_a),
        "run_b": str(run_b),
        "name_a": args.name_a,
        "name_b": args.name_b,
        "checkpoint_a": str(loaded_ckpt_a),
        "checkpoint_b": str(loaded_ckpt_b),
        "dataset_root": str(data_root),
        "num_test_samples": len(items),
        "num_classes": len(labels),
        "snrs": ["clean" if s is None else float(s) for s in snr_list],
        "results": results,
    }

    out_path = Path(args.out_json).resolve() if args.out_json else (run_b / f"snr_compare_{args.seed}.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[*] Saved: {out_path}")


if __name__ == "__main__":
    main()
