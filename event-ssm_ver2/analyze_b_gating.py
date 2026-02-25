import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import torch
from omegaconf import OmegaConf

from event_ssm.layers import TorchS5
from run_training import setup_training


def _summary_stats(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "p10": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


class GateCollector:
    """
    Collect per-sample gate means from every TorchS5 layer.
    Gate value = sigmoid(input_gate_net(u_raw)).
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.layer_values: Dict[str, List[np.ndarray]] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, TorchS5) and getattr(module, "input_gate", False):
                gate_net = getattr(module, "input_gate_net", None)
                if gate_net is None:
                    continue
                layer_name = name

                def _hook(_mod, _inp, out, lname=layer_name):
                    gate = torch.sigmoid(out.detach())
                    per_sample = gate.mean(dim=(1, 2)).cpu().numpy()
                    if lname not in self.layer_values:
                        self.layer_values[lname] = []
                    self.layer_values[lname].append(per_sample)

                self.handles.append(gate_net.register_forward_hook(_hook))

    def clear(self):
        self.layer_values = {}

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def finalize(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        by_layer: Dict[str, np.ndarray] = {}
        for k, v in self.layer_values.items():
            by_layer[k] = np.concatenate(v, axis=0) if len(v) > 0 else np.array([], dtype=np.float32)
        if len(by_layer) == 0:
            return by_layer, np.array([], dtype=np.float32)
        stacked = np.stack([arr for arr in by_layer.values()], axis=1)  # [N, num_layers]
        global_per_sample = stacked.mean(axis=1)
        return by_layer, global_per_sample


def _to_tensors(batch, device):
    inputs, targets, timesteps, lengths = batch
    x = torch.from_numpy(inputs).to(device)
    y = torch.from_numpy(targets).argmax(dim=-1).to(device)
    t = torch.from_numpy(timesteps).to(device)
    l = torch.from_numpy(lengths).to(device)
    return x, y, t, l


def _collect_keyword_stats(
    model: torch.nn.Module,
    loader,
    collector: GateCollector,
    device: torch.device,
    max_samples: int,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, float, float, List[int], int]:
    model.eval()
    n = 0
    sum_v = 0.0
    sum_sq_v = 0.0
    num_v = 0
    length_pool: List[int] = []
    mel_bins = -1

    collector.clear()
    with torch.no_grad():
        for batch in loader:
            x, _y, t, l = _to_tensors(batch, device)
            if mel_bins < 0:
                mel_bins = int(x.shape[-1])
            model(x, t, l, train=False)

            # Aggregate keyword mel stats over valid (unpadded) frames
            lengths_np = l.cpu().numpy().astype(np.int32)
            x_np = x.detach().cpu().numpy()
            for i, li in enumerate(lengths_np):
                li = int(max(0, li))
                if li == 0:
                    continue
                vals = x_np[i, :li, :]
                sum_v += float(vals.sum())
                sum_sq_v += float((vals ** 2).sum())
                num_v += int(vals.size)
                length_pool.append(li)

            n += int(x.shape[0])
            if n >= max_samples:
                break

    by_layer, global_vals = collector.finalize()
    if num_v == 0:
        mean = 0.0
        std = 1.0
    else:
        mean = sum_v / float(num_v)
        var = max(0.0, sum_sq_v / float(num_v) - mean * mean)
        std = float(np.sqrt(var))
        std = std if std > 1e-6 else 1e-6
    return by_layer, global_vals, float(mean), float(std), length_pool, mel_bins


def _wav_to_mel_frames(
    wav_path: Path,
    sr: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    mel_bins: int,
    use_log_mel: bool,
    top_db: float,
) -> np.ndarray:
    y, _ = librosa.load(str(wav_path), sr=sr, mono=True)
    mel_power = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels=mel_bins, power=2.0
    )
    if use_log_mel:
        mel_db = librosa.power_to_db(mel_power, ref=np.max, top_db=float(top_db))
        mel = ((mel_db + float(top_db)) / float(top_db)).astype(np.float32)
    else:
        max_val = float(np.max(mel_power)) if mel_power.size > 0 else 0.0
        if max_val <= 0.0 or not np.isfinite(max_val):
            mel = np.zeros_like(mel_power, dtype=np.float32)
        else:
            mel = (mel_power / (max_val + 1e-8)).astype(np.float32)
    return mel.T  # [L, mel_bins]


def _build_background_mel_cache(
    bg_dir: Path,
    sr: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    mel_bins: int,
    use_log_mel: bool,
    top_db: float,
) -> List[np.ndarray]:
    wavs = sorted(bg_dir.glob("*.wav"))
    if len(wavs) == 0:
        raise FileNotFoundError(f"No background wav found in: {bg_dir}")
    cache = []
    for w in wavs:
        m = _wav_to_mel_frames(
            w, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            mel_bins=mel_bins, use_log_mel=use_log_mel, top_db=top_db
        )
        if m.shape[0] > 0:
            cache.append(m)
    if len(cache) == 0:
        raise RuntimeError("Background mel cache is empty after loading wav files.")
    return cache


def _sample_bg_segment(mel: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    L = int(mel.shape[0])
    if L >= length:
        start = int(rng.integers(0, max(1, L - length + 1)))
        return mel[start:start + length, :]
    out = np.zeros((length, mel.shape[1]), dtype=np.float32)
    out[:L, :] = mel
    return out


def _background_batch_from_lengths(
    lengths: List[int],
    bg_mels: List[np.ndarray],
    pad_unit: int,
    dt_ms: float,
    device: torch.device,
    rng: np.random.Generator,
):
    B = len(lengths)
    mel_bins = int(bg_mels[0].shape[1])
    max_len = max(1, max(lengths))
    pad_length = (max_len // pad_unit) * pad_unit + pad_unit
    x = np.zeros((B, pad_length, mel_bins), dtype=np.float32)
    t = np.zeros((B, pad_length), dtype=np.float32)
    l = np.array(lengths, dtype=np.int32)
    for i, li in enumerate(lengths):
        m = bg_mels[int(rng.integers(0, len(bg_mels)))]
        seg = _sample_bg_segment(m, int(li), rng)
        x[i, :li, :] = seg
        t[i, :li] = dt_ms
    return torch.from_numpy(x).to(device), torch.from_numpy(t).to(device), torch.from_numpy(l).to(device)


def _collect_background_stats(
    model: torch.nn.Module,
    collector: GateCollector,
    device: torch.device,
    total_samples: int,
    batch_size: int,
    length_pool: List[int],
    bg_mels: List[np.ndarray],
    pad_unit: int,
    dt_ms: float,
    rng: np.random.Generator,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    if len(length_pool) == 0:
        raise RuntimeError("No keyword lengths collected; cannot build matched-background batches.")

    model.eval()
    collector.clear()
    made = 0
    with torch.no_grad():
        while made < total_samples:
            cur = min(batch_size, total_samples - made)
            sampled_lengths = list(rng.choice(length_pool, size=cur, replace=True).astype(np.int32))
            x, t, l = _background_batch_from_lengths(
                sampled_lengths, bg_mels, pad_unit, dt_ms, device, rng
            )
            model(x, t, l, train=False)
            made += cur

    return collector.finalize()


def _print_block(title: str, by_layer: Dict[str, np.ndarray], global_vals: np.ndarray):
    print(f"\n=== {title} ===")
    print(f"GLOBAL {json.dumps(_summary_stats(global_vals), ensure_ascii=True)}")
    for layer_name in sorted(by_layer.keys()):
        print(f"{layer_name}: {json.dumps(_summary_stats(by_layer[layer_name]), ensure_ascii=True)}")


def main():
    parser = argparse.ArgumentParser(description="Analyze B-gating values for keyword vs noise inputs.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default="/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2/outputs/42557048_gsc_train_mamba",
        help="Run directory that contains config.yaml and checkpoints/model.pt",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Optional checkpoint path override. Default: <run-dir>/checkpoints/model.pt",
    )
    parser.add_argument("--max-keyword-samples", type=int, default=2048)
    parser.add_argument("--background-samples", type=int, default=2048)
    parser.add_argument("--background-batch-size", type=int, default=64)
    parser.add_argument("--background-rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--analysis-pad-unit", type=int, default=1024)
    parser.add_argument(
        "--background-dir",
        type=str,
        default="",
        help="Optional override path to GSC _background_noise_ directory.",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="",
        help="Optional output json path",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    config_path = run_dir / "config.yaml"
    ckpt_path = Path(args.checkpoint).resolve() if args.checkpoint else (run_dir / "checkpoints" / "model.pt")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cfg = OmegaConf.load(str(config_path))
    # Ensure gate config is passed into model creation path.
    if "ssm" in cfg.model:
        cfg.model.ssm.input_gate = bool(cfg.model.ssm.get("input_gate", True))
        cfg.model.ssm.input_gate_rank = int(cfg.model.ssm.get("input_gate_rank", 0))
    # Lightweight analysis overrides to avoid OOM on login nodes.
    cfg.training.per_device_eval_batch_size = int(args.eval_batch_size)
    cfg.training.per_device_batch_size = int(min(int(args.eval_batch_size), int(cfg.training.get("per_device_batch_size", args.eval_batch_size))))
    cfg.training.pad_unit = int(args.analysis_pad_unit)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _train_loader, _val_loader, test_loader, _data = setup_training(cfg)
    model.to(device)
    state = torch.load(str(ckpt_path), map_location=device)
    load_res = model.load_state_dict(state["state_dict"], strict=False)
    missing = set(load_res.missing_keys)
    unexpected = set(load_res.unexpected_keys)
    allowed_missing = {"attn_pool.weight", "attn_pool.bias"}
    if len(unexpected) > 0 or len(missing - allowed_missing) > 0:
        raise RuntimeError(
            f"Unexpected checkpoint mismatch. missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    print(f"[*] Loaded checkpoint: {ckpt_path}")
    print(f"[*] Device: {device}")

    collector = GateCollector(model)
    if len(collector.handles) == 0:
        collector.close()
        raise RuntimeError("No gated TorchS5 layers found. Check input_gate setting/model checkpoint.")

    keyword_by_layer, keyword_global, keyword_mean, keyword_std, length_pool, mel_bins = _collect_keyword_stats(
        model=model,
        loader=test_loader,
        collector=collector,
        device=device,
        max_samples=int(args.max_keyword_samples),
    )

    sr = int(cfg.training.get("sr", 16000))
    n_fft = int(cfg.training.get("n_fft", 1024))
    hop_length = int(cfg.training.get("hop_length", 160))
    win_length = int(cfg.training.get("win_length", 400))
    top_db = float(cfg.training.get("top_db", 80.0))
    use_log_mel = bool(cfg.training.get("use_log_mel", True))
    dt_ms = float(cfg.training.get("hop_length", 160)) / float(cfg.training.get("sr", 16000)) * 1000.0
    pad_unit = int(cfg.training.get("pad_unit", 8192))
    if args.background_dir:
        bg_dir = Path(args.background_dir).resolve()
    else:
        bg_dir = Path(cfg.training.root).resolve() / "_background_noise_"
    bg_mels = _build_background_mel_cache(
        bg_dir=bg_dir,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        mel_bins=mel_bins,
        use_log_mel=use_log_mel,
        top_db=top_db,
    )
    print(f"[*] Background dir: {bg_dir} | files: {len(bg_mels)}")

    rounds = int(max(1, args.background_rounds))
    rng_master = np.random.default_rng(int(args.seed))
    round_results = []
    all_bg_global = []
    bg_by_layer_acc: Dict[str, List[np.ndarray]] = {}
    for ridx in range(rounds):
        rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        bg_by_layer, bg_global = _collect_background_stats(
            model=model,
            collector=collector,
            device=device,
            total_samples=int(args.background_samples),
            batch_size=int(args.background_batch_size),
            length_pool=length_pool,
            bg_mels=bg_mels,
            pad_unit=pad_unit,
            dt_ms=dt_ms,
            rng=rng,
        )
        round_results.append({
            "round_idx": ridx,
            "background_global": _summary_stats(bg_global),
            "background_per_layer": {k: _summary_stats(v) for k, v in bg_by_layer.items()},
        })
        all_bg_global.append(bg_global)
        for k, v in bg_by_layer.items():
            if k not in bg_by_layer_acc:
                bg_by_layer_acc[k] = []
            bg_by_layer_acc[k].append(v)

    background_global = np.concatenate(all_bg_global, axis=0) if len(all_bg_global) > 0 else np.array([], dtype=np.float32)
    background_by_layer = {k: np.concatenate(v, axis=0) for k, v in bg_by_layer_acc.items()}
    collector.close()

    _print_block("Keyword(test set)", keyword_by_layer, keyword_global)
    _print_block("Background noise(real GSC _background_noise_)", background_by_layer, background_global)

    result = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "background_dir": str(bg_dir),
        "keyword_global": _summary_stats(keyword_global),
        "background_global": _summary_stats(background_global),
        "keyword_per_layer": {k: _summary_stats(v) for k, v in keyword_by_layer.items()},
        "background_per_layer": {k: _summary_stats(v) for k, v in background_by_layer.items()},
        "background_rounds": round_results,
        "keyword_mel_distribution": {
            "mean": keyword_mean,
            "std": keyword_std,
            "num_length_samples": len(length_pool),
        },
    }

    if args.out_json:
        out_path = Path(args.out_json).resolve()
    else:
        out_path = run_dir / "gating_keyword_vs_noise_stats.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[*] Saved report: {out_path}")


if __name__ == "__main__":
    main()
