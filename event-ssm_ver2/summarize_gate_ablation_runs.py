import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_acc_from_log(log_path: Path) -> float:
    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    m = re.findall(r"\[\*\]\s*Test accuracy:\s*([0-9.]+)%", txt)
    if not m:
        raise RuntimeError(f"Missing test accuracy in {log_path}")
    return float(m[-1])


def collect(group: str, logs: List[Path]) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for p in sorted(logs):
        acc = parse_acc_from_log(p)
        out.append((p.name, acc))
    if len(out) == 0:
        raise RuntimeError(f"No logs found for group {group}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Summarize 5x gate/no-gate run accuracies.")
    ap.add_argument("--gate-glob", type=str, required=True)
    ap.add_argument("--nogate-glob", type=str, required=True)
    ap.add_argument("--out-json", type=str, default="")
    args = ap.parse_args()

    gate_logs = [Path(p) for p in sorted(Path().glob(args.gate_glob))]
    nogate_logs = [Path(p) for p in sorted(Path().glob(args.nogate_glob))]

    gate = collect("with_gate", gate_logs)
    nogate = collect("no_gate", nogate_logs)

    gate_vals = np.array([x[1] for x in gate], dtype=np.float64)
    nogate_vals = np.array([x[1] for x in nogate], dtype=np.float64)

    summary: Dict = {
        "with_gate": {
            "n": int(gate_vals.size),
            "mean": float(gate_vals.mean()),
            "std": float(gate_vals.std(ddof=1)) if gate_vals.size > 1 else 0.0,
            "runs": [{"log": k, "test_acc": v} for k, v in gate],
        },
        "no_gate": {
            "n": int(nogate_vals.size),
            "mean": float(nogate_vals.mean()),
            "std": float(nogate_vals.std(ddof=1)) if nogate_vals.size > 1 else 0.0,
            "runs": [{"log": k, "test_acc": v} for k, v in nogate],
        },
        "delta_mean_points": float(gate_vals.mean() - nogate_vals.mean()),
    }

    print(json.dumps(summary, indent=2))
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[*] Saved: {out}")


if __name__ == "__main__":
    main()
