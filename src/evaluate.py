from __future__ import annotations
import json
from pathlib import Path


def save_metrics(m, path="models/metrics.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(m, f, indent=2)
    return m


def print_report(m):
    print("=" * 48)
    print("  DemandForecast — Evaluation Report")
    print("=" * 48)
    for k, v in m.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for sk, sv in v.items():
                if isinstance(sv, float):
                    print(f"    {sk:>20s}: {sv:.4f}")
                else:
                    print(f"    {sk:>20s}: {sv}")
        elif isinstance(v, float):
            print(f"  {k:>28s}: {v:.4f}")
        else:
            print(f"  {k:>28s}: {v}")
