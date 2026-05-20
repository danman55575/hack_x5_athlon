"""Сборка отчёта report.txt по всем экспериментам."""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime


def generate_report(reports_dir: str | Path = "experiments/reports",
                    out_path: str | Path = "experiments/reports/REPORT.txt") -> str:
    reports_dir = Path(reports_dir)
    files = sorted(reports_dir.glob("*.json"))
    rows = []
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                r = json.load(f)
            rows.append(r)
        except Exception:
            continue
    rows.sort(key=lambda r: r.get("cv_mean_mape", float("inf")))

    lines = []
    lines.append("=" * 110)
    lines.append(f"  HACKATHON RTO — EXPERIMENT REPORT — generated at {datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append("=" * 110)
    lines.append(f"  Total experiments: {len(rows)}")
    lines.append("")

    lines.append(f"{'Rank':<5} {'Name':<35} {'Model':<12} {'CV MAPE':>10} {'CV Score':>10} {'Feats':>6} {'Time(s)':>9}")
    lines.append("-" * 110)
    for i, r in enumerate(rows, 1):
        lines.append(f"{i:<5} {r.get('name','')[:34]:<35} {r.get('model','')[:11]:<12} "
                     f"{r.get('cv_mean_mape', float('nan')):>10.4f} "
                     f"{r.get('cv_mean_score', float('nan')):>10.3f} "
                     f"{r.get('feature_count', 0):>6d} {r.get('elapsed_seconds', 0):>9.1f}")
    lines.append("")
    lines.append("=" * 110)
    lines.append("  PER-FOLD DETAILS")
    lines.append("=" * 110)
    for r in rows:
        lines.append("")
        lines.append(f"[{r.get('name')}] model={r.get('model')} ts={r.get('timestamp')}")
        lines.append(f"  params: {json.dumps(r.get('params', {}), ensure_ascii=False)}")
        lines.append(f"  CV mean MAPE: {r.get('cv_mean_mape', float('nan')):.4f}  |  Score: {r.get('cv_mean_score', float('nan')):.3f}")
        for f in r.get("cv_folds", []):
            lines.append(f"    fold {f['fold']:<8} MAPE={f['mape']:.4f} score={f['score']:.3f} "
                         f"n_train={f['n_train']} n_val={f['n_val']}")
        lines.append(f"  submission: {r.get('submission_path', '')}")
    text = "\n".join(lines)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text
