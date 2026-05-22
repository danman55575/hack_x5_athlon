"""Сборка отчёта report.txt по всем экспериментам."""
from __future__ import annotations
import json
import pandas as pd
from pathlib import Path
from datetime import datetime


def save_best_models(rows: list[dict], out_path: str | Path = "experiments/reports/best_models.csv") -> None:
    """Save top 10 best models to CSV file.
    
    Reads existing best_models.csv, adds new rows, sorts by CV MAPE (ascending),
    and keeps only the top 10 models.
    
    Args:
        rows: List of experiment result dictionaries (already sorted by CV MAPE)
        out_path: Path to save best_models.csv
    """
    out_path = Path(out_path)
    
    # Read existing best models if file exists
    if out_path.exists():
        df_existing = pd.read_csv(out_path)
    else:
        df_existing = pd.DataFrame()
    
    # Create DataFrame from new rows
    if rows:
        df_new = pd.DataFrame([
            {
                'name': r.get('name', ''),
                'model': r.get('model', ''),
                'cv_mape': r.get('cv_mean_mape', float('nan')),
                'cv_score': r.get('cv_mean_score', float('nan')),
                'feature_count': r.get('feature_count', 0),
                'elapsed_seconds': r.get('elapsed_seconds', 0),
                'timestamp': r.get('timestamp', ''),
                'submission_path': r.get('submission_path', ''),
            }
            for r in rows
        ])
    else:
        df_new = pd.DataFrame()
    
    # Combine DataFrames
    if not df_existing.empty and not df_new.empty:
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
    elif not df_new.empty:
        df_all = df_new.copy()
    else:
        df_all = df_existing.copy()
    
    # Remove duplicates (keep first occurrence by name)
    df_all = df_all.drop_duplicates(subset=['name'], keep='first')
    
    # Sort by CV MAPE (ascending - lower is better)
    df_all = df_all.sort_values('cv_mape').reset_index(drop=True)
    
    # Keep only top 10
    df_top_10 = df_all.head(10).copy()
    
    # Add rank
    df_top_10.insert(0, 'rank', range(1, len(df_top_10) + 1))
    
    # Save to CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_top_10.to_csv(out_path, index=False, encoding='utf-8')


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

    # Save top 10 models to CSV
    best_models_path = Path(reports_dir).parent / "best_models.csv"
    save_best_models(rows, best_models_path)
    
    # Load best models ranking
    best_models_ranking = {}
    if best_models_path.exists():
        try:
            df_best = pd.read_csv(best_models_path)
            best_models_ranking = dict(zip(df_best['name'], df_best['rank']))
        except Exception:
            pass

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
        
        # Add ranking info
        model_name = r.get('name', '')
        if model_name in best_models_ranking:
            rank = best_models_ranking[model_name]
            lines.append(f"  ★ Rank: #{rank} overall")
        else:
            lines.append(f"  ✗ Not in top 10 overall")
        
        for f in r.get("cv_folds", []):
            lines.append(f"    fold {f['fold']:<8} MAPE={f['mape']:.4f} score={f['score']:.3f} "
                         f"n_train={f['n_train']} n_val={f['n_val']}")
        lines.append(f"  submission: {r.get('submission_path', '')}")
    text = "\n".join(lines)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text
