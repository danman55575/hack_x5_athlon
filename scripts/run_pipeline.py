"""Главный регулируемый скрипт. Запускает весь пайплайн или нужные стадии."""
import argparse, sys, subprocess, glob, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.utils.io import load_yaml
from src.utils.seed import set_seed


def stage_preprocess():
    from src.data import preprocess
    sys.argv = ["preprocess",
                "--in_path", "data/raw/train_2.csv",
                "--out_path", "data/processed/v2.parquet"]
    preprocess.main()


def stage_ets():
    from src.data.loader import load_raw, add_target_row_for_march_2025
    from src.models.ts_features import compute_ets_features
    df = load_raw("data/processed/v2.parquet")
    df = add_target_row_for_march_2025(df)
    feats = compute_ets_features(df, target="rto", n_jobs=-1, min_t_for_prediction=6)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    feats.to_parquet("data/processed/ets_features.parquet", index=False)
    print(f"Saved ETS features: {feats.shape}")


def stage_train(configs, train_path):
    from src.pipeline import run_experiment
    results = []
    for c in configs:
        cfg = load_yaml(c)
        print(f"\n========== {cfg['name']} ==========")
        res = run_experiment(cfg, train_path=train_path)
        print(f"CV weighted MAPE: {res['cv_mean_mape']:.4f}  "
              f"weighted Score: {res['cv_mean_score']:.3f}")
        results.append(res)
    return results


def _latest(glob_pat):
    files = sorted(glob.glob(glob_pat))
    return files[-1] if files else None


def _latest_report(stem):
    rep = _latest(f"experiments/reports/{stem}_*.json")
    if not rep:
        return None, None, None
    ts = Path(rep).stem[len(stem) + 1:]
    oof = f"experiments/oof/{stem}_{ts}_oof.parquet"
    pred = f"experiments/predictions/{stem}_{ts}.csv"
    if not Path(oof).exists():
        oof = _latest(f"experiments/oof/{stem}_*_oof.parquet")
    if not Path(pred).exists():
        pred = _latest(f"experiments/predictions/{stem}_*.csv")
    return rep, oof, pred


ENSEMBLE_STEMS = ["lgbm_l1_log", "lgbm_tweedie",
                  "xgb_l1_log", "cat_l1_log", "ridge_log", "mlp_log"]
BLEND_STEMS = ["lgbm_l1_log", "lgbm_tweedie", "xgb_l1_log", "cat_l1_log"]


def stage_stack(max_mape: float = 12.0):
    oof_files, pred_files, names = [], [], []
    for stem in ENSEMBLE_STEMS:
        rep, oof, pred = _latest_report(stem)
        if not (rep and oof and pred):
            continue
        with open(rep, encoding="utf-8") as f:
            r = json.load(f)
        cv_mape = r.get("cv_mean_mape", 100.0)
        if cv_mape > max_mape:
            print(f"SKIP {stem}: CV MAPE={cv_mape:.3f} > {max_mape}")
            continue
        oof_files.append(oof); pred_files.append(pred); names.append(stem)
        print(f"USE  {stem}: CV MAPE={cv_mape:.3f}")
    if not oof_files:
        print("No OOF found"); return
    cmd = ["python", "-m", "scripts.stack_oof",
           "--oof", *oof_files,
           "--predictions", *pred_files,
           "--out", "data/submissions/stack_latest.csv"]
    subprocess.run(cmd, check=True)


def stage_blend(max_mape: float = 12.0, power: float = 3.0):
    candidates, weights = [], []
    for stem in BLEND_STEMS:
        rep, _, pred = _latest_report(stem)
        if not (rep and pred):
            continue
        with open(rep, encoding="utf-8") as f:
            r = json.load(f)
        cv_mape = r.get("cv_mean_mape", 100.0)
        if cv_mape > max_mape:
            print(f"SKIP blend {stem}: CV MAPE={cv_mape:.3f} > {max_mape}")
            continue
        candidates.append(pred)
        weights.append(1.0 / max(cv_mape, 1.0) ** power)
        print(f"USE  blend {stem}: CV MAPE={cv_mape:.3f}, raw weight={weights[-1]:.4f}")
    if len(candidates) < 2:
        print("Not enough predictions for blend"); return
    weights = np.array(weights, dtype=np.float64); weights /= weights.sum()
    print(f"Normalized weights: {weights.tolist()}")
    cmd = ["python", "-m", "scripts.make_submission",
           "--predictions", *candidates,
           "--weights", *[str(w) for w in weights],
           "--mode", "gmean",
           "--out", "data/submissions/blend_latest.csv"]
    subprocess.run(cmd, check=True)


def stage_blend_with_stack():
    p1 = Path("data/submissions/stack_latest.csv")
    p2 = Path("data/submissions/blend_latest.csv")
    if not (p1.exists() and p2.exists()):
        print("Need both stack_latest.csv and blend_latest.csv; run stack and blend first")
        return
    cmd = ["python", "-m", "scripts.make_submission",
           "--predictions", str(p1), str(p2),
           "--weights", "0.5", "0.5", "--mode", "gmean",
           "--out", "data/submissions/super_blend.csv"]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage",
                    choices=["all", "preprocess", "ets", "train",
                             "stack", "blend", "super"],
                    default="all")
    ap.add_argument("--configs", nargs="+",
                    default=["configs/lgbm.yaml",
                             "configs/lgbm_tweedie.yaml", "configs/xgb.yaml",
                             "configs/catboost.yaml"])
    ap.add_argument("--train_path", default="data/processed/v2.parquet")
    ap.add_argument("--max_mape", type=float, default=12.0)
    args = ap.parse_args()

    set_seed(2026)
    if args.stage in ("all", "preprocess"):
        print("\n>>> STAGE: preprocess"); stage_preprocess()
    if args.stage in ("all", "ets"):
        print("\n>>> STAGE: ets"); stage_ets()
    if args.stage in ("all", "train"):
        print("\n>>> STAGE: train"); stage_train(args.configs, args.train_path)
    if args.stage in ("all", "stack"):
        print("\n>>> STAGE: stack"); stage_stack(args.max_mape)
    if args.stage in ("all", "blend"):
        print("\n>>> STAGE: blend"); stage_blend(args.max_mape)
    if args.stage in ("all", "super"):
        print("\n>>> STAGE: super blend"); stage_blend_with_stack()

    print("\nALL DONE.")


if __name__ == "__main__":
    main()
