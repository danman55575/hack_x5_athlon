"""Главный регулируемый скрипт. Запускает весь пайплайн или нужные стадии.
Примеры:
    python -m scripts.run_pipeline --stage all
    python -m scripts.run_pipeline --stage preprocess
    python -m scripts.run_pipeline --stage ets
    python -m scripts.run_pipeline --stage train --configs configs/lgbm.yaml configs/xgb.yaml
    python -m scripts.run_pipeline --stage stack
    python -m scripts.run_pipeline --stage blend
"""
import argparse, sys, subprocess, glob
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
        print(f"CV MAPE: {res['cv_mean_mape']:.4f}  Score: {res['cv_mean_score']:.3f}")
        results.append(res)
    return results


def _latest(glob_pat):
    files = sorted(glob.glob(glob_pat))
    return files[-1] if files else None


def stage_stack():
    """Стекинг по последним OOF / predictions всех моделей."""
    oof_files, pred_files = [], []
    for stem in ["lgbm_l1_log", "lgbm_mape_weight", "lgbm_tweedie",
                 "xgb_l1_log", "cat_l1_log", "ridge_log", "mlp_log"]:
        oof = _latest(f"experiments/oof/{stem}_*_oof.parquet")
        pred = _latest(f"experiments/predictions/{stem}_*.csv")
        if oof and pred:
            oof_files.append(oof); pred_files.append(pred)
            print(f"Use {stem}: OOF={Path(oof).name}, PRED={Path(pred).name}")
    if not oof_files:
        print("No OOF found"); return
    cmd = ["python", "-m", "scripts.stack_oof",
           "--oof", *oof_files,
           "--predictions", *pred_files,
           "--out", "data/submissions/stack_latest.csv"]
    subprocess.run(cmd, check=True)


def stage_blend():
    """Геометрический бленд лучших трёх моделей."""
    candidates = []
    for stem in ["lgbm_l1_log", "lgbm_mape_weight", "xgb_l1_log",
                 "cat_l1_log", "lgbm_tweedie"]:
        p = _latest(f"experiments/predictions/{stem}_*.csv")
        if p: candidates.append(p)
    if len(candidates) < 2:
        print("Not enough predictions"); return
    cmd = ["python", "-m", "scripts.make_submission",
           "--predictions", *candidates, "--mode", "gmean",
           "--out", "data/submissions/blend_latest.csv"]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "preprocess", "ets", "train", "stack", "blend"],
                    default="all")
    ap.add_argument("--configs", nargs="+",
                    default=["configs/lgbm.yaml", "configs/lgbm_mape.yaml",
                             "configs/lgbm_tweedie.yaml", "configs/xgb.yaml",
                             "configs/catboost.yaml"])
    ap.add_argument("--train_path", default="data/processed/v2.parquet")
    args = ap.parse_args()

    set_seed(2026)
    if args.stage in ("all", "preprocess"):
        print("\n>>> STAGE: preprocess"); stage_preprocess()
    if args.stage in ("all", "ets"):
        print("\n>>> STAGE: ets"); stage_ets()
    if args.stage in ("all", "train"):
        print("\n>>> STAGE: train"); stage_train(args.configs, args.train_path)
    if args.stage in ("all", "stack"):
        print("\n>>> STAGE: stack"); stage_stack()
    if args.stage in ("all", "blend"):
        print("\n>>> STAGE: blend"); stage_blend()

    print("\nALL DONE.")


if __name__ == "__main__":
    main()
