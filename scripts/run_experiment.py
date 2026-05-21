import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.io import load_yaml
from src.pipeline import run_experiment
from src.report import generate_report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--train", default="data/processed/v2.parquet")
    args = p.parse_args()
    cfg = load_yaml(args.config)
    res = run_experiment(cfg, train_path=args.train)
    print(f"\nCV MAPE: {res['cv_mean_mape']:.4f}  |  Score: {res['cv_mean_score']:.3f}")
    print(f"Submission: {res['submission_path']}")
    generate_report()
    print("REPORT updated: experiments/reports/REPORT.txt")


if __name__ == "__main__":
    main()
