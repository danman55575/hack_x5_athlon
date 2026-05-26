import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import run_experiment
from src.report import generate_report
from src.utils.io import load_yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--train", default="data/processed/v2.parquet")
    p.add_argument(
        "--feature-groups",
        nargs="+",
        default=None,
        help="Необязательный список групп фичей для абляций или полного прогона.",
    )
    p.add_argument(
        "--feature-list-path",
        default=None,
        help="Путь к txt/csv со списком выбранных фичей для точного воспроизведения полного прогона.",
    )
    p.add_argument(
        "--skip-final-train",
        action="store_true",
        help="Остановиться после CV и не строить финальный прогноз на март 2025.",
    )
    args = p.parse_args()

    cfg = load_yaml(args.config)
    if args.feature_groups:
        cfg["feature_groups"] = args.feature_groups
        cfg["name"] = f"{cfg['name']}_{'_'.join(args.feature_groups)}"
    if args.feature_list_path:
        cfg["feature_allowlist_path"] = args.feature_list_path
        cfg["name"] = f"{cfg['name']}_{Path(args.feature_list_path).stem}"
    if args.skip_final_train:
        cfg["skip_final_train"] = True

    res = run_experiment(cfg, train_path=args.train)
    print(f"\nCV MAPE: {res['cv_mean_mape']:.4f}  |  Score: {res['cv_mean_score']:.3f}")
    if res["submission_path"]:
        print(f"Submission: {res['submission_path']}")
    else:
        print("Финальный прогноз не строился (--skip-final-train).")
    generate_report()
    print("REPORT updated: experiments/reports/REPORT.txt")


if __name__ == "__main__":
    main()
