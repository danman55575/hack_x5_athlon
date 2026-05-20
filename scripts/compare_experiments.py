"""Просто пересоздаёт REPORT.txt из всех json в experiments/reports."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.report import generate_report

if __name__ == "__main__":
    text = generate_report()
    print(text)
