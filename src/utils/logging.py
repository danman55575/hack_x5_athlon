import logging
import sys
from pathlib import Path
from datetime import datetime

def get_logger(name: str, log_dir: Path | str = "experiments/logs", level=logging.INFO) -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = log_dir / f"{name}_{ts}.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(logfile, encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)
    logger.propagate = False
    logger.info(f"Logging to {logfile}")
    return logger
