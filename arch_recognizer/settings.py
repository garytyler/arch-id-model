import logging
from pathlib import Path

APP_NAME: str = "arch-recognizer"
DATE_FORMAT: str = r"%Y-%m-%d-%H:%M:%S.%f"
DEFAULT_LOG_LEVEL = logging.INFO

BASE_DIR: Path = Path(__file__).parent.parent.absolute()
DATASET_DIR: Path = BASE_DIR / "dataset"
OUTPUT_DIR: Path = BASE_DIR / "output"
CP_DIR: Path = OUTPUT_DIR / "cp"
PY_LOGS_DIR: Path = OUTPUT_DIR / "py"
TB_LOGS_DIR: Path = OUTPUT_DIR / "tb"
