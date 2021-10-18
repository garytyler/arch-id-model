import logging
from pathlib import Path

APP_NAME: str = "arch-recognizer"
TIMESTAMP_FORMAT: str = r"%Y-%m-%d-%H:%M:%S"
# TIMESTAMP_FORMAT: str = r"%Y-%m-%d-%H:%M:%S.%f"
DEFAULT_LOG_LEVEL = logging.INFO
SEED = 123456
BASE_DIR: Path = Path(__file__).parent.parent.absolute()
DATASET_DIR: Path = BASE_DIR / "dataset"
