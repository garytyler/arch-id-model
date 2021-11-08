import logging
from pathlib import Path

APP_NAME: str = "arch-recognizer"
TIMESTAMP_FORMAT: str = r"%Y-%m-%d-%H:%M:%S"
DEFAULT_LOG_LEVEL = logging.INFO
SEED = 123456
BASE_DIR: Path = Path(__file__).parent.parent.absolute()

# Metrics
WEIGHTS = ["imagenet", "none"]
