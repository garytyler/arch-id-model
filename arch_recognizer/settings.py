import logging
from pathlib import Path

APP_NAME: str = "arch-recognizer"
BASE_DIR: Path = Path(__file__).parent.parent.absolute()
SOURCE_DIR: Path = BASE_DIR / "dataset"
INPUT_DIR: Path = BASE_DIR / "input"
OUTPUT_DIR: Path = BASE_DIR / "output"
CHECKPOINTS_DIR: Path = OUTPUT_DIR / "checkpoints"
LOGS_DIR: Path = OUTPUT_DIR / "logs"
TB_DIR: Path = OUTPUT_DIR / "tb"
DATE_FORMAT: str = r"%Y-%m-%d-%H:%M:%S"
DEFAULT_LOG_LEVEL = logging.INFO
