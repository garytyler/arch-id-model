import datetime
import logging
import sys
from pathlib import Path
from typing import Union

import tensorflow as tf

from .settings import APP_NAME, TIMESTAMP_FORMAT

app_log_formatter = logging.Formatter(
    fmt="[%(asctime)s.%(msecs)03d][%(name)s][%(levelname)s] %(message)s",
    datefmt=TIMESTAMP_FORMAT,
)


def initialize_loggers(
    app_log_level: Union[str, int], tf_log_level: Union[str, int], log_dir: Path = None
):
    app_log_level = (
        app_log_level.upper() if isinstance(app_log_level, str) else app_log_level
    )
    tf_log_level = (
        tf_log_level.upper() if isinstance(tf_log_level, str) else tf_log_level
    )

    # Configure app logger
    app_log_stream_handler = logging.StreamHandler(sys.stdout)
    app_log_stream_handler.setFormatter(app_log_formatter)
    app_log = logging.getLogger(APP_NAME)
    app_log.propagate = False  # https://stackoverflow.com/a/33664610
    app_log.setLevel(app_log_level)
    app_log.addHandler(app_log_stream_handler)

    # Configure tensorflow logger
    tf_log_stream_handler = logging.StreamHandler(sys.stdout)
    tf_log_stream_handler.setFormatter(app_log_formatter)
    tf_log = tf.get_logger()
    tf_log.propagate = False  # https://stackoverflow.com/a/33664610
    tf_log.setLevel(tf_log_level)
    tf_log.addHandler(tf_log_stream_handler)

    if log_dir:
        session_log_path = (
            log_dir / f"{datetime.datetime.now().strftime(TIMESTAMP_FORMAT)}.log"
        )
        session_log_path.parent.mkdir(parents=True, exist_ok=True)
        session_log_file_handler = logging.FileHandler(session_log_path)
        app_log.addHandler(session_log_file_handler)
        tf_log.addHandler(session_log_file_handler)
