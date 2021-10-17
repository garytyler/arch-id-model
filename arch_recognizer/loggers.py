import datetime
import logging
import sys
from typing import Union

import tensorflow as tf

from . import settings

app_log_formatter = logging.Formatter(
    fmt="[%(asctime)s.%(msecs)03d][%(name)s][%(levelname)s] %(message)s",
    datefmt=settings.LOG_DATE_FORMAT,
)


def initialize_loggers(app_log_level: Union[str, int], tf_log_level: Union[str, int]):
    app_log_level = (
        app_log_level.upper() if isinstance(app_log_level, str) else app_log_level
    )
    tf_log_level = (
        tf_log_level.upper() if isinstance(tf_log_level, str) else tf_log_level
    )

    session_log_path = (
        settings.PY_LOGS_DIR
        / f"{datetime.datetime.now().strftime(settings.TIMESTAMP_FORMAT)}.log"
    )
    session_log_path.parent.mkdir(parents=True, exist_ok=True)
    session_log_file_handler = logging.FileHandler(session_log_path)

    # Configure arch-recognizer logger
    app_log_stream_handler = logging.StreamHandler(sys.stdout)
    app_log_stream_handler.setFormatter(app_log_formatter)
    app_log = logging.getLogger(settings.APP_NAME)
    app_log.propagate = False  # https://stackoverflow.com/a/33664610
    app_log.setLevel(app_log_level)
    app_log.addHandler(app_log_stream_handler)
    app_log.addHandler(session_log_file_handler)

    # Configure tensorflow logger
    tf_log_stream_handler = logging.StreamHandler(sys.stdout)
    tf_log_stream_handler.setFormatter(app_log_formatter)
    tf_log = tf.get_logger()
    tf_log.propagate = False  # https://stackoverflow.com/a/33664610
    tf_log.setLevel(tf_log_level)
    tf_log.addHandler(tf_log_stream_handler)
    tf_log.addHandler(session_log_file_handler)
