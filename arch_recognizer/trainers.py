import logging
import os
import shutil
import tempfile
from pathlib import Path

import tensorboard
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from . import settings
from .loggers import app_log_formatter
from .models import CNN_APPS
from .runs import TrainingRun
from .settings import APP_NAME, DATASET_DIR, PY_LOGS_DIR, SEED, TB_LOGS_DIR
from .splitting import generate_dataset_splits

log = logging.getLogger(settings.APP_NAME)


class Trainer:

    run_loggers = [log]

    def __init__(
        self,
        data_proportion: float,
        max_epochs: int,
        profile: bool,
        backup_freq: int,
        test_freq: int,
        patience: float,
    ):
        self.data_proportion = data_proportion

        if not DATASET_DIR.exists() or not list(DATASET_DIR.iterdir()):
            raise EnvironmentError(
                "If running module directly, add source dataset to ./dataset "
                "with structure root/classes/images"
            )

        self.splits_dir: Path = Path(tempfile.mkdtemp(prefix=f"{APP_NAME}-splits-"))

        # Define hyperparams
        self.hp_cnn_model = hp.HParam("model", hp.Discrete(list(CNN_APPS.keys())))
        self.hp_weights = hp.HParam("weights", hp.Discrete(["", "imagenet"]))
        self.hp_learning_rate = hp.HParam(
            "learning_rate", hp.Discrete([float(1e-3), float(3e-3), float(5e-3)])
        )
        self.metric_accuracy = "accuracy"

        # Collect hyperparameter combinations and create training runs
        run_num = 0
        self.hparam_combinations = []
        self.training_runs = []
        for cnn_model in self.hp_cnn_model.domain.values:
            for weights in self.hp_weights.domain.values:
                for learning_rate in self.hp_learning_rate.domain.values:
                    self.hparam_combinations.append(
                        {
                            self.hp_cnn_model: cnn_model,
                            self.hp_weights: weights,
                            self.hp_learning_rate: learning_rate,
                        }
                    )
                    self.training_runs.append(
                        TrainingRun(
                            name=(
                                f"run-{run_num}"
                                f"-{cnn_model}"
                                f"-{weights or 'none'}"
                                f"-{learning_rate}"
                            ),
                            max_epochs=max_epochs,
                            profile=profile,
                            backup_freq=backup_freq,
                            test_freq=test_freq,
                            patience=patience,
                            splits_dir=self.splits_dir,
                            metrics=[self.metric_accuracy],
                            cnn_model=cnn_model,
                            weights=weights,
                            learning_rate=learning_rate,
                        )
                    )
                    run_num += 1

    def train(self):
        self._launch_tensorboard()

        generate_dataset_splits(
            src_dir=DATASET_DIR,
            dst_dir=self.splits_dir,
            seed=SEED,
            proportion=self.data_proportion,
        )

        with tf.summary.create_file_writer(
            str(TB_LOGS_DIR / "hparam_tuning")
        ).as_default():
            hp.hparams_config(
                hparams=[self.hp_cnn_model, self.hp_weights, self.hp_learning_rate],
                metrics=[hp.Metric(self.metric_accuracy, display_name="Accuracy")],
            )

        for hparams, training_run in zip(self.hparam_combinations, self.training_runs):
            self._set_run_log_file(
                PY_LOGS_DIR / training_run.name / f"{training_run.name}.log"
            )
            hparams_file_writer = tf.summary.create_file_writer(str(TB_LOGS_DIR))
            with hparams_file_writer.as_default():
                hp.hparams(hparams)

            with tf.distribute.MirroredStrategy().scope():
                accuracy = training_run.execute()

            with hparams_file_writer.as_default():
                tf.summary.scalar(self.metric_accuracy, accuracy, step=1)

    def __del__(self):
        shutil.rmtree(self.splits_dir, ignore_errors=True)

    def _set_run_log_file(self, path, level=logging.INFO):
        for logger in self.run_loggers:
            if getattr(self, "run_log_file_handler", None) and logger.hasHandlers():
                logger.removeHandler(self.run_log_file_handler)
        os.makedirs(path.parent, exist_ok=True)
        self.run_log_file_handler = logging.FileHandler(path)
        self.run_log_file_handler.setFormatter(app_log_formatter)
        self.run_log_file_handler.setLevel(level)
        for logger in self.run_loggers:
            logger.addHandler(self.run_log_file_handler)

    def _launch_tensorboard(self):
        tb = tensorboard.program.TensorBoard()
        tb.configure(argv=[None, f"--logdir={TB_LOGS_DIR}", "--bind_all"])
        url = tb.launch()
        log.info(f"Tensorflow listening on {url}")