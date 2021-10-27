import logging
import os
import shutil
import tempfile
from pathlib import Path

import tensorboard
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from . import settings
from .cnns import CNN_APPS
from .loggers import app_log_formatter
from .runs import TrainingRun
from .settings import APP_NAME, SEED
from .splitting import generate_dataset_splits

log = logging.getLogger(settings.APP_NAME)


class TrainingSession:

    run_loggers = [log]

    def __init__(
        self,
        dir: Path,
        dataset_dir: Path,
        data_proportion: float,
        min_accuracy: float,
        max_epochs: int,
        profile: bool,
    ):
        self.dir = dir
        self.cp_dir = self.dir / "checkpoints"
        self.cp_dir.mkdir(parents=True, exist_ok=True)
        self.sv_dir = self.dir / "saves"
        self.sv_dir.mkdir(parents=True, exist_ok=True)
        self.py_dir = self.dir / "logs"
        self.py_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir = self.dir / "tensorboard"
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir: Path = dataset_dir
        self.data_proportion: float = data_proportion
        self.min_accuracy: float = min_accuracy

        if not self.dataset_dir.exists() or not list(self.dataset_dir.iterdir()):
            raise EnvironmentError(f"Dataset dir not found: {self.dataset_dir}")

        self.splits_dir: Path = Path(tempfile.mkdtemp(prefix=f"{APP_NAME}-splits-"))

        # Define hyperparams
        self.hp_cnn_model = hp.HParam("model", hp.Discrete(list(CNN_APPS.keys())))
        self.hp_weights = hp.HParam("weights", hp.Discrete(["", "imagenet"]))
        self.hp_learning_rate = hp.HParam(
            "learning_rate", hp.Discrete([float(1e-3), float(2e-3)])
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
                    run_name = (
                        f"{self.dir.name}"
                        f"-{run_num}"
                        f"-{cnn_model}"
                        f"-{weights or 'none'}"
                        f"-{learning_rate}"
                    )
                    self.training_runs.append(
                        TrainingRun(
                            name=run_name,
                            max_epochs=max_epochs,
                            profile=profile,
                            test_freq=1,
                            patience=80,
                            splits_dir=self.splits_dir,
                            metrics=[self.metric_accuracy],
                            cnn_model=cnn_model,
                            weights=weights,
                            learning_rate=learning_rate,
                            min_accuracy=min_accuracy,
                            dataset_dir=self.dataset_dir,
                            cp_dir=self.cp_dir / run_name,
                            sv_dir=self.sv_dir / run_name,
                            py_dir=self.py_dir / run_name,
                            tb_dir=self.tb_dir / run_name,
                        )
                    )
                    run_num += 1

    def execute(self):
        self._launch_tensorboard()

        generate_dataset_splits(
            src_dir=self.dataset_dir,
            dst_dir=self.splits_dir,
            seed=SEED,
            proportion=self.data_proportion,
        )

        with tf.summary.create_file_writer(
            str(self.tb_dir / "hparam_tuning")
        ).as_default():
            hp.hparams_config(
                hparams=[self.hp_cnn_model, self.hp_weights, self.hp_learning_rate],
                metrics=[hp.Metric(self.metric_accuracy, display_name="Accuracy")],
            )

        for hparams, training_run in zip(self.hparam_combinations, self.training_runs):
            self._set_run_log_file(training_run.py_dir / f"{training_run.name}.log")
            if training_run.is_completed():
                continue
            # Perform all training in tb file writer context to send full stats
            with tf.summary.create_file_writer(str(self.tb_dir)).as_default():
                hp.hparams(hparams)

                try:
                    accuracy = training_run.execute()
                except Exception as err:
                    log.error(err)
                    raise err

                tf.summary.scalar(self.metric_accuracy, accuracy, step=1)

    def __del__(self):
        try:
            _splits_dir = getattr(self, "splits_dir")
        except AttributeError:
            pass
        else:
            shutil.rmtree(_splits_dir, ignore_errors=True)

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
        tb.configure(argv=[None, f"--logdir={str(self.tb_dir)}", "--bind_all"])
        url = tb.launch()
        log.info(f"Tensorflow listening on {url}")
