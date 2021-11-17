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
from .runs import TrainingRun
from .settings import APP_NAME, BASE_CNNS, SEED
from .splitting import generate_dataset_splits

log = logging.getLogger(settings.APP_NAME)


class TrainingSession:
    def __init__(
        self,
        session_dir: Path,
        dataset_dir: Path,
        batch_size: int,
        max_epochs: int,
        data_proportion: float,
        min_accuracy: float,
        disable_tensorboard_server: bool,
    ):
        self.session_dir: Path = session_dir
        self.cp_dir = self.session_dir / "checkpoints"
        self.cp_dir.mkdir(parents=True, exist_ok=True)
        self.sv_dir = self.session_dir / "models"
        self.sv_dir.mkdir(parents=True, exist_ok=True)
        self.py_dir = self.session_dir / "logs"
        self.py_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir = self.session_dir / "tensorboard"
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        self.cm_dir = self.session_dir / "confusion"
        self.cm_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir: Path = dataset_dir
        self.max_epochs: int = max_epochs
        self.batch_size: int = batch_size
        self.data_proportion: float = data_proportion
        self.min_accuracy: float = min_accuracy
        self.disable_tensorboard_server = disable_tensorboard_server

        if not self.dataset_dir.exists() or not list(self.dataset_dir.iterdir()):
            raise EnvironmentError(f"Dataset dir not found: {self.dataset_dir}")

        self.splits_dir: Path = Path(tempfile.mkdtemp(prefix=f"{APP_NAME}-splits-"))
        self.class_names = [
            i.name for i in sorted(self.dataset_dir.iterdir()) if i.is_dir()
        ]

        # Define hyperparams
        self.hp_base_cnn = hp.HParam("base_cnn", hp.Discrete(list(BASE_CNNS.keys())))
        self.hp_weights = hp.HParam("weights", hp.Discrete(["imagenet", "none"]))
        self.hp_learning_rate = hp.HParam(
            "learning_rate", hp.Discrete([float(1e-3), float(2e-3)])
        )
        self.metric_accuracy = "accuracy"

        # Collect hyperparameter combinations and create training runs
        run_num = 0
        self.hparam_combinations = []
        self.training_runs = []
        for base_cnn in self.hp_base_cnn.domain.values:
            for weights in self.hp_weights.domain.values:
                # for learning_rate in self.hp_learning_rate.domain.values:
                self.hparam_combinations.append(
                    {self.hp_base_cnn: base_cnn, self.hp_weights: weights}
                )
                run_name = (
                    f"{self.session_dir.name}"
                    f"-{run_num}"
                    f"-{BASE_CNNS[base_cnn].name}"
                    f"-{weights or 'none'}"
                )
                self.training_runs.append(
                    TrainingRun(
                        name=run_name,
                        test_freq=1,
                        splits_dir=self.splits_dir,
                        max_epochs=self.max_epochs,
                        class_names=self.class_names,
                        metrics=[self.metric_accuracy],
                        metric_accuracy=self.metric_accuracy,
                        base_cnn=BASE_CNNS[base_cnn],
                        weights=weights,
                        min_accuracy=min_accuracy,
                        dataset_dir=self.dataset_dir,
                        batch_size=self.batch_size,
                        cp_dir=self.cp_dir / run_name,
                        sv_dir=self.sv_dir / run_name,
                        py_dir=self.py_dir / run_name,
                        tb_dir=self.tb_dir / run_name,
                        cm_dir=self.cm_dir / run_name,
                    )
                )
                run_num += 1

    def execute(self):
        if not self.disable_tensorboard_server:
            self._launch_tensorboard()

        generate_dataset_splits(
            src_dir=self.dataset_dir,
            dst_dir=self.splits_dir,
            seed=SEED,
            proportion=self.data_proportion,
        )

        hp_dir = self.tb_dir / "hparam_tuning"
        with tf.summary.create_file_writer(str(hp_dir)).as_default():
            hp.hparams_config(
                hparams=[self.hp_base_cnn, self.hp_weights],
                metrics=[hp.Metric(self.metric_accuracy, display_name="Accuracy")],
            )

        for hparams, run in zip(self.hparam_combinations, self.training_runs):
            self._set_run_log_file(run.py_dir / f"{run.name}.log")
            if run.is_completed():
                continue
            # Perform all training in tb file writer context to send full stats
            with tf.summary.create_file_writer(str(hp_dir / run.name)).as_default():
                hp.hparams(hparams)

                try:
                    accuracy = run.execute()
                except Exception as err:
                    log.error(err)
                    raise err
                else:
                    tf.summary.scalar(self.metric_accuracy, accuracy, step=1)

    def __del__(self):
        try:
            _splits_dir = getattr(self, "splits_dir")
        except AttributeError:
            pass
        else:
            shutil.rmtree(_splits_dir, ignore_errors=True)

    def _set_run_log_file(self, path, level=logging.INFO):
        if getattr(self, "run_log_file_handler", None):
            log.removeHandler(self.run_log_file_handler)
        os.makedirs(path.parent, exist_ok=True)
        self.run_log_file_handler = logging.FileHandler(path)
        self.run_log_file_handler.setFormatter(app_log_formatter)
        self.run_log_file_handler.setLevel(level)
        log.addHandler(self.run_log_file_handler)

    def _launch_tensorboard(self):
        tb = tensorboard.program.TensorBoard()
        tb.configure(argv=[None, f"--logdir={str(self.tb_dir)}", "--bind_all"])
        url = tb.launch()
        log.info(f"Tensorflow listening on {url}")
