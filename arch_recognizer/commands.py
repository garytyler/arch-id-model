import logging
import os
import random
import subprocess as sp
import sys
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf

from arch_recognizer.cnns import CNN_APPS
from arch_recognizer.splitting import generate_dataset_splits

from . import sessions
from .loggers import initialize_loggers
from .settings import APP_NAME, BASE_DIR, SEED

log = logging.getLogger(APP_NAME)


def _get_saved_model_dir(args):
    session_models_dir = Path(args.output_dir / f"{int(args.session):04}" / "saves")
    run_dir = None
    for d in sorted(list(session_models_dir.iterdir())):
        if (
            d.name.split("-")[2] == args.cnn_model
            and d.name.split("-")[3] == args.weights
        ):
            run_dir = d
            break

    if not run_dir:
        raise RuntimeError(
            f"Specified run dir not found in session models dir {session_models_dir}"
        )

    max_accuracy = 0.0
    model_dir = None
    for d in run_dir.iterdir():
        accuracy = float(d.name.split("-")[7])
        if max_accuracy < accuracy:
            max_accuracy = accuracy
            model_dir = d

    if not model_dir:
        raise RuntimeError(f"Specified model dir not found in run dir {run_dir}")
    return model_dir


def _create_session(args):
    return sessions.TrainingSession(
        session_dir=_get_session_dir(
            output_dir=args.output_dir,
            resume=args.session,
            force_resume_session=True,
        ),
        dataset_dir=args.dataset_dir,
        data_proportion=1.0,
        min_accuracy=1.0,
        disable_tensorboard_server=True,
    )


def predict(args):
    # Initialize loggers
    initialize_loggers(
        app_log_level=args.log_level,
        tf_log_level=args.tf_log_level,
    )

    log.info("Parsing saved model location...")
    model_dir = _get_saved_model_dir(args)

    log.info(f"Loading saved model {model_dir.name}")
    model = tf.keras.models.load_model(model_dir)
    # model.load_weights(model_dir)

    log.info("Generating test data...")
    session = _create_session(args)
    generate_dataset_splits(
        src_dir=session.dataset_dir,
        dst_dir=session.splits_dir,
        seed=SEED,
        proportion=0.1,
    )
    test_files = [
        os.path.join(path, filename)
        for path, _, files in os.walk(session.splits_dir / "test")
        for filename in files
        if filename.lower().endswith(".jpg")
    ]

    log.info("Generating predictions...")
    for img_path in [Path(random.choice(test_files)) for _ in range(args.count)]:
        img = tf.keras.preprocessing.image.load_img(
            img_path, target_size=CNN_APPS[args.cnn_model]["image_size"]
        )

        img_array = tf.keras.preprocessing.image.img_to_array(img)

        img_array = CNN_APPS[args.cnn_model]["preprocessor"](img_array)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        scores = tf.nn.softmax(predictions[0])
        pred_y = session.class_names[np.argmax(scores)]
        true_y = img_path.parent.name
        print(
            f"path: {img_path}"
            f"\npred: {pred_y}"
            f"\ntrue: {true_y}"
            f"\nconf: {100 * np.max(scores):.2f}%",
            *(
                f"\n\t{n:0>2}-{session.class_names[n]:.<25}{100 * i:.2f}%"
                for n, i in enumerate(scores)
            ),
        )


def train(args):
    session_dir = _get_session_dir(
        output_dir=args.output_dir,
        resume=args.resume,
        force_resume_session=args.force_resume_session,
    )

    # Initialize loggers
    initialize_loggers(
        app_log_level=args.log_level,
        tf_log_level=args.tf_log_level,
        log_dir=session_dir,
    )

    # Configure eager execution of tf.function calls
    tf.config.run_functions_eagerly(args.eager)

    # Start training
    trainer = sessions.TrainingSession(
        session_dir=session_dir,
        dataset_dir=args.dataset_dir,
        data_proportion=args.data_proportion,
        min_accuracy=args.min_accuracy,
        max_epochs=args.max_epochs,
        profile=args.profile,
        disable_tensorboard_server=args.disable_tensorboard_server,
    )
    trainer.execute()


def _get_session_dir(output_dir: Path, resume: str, force_resume_session=bool):
    current_commit_hash: str = _get_current_git_commit_hash()
    session_commit_hash_file_name: str = "commit_hash"

    # Get existing session_dirs
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_sessions: List[int] = [
        int(n.name)
        for n in filter(
            lambda i: len(i.name) == 4 and all([c.isdigit() for c in i.name]),
            list(output_dir.iterdir()),
        )
    ]

    # Handle no resume request
    if resume is None:
        if not len(existing_sessions):
            session_dir = output_dir / "0001"
        else:
            session_dir = output_dir / f"{sorted(existing_sessions)[-1] + 1:04}"
        session_dir.mkdir(parents=True, exist_ok=True)
        with open(session_dir / session_commit_hash_file_name, "w") as f:
            f.write(current_commit_hash)
        return session_dir
    elif resume is -1:
        session_dir = output_dir / f"{sorted(existing_sessions)[-1]:04}"
    elif resume not in existing_sessions:
        raise ValueError(f"Session {resume:04} not found")
    else:
        session_dir = output_dir / f"{resume:04}"

    if force_resume_session:
        return session_dir

    # Check if changes
    current_changes = _get_current_git_changes()
    if current_changes:
        print(f"Please stash or commit the current changes:\n{current_changes}")
        sys.exit()

    # Compare hashes
    commit_hash_file_path = session_dir / session_commit_hash_file_name
    if not commit_hash_file_path.exists():
        raise FileNotFoundError(f"Commit hash found: {commit_hash_file_path}")
    with open(commit_hash_file_path, "r") as f:
        existing_commit_hash = f.readlines()[0].strip()
    if current_commit_hash != existing_commit_hash:
        print(
            f"Cannot resume session {resume:04}. Commit hashes don't match:\n"
            f" Current: {existing_commit_hash}\n"
            f" Session: {current_commit_hash}\n"
            f"Aborting."
        )
        sys.exit()

    return session_dir


def _get_current_git_commit_hash():
    for d in BASE_DIR.iterdir():
        if d.name == ".git":
            command = ["git", "rev-parse", "--verify", "HEAD"]
            return sp.check_output(command, cwd=BASE_DIR).decode().strip()
    else:
        raise FileNotFoundError(d)


def _get_current_git_changes():
    for d in BASE_DIR.iterdir():
        if d.name == ".git":
            command = ["git", "status", "-s", "-uall"]
            return sp.check_output(command, cwd=BASE_DIR).decode().strip()
    else:
        raise FileNotFoundError(d)
