import logging
import subprocess as sp
import sys
from pathlib import Path
from typing import List

import tensorflow as tf

from . import sessions
from .loggers import initialize_loggers
from .settings import APP_NAME, BASE_DIR

log = logging.getLogger(APP_NAME)


def train(args):
    session_dir = _get_session_dir(
        output_dir=args.output_dir,
        session=args.session,
        force_resume_session=args.force_resume_session,
    )

    # Initialize loggers
    initialize_loggers(
        app_log_level=args.log_level,
        tf_log_level=args.tf_log_level,
        log_dir=session_dir,
    )

    # Configure eager execution of tf.function calls
    tf.config.run_functions_eagerly(False)

    # Start training
    trainer = sessions.TrainingSession(
        session_dir=session_dir,
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        data_proportion=args.data_proportion,
        min_accuracy=args.min_accuracy,
        disable_tensorboard_server=args.disable_tensorboard_server,
    )
    trainer.execute()


def _get_session_dir(output_dir: Path, session: str, force_resume_session=bool) -> Path:
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
    if session is None:
        if not len(existing_sessions):
            session_dir = output_dir / "0001"
        else:
            session_dir = output_dir / f"{sorted(existing_sessions)[-1] + 1:04}"
        session_dir.mkdir(parents=True, exist_ok=True)
        with open(session_dir / session_commit_hash_file_name, "w") as f:
            f.write(current_commit_hash)
        return session_dir
    elif session is -1:
        session_dir = output_dir / f"{sorted(existing_sessions)[-1]:04}"
    elif session not in existing_sessions:
        raise ValueError(f"Session {session:04} not found")
    else:
        session_dir = output_dir / f"{session:04}"

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
            f"Cannot resume session {session:04}. Commit hashes don't match:\n"
            f" Current: {current_commit_hash}\n"
            f" Session: {existing_commit_hash}\n"
            f"Aborting."
        )
        sys.exit()

    return Path(session_dir)


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
