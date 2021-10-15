import filecmp
import os
import random
import shutil
from pathlib import Path
from typing import Dict

import numpy as np


def generate_dataset_splits(
    src_dir: Path,
    dst_dir: Path,
    seed: int,
    ratios: Dict[str, float] = {"val": 0.15, "test": 0.15},
    proportion: float = 1.0,
) -> Path:
    rng = np.random.default_rng(seed=seed)
    dst_files: dict = {"train": dict()}
    dst_files.update({k: {} for k in ratios})

    # Generate new
    for src_class_dir in src_dir.iterdir():
        src_class_files = list(src_class_dir.iterdir())
        src_class_count = round(len(src_class_files) * proportion)

        # Shuffle files
        random.shuffle(src_class_files, random=rng.random)

        # Set counts
        counts = {k: round(src_class_count * v) for k, v in ratios.items()}
        counts["train"] = src_class_count - sum(counts.values())

        # Sort into lists
        for split_name in dst_files:
            dst_files[split_name][src_class_dir.name] = [
                src_class_files.pop().name for _ in range(counts[split_name])
            ]

    for split_name, classes in dst_files.items():
        for class_name, file_names in classes.items():
            dst_handled_paths = set()
            os.makedirs(dst_dir / split_name / class_name, exist_ok=True)
            # Copy files only if missing or if name and hash don't match
            for file_name in file_names:
                dst = dst_dir / split_name / class_name / file_name
                src = src_dir / class_name / file_name
                if not dst.exists():
                    shutil.copyfile(src, dst)
                elif not filecmp.cmp(dst, src):
                    os.remove(dst)
                    shutil.copyfile(src, dst)
                dst_handled_paths.add(dst)
            # Remove extra files in the destination directory
            dst_existing_paths = set(Path(dst_dir / split_name / class_name).iterdir())
            for unhandled_path in dst_existing_paths.difference(dst_handled_paths):
                os.remove(unhandled_path)

    return dst_dir
