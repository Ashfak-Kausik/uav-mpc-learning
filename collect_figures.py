#!/usr/bin/env python3
"""Collect generated project figures into a single repository folder.

Run this from the repository root or execute this script directly. It scans the
project tree for image files created by the project and copies them into a
single `figures/` directory at the repo root.

Example:
    python3 collect_figures.py

The script excludes virtual environment folders such as `.venv` and
preserves the source-relative directory structure under `figures/` when
necessary to avoid filename collisions.
"""

import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.svg', '.pdf', '.eps'}
EXCLUDE_DIRS = {'.venv', '__pycache__', '.git'}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def should_skip(path: Path) -> bool:
    return any(part in EXCLUDE_DIRS for part in path.parts)


def collect_images(repo_root: Path, target_root: Path) -> int:
    target_root.mkdir(parents=True, exist_ok=True)
    copied = 0

    for path in sorted(repo_root.rglob('*')):
        if path.is_dir() or should_skip(path.relative_to(repo_root)):
            continue
        if is_image_file(path):
            rel_path = path.relative_to(repo_root)
            # Keep project-relative subdirectories under figures if needed.
            dest_path = target_root / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest_path)
            copied += 1
            print(f'Copied: {rel_path} -> {dest_path.relative_to(repo_root)}')

    return copied


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    target_root = repo_root / 'figures'

    print(f'Repo root: {repo_root}')
    print(f'Output dir: {target_root}')

    copied = collect_images(repo_root, target_root)

    if copied == 0:
        print('No project image files were found to collect.')
    else:
        plural = 's' if copied != 1 else ''
        print(f'Collected {copied} image file{plural} into {target_root}.')


if __name__ == '__main__':
    main()
