#!/usr/bin/env python3
"""
Delete all intermediate checkpoints (files inside any 'checkpoints/' folder)
without touching model.pt / model_last.pt at the run root.

Usage:
    python scripts/clean_checkpoints.py experiments/my_study       # dry run
    python scripts/clean_checkpoints.py experiments/my_study --go  # actually delete
    python scripts/clean_checkpoints.py experiments/               # whole experiments dir
"""

import argparse
import shutil
import sys
from pathlib import Path


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def find_checkpoint_dirs(root: Path) -> list[Path]:
    return sorted(root.rglob("checkpoints"))


def main():
    parser = argparse.ArgumentParser(description="Clean intermediate checkpoints.")
    parser.add_argument("root", type=Path, help="Study or experiments root directory.")
    parser.add_argument(
        "--go", action="store_true", help="Actually delete (default is dry run)."
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        print(f"ERROR: {root} does not exist.")
        sys.exit(1)

    ckpt_dirs = find_checkpoint_dirs(root)
    if not ckpt_dirs:
        print("No 'checkpoints' folders found.")
        return

    total_freed = 0
    total_files = 0

    for ckpt_dir in ckpt_dirs:
        contents = list(ckpt_dir.iterdir())
        if not contents:
            continue

        size = dir_size(ckpt_dir)
        n = sum(1 for f in ckpt_dir.rglob("*") if f.is_file())
        rel = ckpt_dir.relative_to(root)
        print(f"  {'[DRY RUN] ' if not args.go else ''}delete {n} file(s) in {rel}  ({fmt_bytes(size)})")

        if args.go:
            for item in contents:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

        total_freed += size
        total_files += n

    if total_files == 0:
        print("All checkpoints folders are already empty.")
        return

    action = "Freed" if args.go else "Would free"
    print(f"\n{action} {fmt_bytes(total_freed)} across {total_files} file(s) in {len(ckpt_dirs)} run(s).")
    if not args.go:
        print("Re-run with --go to actually delete.")


if __name__ == "__main__":
    main()
