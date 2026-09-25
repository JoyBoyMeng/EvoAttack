from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def copy_memory(src: Path, dst: Path, *, overwrite: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source memory path does not exist: {src}")
    if not src.is_dir():
        raise NotADirectoryError(f"Source memory path is not a directory: {src}")

    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {dst}. Use --overwrite to replace it.")
        shutil.rmtree(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy one clean init target memory directory into separate train/test memories."
    )
    parser.add_argument(
        "--init_mem_path",
        required=True,
        help="Clean source target memory directory, for example memory_db/mem0_normal_all_agents_v003.",
    )
    parser.add_argument(
        "--train_mem_path",
        required=True,
        help="Destination memory directory used by training.",
    )
    parser.add_argument(
        "--test_mem_path",
        required=True,
        help="Destination memory directory used by final testing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing train/test destination directories before copying.",
    )
    args = parser.parse_args()

    src = Path(args.init_mem_path)
    train_dst = Path(args.train_mem_path)
    test_dst = Path(args.test_mem_path)

    if train_dst.resolve() == test_dst.resolve():
        raise ValueError("train_mem_path and test_mem_path must be different directories.")

    copy_memory(src, train_dst, overwrite=args.overwrite)
    copy_memory(src, test_dst, overwrite=args.overwrite)

    print("Prepared target memories:")
    print(f"  source: {src}")
    print(f"  train:  {train_dst}")
    print(f"  test:   {test_dst}")


if __name__ == "__main__":
    main()

