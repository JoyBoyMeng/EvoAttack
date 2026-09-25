from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def resolve_init_path(memory_db_dir: Path, init_memory: str) -> Path:
    init_path = Path(init_memory)
    if init_path.is_absolute() or init_path.parent != Path("."):
        return init_path
    return memory_db_dir / init_path


def copy_memory(src: Path, dst: Path, *, overwrite: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Init memory path does not exist: {src}")
    if not src.is_dir():
        raise NotADirectoryError(f"Init memory path is not a directory: {src}")

    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {dst}. Use --overwrite to replace it.")
        shutil.rmtree(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy one init memory library under ASB/memory_db into a train memory, "
            "a test memory, or both for attack-agent experiments."
        )
    )
    parser.add_argument(
        "--init_memory",
        required=True,
        help=(
            "Init memory name under memory_db, or a full/relative path. Example: "
            "mem0_normal_all_agents_v003"
        ),
    )
    parser.add_argument(
        "--memory_db_dir",
        default="memory_db",
        help="Memory DB root directory. Use this script from ASB; default: memory_db.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove each selected destination before copying.",
    )
    parser.add_argument(
        "--copy_mode",
        choices=("both", "train", "test"),
        default="both",
        help=(
            "Which destination to create. Use 'train' during training and 'test' "
            "immediately before a test so both are copied independently from init_memory."
        ),
    )
    parser.add_argument(
        "--train_dst",
        default=None,
        help="Optional explicit train memory destination. Defaults to <init>_4attacktrain.",
    )
    parser.add_argument(
        "--test_dst",
        default=None,
        help="Optional explicit test memory destination. Defaults to <init>_4attacktest.",
    )
    args = parser.parse_args()

    memory_db_dir = Path(args.memory_db_dir)
    src = resolve_init_path(memory_db_dir, args.init_memory)
    parent = src.parent
    train_dst = Path(args.train_dst) if args.train_dst else parent / f"{src.name}_4attacktrain"
    test_dst = Path(args.test_dst) if args.test_dst else parent / f"{src.name}_4attacktest"

    selected_destinations = []
    if args.copy_mode in {"both", "train"}:
        selected_destinations.append(("train", train_dst))
    if args.copy_mode in {"both", "test"}:
        selected_destinations.append(("test", test_dst))

    if src.resolve() in {dst.resolve() for _, dst in selected_destinations}:
        raise ValueError("init_memory must not already be one of the derived train/test destinations.")
    if args.copy_mode == "both" and train_dst.resolve() == test_dst.resolve():
        raise ValueError("train_dst and test_dst must be different directories.")

    for _, destination in selected_destinations:
        copy_memory(src, destination, overwrite=args.overwrite)

    print("Prepared attack-agent target memories:")
    print(f"  init:  {src}")
    for label, destination in selected_destinations:
        print(f"  {label}: {destination}")


if __name__ == "__main__":
    main()
