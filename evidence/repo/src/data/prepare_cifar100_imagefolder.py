# @author HCMUT HPC Summer School Hackathon Organizers
# @permission DO_NOT_EDIT
# @note Reference workload or workflow infrastructure; teams should not modify this file.

"""Download CIFAR-100 and export deterministic ImageFolder splits."""

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

from torchvision.datasets import CIFAR100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare full CIFAR-100 as train/val/test ImageFolder splits."
    )
    parser.add_argument("--download-root", required=True)
    parser.add_argument("--output-dir", default="data/cifar100")
    parser.add_argument("--val-count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args()


def export_examples(
    dataset: CIFAR100,
    indices: list[int],
    split_dir: Path,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for source_index in indices:
        image, target = dataset[source_index]
        class_name = dataset.classes[target]
        class_dir = split_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        image.save(class_dir / f"{class_name}_{source_index:05d}.png")
        counts[class_name] += 1
    return counts


def main() -> None:
    args = parse_args()
    if args.val_count != 5000:
        raise ValueError(
            "--val-count must be 5000 so the fixed split remains "
            "train=45000, val=5000"
        )

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"{output_dir} is not empty. Use --overwrite to replace it."
            )
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading/loading CIFAR-100 in: {args.download_root}")
    source_train = CIFAR100(root=args.download_root, train=True, download=True)
    source_test = CIFAR100(root=args.download_root, train=False, download=True)

    indices = list(range(len(source_train)))
    random.Random(args.seed).shuffle(indices)
    val_indices = indices[: args.val_count]
    train_indices = indices[args.val_count :]
    test_indices = list(range(len(source_test)))

    split_plan = (
        ("train", source_train, train_indices),
        ("val", source_train, val_indices),
        ("test", source_test, test_indices),
    )
    print(
        f"Using deterministic seed {args.seed}: "
        f"train={len(train_indices)}, val={len(val_indices)}, "
        f"test={len(test_indices)}"
    )

    for split, dataset, split_indices in split_plan:
        print(f"Exporting {split}: {len(split_indices)} images")
        counts = export_examples(dataset, split_indices, output_dir / split)
        print(
            f"  {split}: {sum(counts.values())} images across "
            f"{len(counts)} class folders"
        )

    print(f"CIFAR-100 ImageFolder dataset ready: {output_dir}")
    print("Final counts: train=45000, val=5000, test=10000")


if __name__ == "__main__":
    main()
