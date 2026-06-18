# @author HCMUT HPC Summer School Hackathon Organizers
# @permission DO_NOT_EDIT
# @note Reference workload or workflow infrastructure; teams should not modify this file.

"""Evaluate a saved custom ResNet checkpoint on validation or test data."""

import argparse
import json

import torch
from torch import nn

from data.datasets import create_dataloaders
from engine.checkpointing import load_model_checkpoint
from engine.evaluator import evaluate_model
from models import create_model
from utils.config import load_config, validate_config
from utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()

    config = load_config(args.config)
    validate_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Request a GPU and use Apptainer --nv."
        )

    seed_everything(int(config["seed"]))
    device = torch.device("cuda")
    loaders, _ = create_dataloaders(
        dataset_dir=config["dataset_dir"],
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        augmentation=str(config["augmentation"]),
        seed=int(config["seed"]),
    )
    model = create_model(
        name=config["model_name"], num_classes=config["num_classes"]
    ).to(device)
    checkpoint = load_model_checkpoint(args.checkpoint, model, device)
    metrics = evaluate_model(
        model, loaders[args.split], nn.CrossEntropyLoss(), device
    )
    print(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "checkpoint_epoch": checkpoint.get("epoch", "unknown"),
                "split": args.split,
                **metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
