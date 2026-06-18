# @author HCMUT HPC Summer School Hackathon Organizers
# @permission DO_NOT_EDIT
# @note Reference workload or workflow infrastructure; teams should not modify this file.

"""CIFAR-100 transforms."""

from torchvision import transforms


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def build_transforms(augmentation: str = "basic"):
    augmentation = augmentation.lower()
    normalize = transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)

    if augmentation == "basic":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif augmentation == "none":
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        raise ValueError(
            f"Unknown augmentation {augmentation!r}; expected 'none' or 'basic'"
        )

    evaluation_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, evaluation_transform
