"""CIFAR-10 data pipeline."""

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def get_cifar10(batch_size: int = 512, num_workers: int = 4, val_split: float = 0.0):
    """Return train (+ optional val) and test DataLoaders for CIFAR-10.

    If val_split > 0, a fraction of the training set is held out for validation.
    """
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform,
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform,
    )

    if val_split > 0:
        val_size = int(len(train_set) * val_split)
        train_size = len(train_set) - val_size
        train_set, val_set = random_split(train_set, [train_size, val_size],
                                          generator=torch.Generator().manual_seed(42))
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)
    else:
        val_loader = None

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
