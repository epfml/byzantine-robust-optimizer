import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from .resnet import get_resnet_model
from .resnet_gn import get_resnet_model_gn
from ..utils import log_dict


def get_resnet20(use_cuda=False, gn=False):
    if gn:
        print("Using group normalization")
        return get_resnet_model_gn(
            model="resnet20", version=1, dtype="fp32", num_classes=10, use_cuda=use_cuda
        )

    print("Using Batch normalization")
    return get_resnet_model(
        model="resnet20", version=1, dtype="fp32", num_classes=10, use_cuda=use_cuda
    )


def cifar10(
    data_dir,
    train,
    download,
    batch_size,
    shuffle=None,
    sampler_callback=None,
    dataset_cls=datasets.CIFAR10,
    drop_last=True,
    **loader_kwargs
):
    if sampler_callback is not None and shuffle is not None:
        raise ValueError

    cifar10_stats = {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
    }

    if train:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(cifar10_stats["mean"], cifar10_stats["std"]),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(cifar10_stats["mean"], cifar10_stats["std"]),
            ]
        )

    dataset = dataset_cls(
        root=data_dir, train=train, download=download, transform=transform
    )

    dataset.targets = torch.LongTensor(dataset.targets)

    sampler = sampler_callback(dataset) if sampler_callback else None
    log_dict(
        {
            "Type": "Setup",
            "Dataset": "cifar10",
            "data_dir": data_dir,
            "train": train,
            "download": download,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "sampler": sampler.__str__() if sampler else None,
        }
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=drop_last,
        **loader_kwargs,
    )
