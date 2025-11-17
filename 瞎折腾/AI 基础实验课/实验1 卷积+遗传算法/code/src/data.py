from pathlib import Path
from typing import Optional
import torch
import torchvision
import torchvision.transforms as T
try:
    import config as C
except Exception:  # pragma: no cover
    from . import config as C


def build_transforms(is_train: bool) -> T.Compose:
    if is_train:
        augs = [
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, padding=4),
            T.ToTensor(),
        ]
    else:
        augs = [T.ToTensor()]
    return T.Compose(augs)


def load_cifar10(is_train: bool,
                 batch_size: int,
                 sample_count: Optional[int] = None,
                 seed: Optional[int] = None,
                 num_workers: int = C.NUM_WORKERS):
    dataset = torchvision.datasets.CIFAR10(
        root=str(C.DATA_ROOT), train=is_train,
        transform=build_transforms(is_train), download=True,
    )

    # 子集抽样（仅对训练集启用更有意义；测试也可用于加速）
    if sample_count:
        n = len(dataset)
        if sample_count >= n:
            indices = list(range(n))
        else:
            if seed is not None:
                g = torch.Generator()
                g.manual_seed(seed)
                indices = torch.randperm(n, generator=g)[:sample_count].tolist()
            else:
                indices = torch.randperm(n)[:sample_count].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers
    )
    return loader
