# type: ignore
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Sequence, Tuple
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, resize, hflip
import random
from torch.utils.data import Dataset


def random_roll(image: torch.Tensor) -> torch.Tensor:
    image = torch.roll(image, int(random.random() * image.size(-1)), dims=-1)
    return image


def habitat_transforms(image: torch.Tensor) -> torch.Tensor:
    image = random_roll(image)
    if random.random() > 0.5:
        image = hflip(image)
    return image


class Habitat(Dataset):
    def __init__(self, data_dir: str, use_transform: bool = False, **kwargs) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.use_transform = use_transform
        self._meta_path = None
        self._size = None
        self._paths = None
        self._headings = None
        self._timestamps = None
        self._coordinates = None

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return habitat_transforms(x)

    @property
    def meta_path(self) -> str:
        if self._meta_path is None:
            path = Path(self.data_dir)
            path = path / f"{path.stem}.json"
            self._meta_path = path.absolute().resolve().as_posix()
        return self._meta_path

    @property
    def paths(self) -> Sequence[Path]:
        if self._paths is None:
            self._paths = list(Path(self.data_dir).glob("*.png"))
            self._paths = sorted(self._paths, key=lambda x: int(x.stem.split("_")[1]))
        return self._paths

    @property
    def coordinates(self):
        if self._coordinates is None:
            with open(self.meta_path, "r") as file:
                coordinates = json.load(file)["pose_list"]
            coordinates = torch.from_numpy(np.array(coordinates)).double()
            coordinates = coordinates[:, [0, 2]]
            self._coordinates = coordinates
        return self._coordinates

    @property
    def headings(self):
        if self._headings is None:
            with open(self.meta_path, "r") as file:
                headings = json.load(file)["pose_list"]
            self._headings = torch.as_tensor(headings, dtype=torch.float64)[:, 3:]
        return self._headings

    @property
    def timestamps(self) -> torch.Tensor:
        if self._timestamps is None:
            timestamps = range(len(self))
            self._timestamps = torch.tensor(timestamps)
        return self._timestamps

    def load(self, idx: int) -> Image:
        return resize(
            pil_to_tensor(Image.open(str(self.paths[idx])).convert("RGB")), (64, 256)
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        tensor = self.load(idx)
        if self.use_transform:
            tensor = self.transform(tensor)
        return tensor

    def __len__(self) -> int:
        if self._size is None:
            self._size = len(self.paths)
        return self._size
