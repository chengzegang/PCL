from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Resize

from . import models
from .dataset import ImageFolder
from . import metrics


@torch.no_grad()
def evaluate(
    model_name: str,
    pretrained: bool,
    feature_size: int | None,
    weights: str | None,
    datadir: str,
    metafile: str,
    selective_mask_path: str | None,
    coordinate_columns: Tuple[str, ...],
    direction_columns: Tuple[str, ...],
    spatial_radius: float,
    batch_size: int,
    num_workers: int,
    device: str,
    atks: Tuple[int, ...],
    image_shape: Tuple[int, int],
    **kwargs,
):
    metadata = pd.read_csv(metafile)
    coo_lists = metadata[coordinate_columns].values
    coordinates = torch.as_tensor(coo_lists, dtype=torch.float64)
    direct_lists = metadata[direction_columns].values
    directions = torch.as_tensor(direct_lists, dtype=torch.float64)
    distances = torch.cdist(coordinates, coordinates)
    neighbors = distances < spatial_radius
    selective_mask = None
    if selective_mask_path is not None:
        selective_mask = torch.load(selective_mask_path)

    model = models.build(
        model_name,
        pretrained if weights is None else False,
        feature_size,
        **kwargs,
    )
    dataset = ImageFolder(datadir, Resize(image_shape, antialias=True))
    dataloader = DataLoader(dataset, batch_size, num_workers=num_workers)

    if weights is not None:
        model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    feat_chks = []
    for idx, image in enumerate(dataloader):
        image = image.to(device).float()
        feat_chks.append(model(image).cpu())
    feats = torch.cat(feat_chks, dim=0)

    fdists = torch.cdist(feats, feats)
    topk = torch.as_tensor(atks, dtype=torch.int64)
    recalls = metrics.recalls(fdists, neighbors, topk, selective_mask)
    divers = metrics.heading_diversity(fdists, neighbors, directions, selective_mask)
    return recalls, divers
