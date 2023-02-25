from __future__ import annotations
import math
from typing import Sequence, Tuple

import torch


@torch.jit.script
def cdist(
    matrix1: torch.Tensor, matrix2: torch.Tensor, chunk_size: int = 2048
) -> torch.Tensor:
    """
    matrix1: torch.Tensor, (N, D)
    matrix2 : torch.Tensor, (M, D)
    chunk_size: int, chunk size for memory efficiency
    return:
    d_matrix: distance matrix, (N, M)
    """

    d_matrix = torch.zeros(
        (matrix1.size(0), matrix2.size(0)), device=matrix1.device, dtype=torch.float64
    )
    for i in range(0, matrix1.size(0), chunk_size):
        for j in range(0, matrix2.size(0), chunk_size):
            local1 = matrix1[i : i + chunk_size].to(torch.float64)
            local2 = matrix2[j : j + chunk_size].to(torch.float64)
            d_matrix[i : i + chunk_size, j : j + chunk_size] = torch.cdist(
                local1, local2
            )
    return d_matrix


@torch.jit.script
def row_sum(
    matrix: torch.Tensor, dtype: torch.dtype = torch.float64, chunk_size: int = 2048
) -> torch.Tensor:
    """
    matrix: torch.Tensor, (N, M)
    chunk_size: int, chunk size for memory efficiency
    return:
    row_sum: torch.Tensor, (N)
    """
    row_sum = torch.zeros((matrix.size(0),), device=matrix.device, dtype=dtype)
    for i in range(0, matrix.size(0), chunk_size):
        local = matrix[i : i + chunk_size].to(dtype)
        row_sum[i : i + chunk_size] = local.sum(-1)
    return row_sum


def recalls(
    fdists: torch.Tensor,
    gt: torch.Tensor,
    k: torch.Tensor,
    selective_mask: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, ...]:
    gt = gt.to(fdists.device).type_as(fdists)
    selective_mask = (
        torch.ones_like(gt, dtype=torch.bool)
        if selective_mask is None
        else selective_mask
    )
    selective_mask = selective_mask.to(fdists.device).type_as(fdists)
    k = torch.as_tensor(k, dtype=torch.int64).to(fdists.device)

    selective_mask.fill_diagonal_(False)
    if selective_mask is not None:
        fdists = fdists.masked_fill(~selective_mask.to_dense(), torch.inf)
        gt = gt.to_dense() & selective_mask.to_dense()

    gt = gt.to_dense()
    non_trivial = torch.sum(gt, dim=-1) >= torch.max(k)
    fdists = fdists[non_trivial]
    gt = gt[non_trivial]

    _, sorted_indices = torch.topk(
        fdists, k=int(torch.max(k).item()), dim=-1, largest=False
    )

    sorted_gt = torch.gather(gt, 1, sorted_indices)
    first_occurs = torch.argmax(sorted_gt.to(torch.int8), dim=1, keepdim=True)
    first_occurs[~torch.gather(sorted_gt, 1, first_occurs)] = len(gt) + 1

    recalls = first_occurs.view(-1, 1) < k.view(1, -1)

    recalls_per_frame = recalls.clone().cpu()
    recalls = recalls.double().mean(dim=0)
    return recalls.cpu(), recalls_per_frame, non_trivial


@torch.jit.script
def topk_hdiver(A: torch.Tensor, D: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    this function computes the heading diversity of top-k points in distance matrix for each row.
    A: (N, N) pairwise angle matrix
    D: (N, N) pairwise distance matrix
    K: (N, 1) top-k value for each point

    return:
    HD: (N, 8): top-k heading diversity for each point
    """
    # calculate the positive size vector
    max_k = torch.max(K).item()
    # find the top-max_p_size points for each point
    TH, TI = torch.topk(D, dim=-1, k=int(max_k), largest=False)

    # select the angle matrix of top-max_p_size points
    TI = TI.to(A.device)
    TA = torch.gather(A, 1, TI)

    # mask out extra points
    index = torch.arange(TA.size(1), device=TA.device).view(1, -1)
    TA[index >= K.view(-1, 1)] = 0
    # split the angle matrix into ranges of pi/4
    # by floor dividing the angle matrix by pi/4
    TA = torch.div(TA, (torch.pi / 4), rounding_mode="floor")
    # calulate the histogram of each row of the angle matrix
    HD = [torch.histc(row, min=0, max=8, bins=8) > 0 for row in torch.unbind(TA, dim=0)]
    results = torch.stack(HD)
    return results


def heading_diversity(
    fdists: torch.Tensor,
    gt: torch.Tensor,
    headings: torch.Tensor,
    selective_mask: torch.Tensor | None = None,
    chunk_size: int = 512,
) -> Tuple[torch.Tensor, ...]:
    """
    coordinates: (N, 2) coordinates of each point
    F: (N, K) feature matrix of each point
    P: (N, N) positive matrix

    return:
    R: (N, 8): heading diversity coverage of true positives over ground truth
    """
    gt = gt.to(fdists.device)
    headings = headings.to(fdists.device)
    # remove the last point
    # get pairwise directional angle matrix
    headings = headings / torch.linalg.vector_norm(headings, dim=-1, keepdim=True)
    pairwise_hdiff = headings.unsqueeze(0) - headings.unsqueeze(1)
    A: torch.Tensor = (
        torch.linalg.vector_norm(pairwise_hdiff, dim=-1).clamp(1e-8, 2 - 1e-8)
        * torch.pi
    )

    # calculate distance matrix
    # mask out self with inf
    if selective_mask is not None:
        selective_mask = selective_mask.to(fdists.device)
        fdists.masked_fill_(~selective_mask, torch.inf)
    fdists.fill_diagonal_(torch.inf)
    # calculate  ground-truth size for each point
    GTS = row_sum(gt, dtype=torch.long, chunk_size=chunk_size)
    # compute heading diversity for positives
    HD = topk_hdiver(A, fdists, GTS)

    # build a pseudo distance matrix for ground truth
    # where the distance between positive pairs are 0,
    # otherwie 1
    DGT = torch.full_like(fdists, torch.inf, device=A.device)
    DGT.fill_diagonal_(torch.inf)
    DGT[gt] = 0
    # compute heading diversity for ground-truth
    GHD = topk_hdiver(A, DGT, GTS)

    # remove the first and last bin
    HD = HD[:, 1:-1]
    GHD = GHD[:, 1:-1]

    # compute the HD-coverage of true positives over ground-truth positives
    TP = torch.logical_and(HD, GHD)
    R = row_sum(TP, torch.long, chunk_size).double() / (
        row_sum(GHD, torch.long, chunk_size) + 1e-8
    )
    # remove rows with no ground-truth positives
    non_trivial = row_sum(GHD, torch.long, chunk_size) > 0
    R = R[non_trivial]
    # compute the mean of the HD
    R_per_point = R.clone()
    R = torch.mean(R.double())
    return R, R_per_point, non_trivial
