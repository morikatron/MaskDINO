from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .border_head import _make_group_norm


class InstanceMaskRefineHead(nn.Module):
    """
    Shared full-resolution mask feature refiner for instance segmentation.

    It takes:
    - coarse mask features from the pixel decoder
    - low-level backbone features for spatial detail
    - the full-resolution RGB image

    It outputs shared full-resolution mask features that are combined with
    query-specific mask embeddings to predict refined instance masks.
    """

    def __init__(
        self,
        mask_dim: int,
        low_level_channels: int,
        hidden_dim: int = 256,
        image_hidden_dim: int = 64,
        full_res_hidden_dim: int = 128,
    ):
        super().__init__()
        mid_hidden_dim = max(hidden_dim // 2, 128)
        self.mask_proj = nn.Sequential(
            nn.Conv2d(mask_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            _make_group_norm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(low_level_channels, hidden_dim, kernel_size=1, bias=False),
            _make_group_norm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.image_s4_stem = nn.Sequential(
            nn.Conv2d(3, image_hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            _make_group_norm(image_hidden_dim, max_groups=8),
            nn.ReLU(inplace=True),
            nn.Conv2d(image_hidden_dim, image_hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            _make_group_norm(image_hidden_dim, max_groups=8),
            nn.ReLU(inplace=True),
        )
        self.image_s2_stem = nn.Sequential(
            nn.Conv2d(3, mid_hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            _make_group_norm(mid_hidden_dim, max_groups=16),
            nn.ReLU(inplace=True),
        )
        self.image_s1_stem = nn.Sequential(
            nn.Conv2d(3, full_res_hidden_dim, kernel_size=3, padding=1, bias=False),
            _make_group_norm(full_res_hidden_dim, max_groups=16),
            nn.ReLU(inplace=True),
        )
        fuse_in_channels = hidden_dim * 2 + image_hidden_dim
        self.fuse_s4 = nn.Sequential(
            nn.Conv2d(fuse_in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            _make_group_norm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            _make_group_norm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.refine_s2 = nn.Sequential(
            nn.Conv2d(hidden_dim + mid_hidden_dim, mid_hidden_dim, kernel_size=3, padding=1, bias=False),
            _make_group_norm(mid_hidden_dim, max_groups=16),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_hidden_dim, mid_hidden_dim, kernel_size=3, padding=1, bias=False),
            _make_group_norm(mid_hidden_dim, max_groups=16),
            nn.ReLU(inplace=True),
        )
        self.refine_s1 = nn.Sequential(
            nn.Conv2d(mid_hidden_dim + full_res_hidden_dim, full_res_hidden_dim, kernel_size=3, padding=1, bias=False),
            _make_group_norm(full_res_hidden_dim, max_groups=16),
            nn.ReLU(inplace=True),
            nn.Conv2d(full_res_hidden_dim, full_res_hidden_dim, kernel_size=3, padding=1, bias=False),
            _make_group_norm(full_res_hidden_dim, max_groups=16),
            nn.ReLU(inplace=True),
        )
        self.predictor = nn.Conv2d(full_res_hidden_dim, mask_dim, kernel_size=1)

    def forward(
        self,
        mask_features: torch.Tensor,
        low_level_features: torch.Tensor,
        images: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        target_s4 = low_level_features.shape[-2:]
        target_s1 = output_size if output_size is not None else images.shape[-2:]

        image_s4 = self.image_s4_stem(images)
        if image_s4.shape[-2:] != target_s4:
            image_s4 = F.interpolate(image_s4, size=target_s4, mode="bilinear", align_corners=False)
        if mask_features.shape[-2:] != target_s4:
            mask_features = F.interpolate(mask_features, size=target_s4, mode="bilinear", align_corners=False)

        fused = torch.cat(
            [
                self.mask_proj(mask_features),
                self.low_level_proj(low_level_features),
                image_s4,
            ],
            dim=1,
        )
        x = self.fuse_s4(fused)

        image_s2 = self.image_s2_stem(images)
        x = F.interpolate(x, size=image_s2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.refine_s2(torch.cat([x, image_s2], dim=1))

        image_s1 = self.image_s1_stem(images)
        if image_s1.shape[-2:] != target_s1:
            image_s1 = F.interpolate(image_s1, size=target_s1, mode="bilinear", align_corners=False)
        x = F.interpolate(x, size=target_s1, mode="bilinear", align_corners=False)
        x = self.refine_s1(torch.cat([x, image_s1], dim=1))
        return self.predictor(x)
