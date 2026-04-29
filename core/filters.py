"""
Regional Prompter - Spatial Mask Builder
Builds binary spatial filter tensors for each region (BASE + COL areas).
Filters are used by RP KSampler for area conditioning.
"""
from __future__ import annotations

import torch
from typing import List

# Dual relative/absolute import (varies by ComfyUI load method)
try:
    from .regions import make_filters
except ImportError:
    from core.regions import make_filters


def rebuild_filters(
    region_rows:     list,
    h:               int,
    w:               int,
    mode:            str  = "Horizontal",
    usebase:         bool = False,
    device                = "cpu",
    latent_channels: int  = 4,
    batch:           int  = 1,
) -> List[torch.Tensor]:
    """
    RegionRow → spatial mask List[Tensor[C,H,W]].
    Original: makefilters() * batch → total areas*batch
    """
    raw    = make_filters(region_rows=region_rows, h=h, w=w,
                          mode=mode, usebase=usebase, device=str(device))
    single = [f.expand(latent_channels, h, w).clone() for f in raw]
    return single * batch          # SD-WebUI original: filters * batch


class RPLatentCompositor:
    """
    Spatially blend latent per division at every sampling step.
    Full port of SD-WebUI denoised_callback_s().
    """

    def __init__(self, region_rows, areas, mode="Horizontal",
                 usebase=False, latent_channels=4):
        self.region_rows      = region_rows
        self.areas            = areas
        self.mode             = mode
        self.usebase          = usebase
        self.latent_channels  = latent_channels
        self._filters: List[torch.Tensor] = []
        self._last_shape      = None
        self._last_batch      = None

    def _ensure_filters(self, h, w, batch, device):
        if (self._last_shape != (h, w) or
                self._last_batch != batch or
                not self._filters):
            self._filters    = rebuild_filters(
                region_rows=self.region_rows, h=h, w=w,
                mode=self.mode, usebase=self.usebase,
                device=device, latent_channels=self.latent_channels,
                batch=batch,
            )
            self._last_shape = (h, w)
            self._last_batch = batch
            print(f"[RPCompositor] filters rebuilt: {len(self._filters)} "
                  f"({h}x{w}) batch={batch}")

    def composite(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [(areas+1)*batch, C, H, W]
        Full port of SD-WebUI original logic.
        """
        areas  = self.areas
        device = x.device
        h, w   = x.shape[-2], x.shape[-1]
        total  = x.shape[0]
        batch  = total // (areas + 1)

        self._ensure_filters(h, w, batch, device)

        if len(self._filters) < areas * batch:
            return x

        xt = x.clone()
        for b in range(batch):
            for a in range(areas):
                fil = self._filters[a + b * areas].to(device)
                x[a + b * areas] = (
                    xt[b + a * batch] * fil
                    + xt[-batch + b]  * (1.0 - fil)
                )
        return x

    @property
    def filters(self):
        return self._filters
