"""Bias-free, signal-conditioned readout."""
import torch
from torch import nn
from torch.nn import functional as F


class SpectralReadout(nn.Module):
    def __init__(self, response):
        super().__init__()
        selected = response.detach()[torch.tensor([[0, 3, 7], [1, 2, 4]], device=response.device)].clone()
        lifts = []
        for matrix in selected.double():
            norm = matrix.square().sum(0).sqrt()
            normalized = matrix / norm[None, :]
            lifts.append(torch.linalg.solve(
                normalized.T @ normalized + .1 * torch.eye(5, dtype=matrix.dtype, device=matrix.device),
                normalized.T) / norm[:, None])
        self.register_buffer('selected_response', selected)
        self.register_buffer('residual_lift', torch.stack(lifts).to(response.dtype))
        options = dict(device=response.device, dtype=response.dtype, bias=False)
        self.context = nn.Sequential(
            nn.Conv2d(13, 24, 3, padding=1, **options), nn.SiLU(),
            nn.Conv2d(24, 24, 3, padding=1, **options), nn.SiLU(),
            nn.Conv2d(24, 5, 1, **options))
        nn.init.zeros_(self.context[-1].weight)
        self.verify_identity = False

    def forward(self, components, image, branch):
        residual = image - torch.einsum('ck,bkhw->bchw', self.selected_response[branch], components)
        lifted = torch.einsum('kc,bchw->bkhw', self.residual_lift[branch], residual)
        scale = (F.avg_pool2d(image.square().mean(1, keepdim=True), 9, stride=1, padding=4) + .008**2).sqrt()
        features = torch.cat((components / scale, lifted / scale, image / scale), 1)
        result = (components + scale * self.context(features)).clamp_min(0)
        if self.verify_identity:
            assert torch.equal(result, components)
        return result
