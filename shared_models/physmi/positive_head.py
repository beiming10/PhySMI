"""Smooth nonnegative13-dye head; no hard floor, clamp or presence mask."""
import torch
import torch.nn.functional as F
from .core import UnmixingHead
class PositiveDyeHead(UnmixingHead):
    def forward(self,features):
        logits=self.fc_abundance(features)
        abundance=self.softplus_tau*F.softplus(logits/self.softplus_tau)
        derivative=torch.sigmoid(logits.detach()/self.softplus_tau)
        self.last_min_dye_max_derivative=float(derivative.amax(dim=(0,2,3)).amin().item())
        dictionary=torch.cat([F.relu(self.endmember_dict_signal),F.relu(self.endmember_dict_background)],dim=0)
        reconstruction=torch.einsum('km,bkhw->bmhw',dictionary,abundance)
        if not torch.isfinite(abundance).all() or not torch.isfinite(reconstruction).all():raise RuntimeError('Nonfinite softplus output')
        return reconstruction,abundance
