import torch
from torch import nn
from .static_readout import SpectralReadout

class PanelFSR(nn.Module):
    def __init__(self,core,response):
        super().__init__();self.core=core
        self.condition=nn.Sequential(nn.Linear(5,32),nn.SiLU(),nn.Linear(32,64))
        nn.init.zeros_(self.condition[-1].weight);nn.init.zeros_(self.condition[-1].bias)
        self._film=None;core.conv_in.register_forward_hook(self.inject)
        self.branch_film=nn.Parameter(torch.zeros(64))
        self.absence_log_gain=nn.Parameter(torch.zeros(5))
        self.register_buffer('response',response.detach().clone())
        self.readout=SpectralReadout(response)
    def inject(self,module,args,features):
        if self._film is None:return features
        scale,bias=self._film.chunk(2,1)
        return features*(1+scale[:,:,None,None])+bias[:,:,None,None]
    def forward(self,image,branch=0):
        if image.shape[1]!=3 or branch not in (0,1):raise ValueError('Three observed bands, A/B branch')
        fixed=image.new_ones((image.shape[0],5))
        self._film=self.condition(fixed)+branch*self.branch_film[None]
        try:
            _,components=self.core(image)
            components=self.readout(components,image,branch)
            return torch.einsum('ck,bkhw->bchw',self.response,components),components
        finally:self._film=None
