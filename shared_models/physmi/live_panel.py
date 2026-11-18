import torch
from torch import nn

class PanelFSR(nn.Module):
    def __init__(self,core):
        super().__init__();self.core=core
        self.condition=nn.Sequential(nn.Linear(5,32),nn.SiLU(),nn.Linear(32,64))
        nn.init.zeros_(self.condition[-1].weight);nn.init.zeros_(self.condition[-1].bias)
        self._film=None;core.conv_in.register_forward_hook(self.inject)
    def inject(self,module,args,features):
        if self._film is None:return features
        scale,bias=self._film.chunk(2,1)
        return features*(1+scale[:,:,None,None])+bias[:,:,None,None]
    def forward(self,image,panel):
        if image.shape[1]!=3 or panel.shape!=(image.shape[0],5):raise ValueError('Three image channels, five panel indicators')
        if not bool(torch.all((panel==0)|(panel==1))):raise ValueError('Binary known label panel')
        self._film=self.condition(panel.to(image))
        try:return self.core(image)
        finally:self._film=None
