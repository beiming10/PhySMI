"""Observation objectives and one unlabelled shared image critic."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .cells_loss import objective as cells_objective

def eight_channel_objective(ra,rb,xa,xb,observed,ya,yb,c):
    recon=.5*(F.l1_loss(ra,observed)+F.l1_loss(rb,observed))
    cross=.5*(F.l1_loss(ra[:,c['B']],yb)+F.l1_loss(rb[:,c['A']],ya))
    consistency=F.l1_loss(xa,xb)
    return 250000.*recon+5256.832431235398*cross+consistency

class EcoliLikelihood(nn.Module):
    def __init__(self,config,calibration):
        super().__init__();self.c=config
        with np.load(Path(calibration)/'ecoli_noise.npz') as a:
            for name,key in [('E','matrix83'),('vcurves','variance83')]:self.register_buffer(name,torch.from_numpy(a[key].copy()))
        with np.load(Path(calibration)/'ecoli_likelihood.npz') as a:
            for name,key in [('grid','grid_signal_DN'),('probs','nll'),('qmeans','quantized_mean')]:self.register_buffer(name,torch.from_numpy(a[key].copy()))
    def forward(self,x,y):
        c=self.c;nb=83
        bandindex=torch.arange(nb,device=x.device)[None,:,None,None]
        signal=torch.einsum('mk,bkhw->bmhw',self.E,x*255)
        mu=signal.clamp(0,float(self.grid[-1]));hi=torch.bucketize(mu.contiguous(),self.grid).clamp(1,len(self.grid)-1);lo=hi-1
        weight=(mu-self.grid[lo])/(self.grid[hi]-self.grid[lo])
        observed=(y*255).round().long().clamp(0,255)
        low=self.probs[bandindex,lo,observed];high=self.probs[bandindex,hi,observed]
        nll=(1-weight)*low+weight*high
        quantized_mean=(1-weight)*self.qmeans[bandindex,lo]+weight*self.qmeans[bandindex,hi]
        variance=(1-weight.detach())*self.vcurves[bandindex,lo]+weight.detach()*self.vcurves[bandindex,hi]
        nll=nll+(signal-float(self.grid[-1])).clamp_min(0)/torch.sqrt(self.vcurves[:,-1][None,:,None,None]+1)
        inputsignal=y[:,c['bands_for_loss_balance']].detach().amax(1,keepdim=True)*255
        def balanced(z):
            z=z.mean(1,keepdim=True);terms=[z.mean()]
            for low,high in [(1,4),(4,20),(20,1e9)]:
                mask=(inputsignal>=low)&(inputsignal<high)
                if mask.any():terms.append(z[mask].mean())
            return sum(terms)/len(terms)
        losses=[];fluxes=[]
        for left,right in zip(c['view_cuts'][:-1],c['view_cuts'][1:]):
            losses.append(balanced(nll[:,left:right]));flux=[]
            for size in [16,64,128]:
                residual=F.avg_pool2d(quantized_mean[:,left:right]-y[:,left:right]*255,size)
                std=torch.sqrt(F.avg_pool2d(variance[:,left:right].detach(),size)/(size*size)+.25**2)
                flux.append(F.huber_loss(residual/std,torch.zeros_like(residual),delta=1.))
            fluxes.append(sum(flux)/len(flux))
        main=.5*losses[0]+.5*sum(losses[1:])/4
        flux=.5*fluxes[0]+.5*sum(fluxes[1:])/4
        return main+.2*balanced(nll[:,c['bands_for_loss_balance']])+.1*flux

def capped_gradients(observed,adversarial,parameters,ratio=.1):
    params=[p for p in parameters if p.requires_grad]
    a=torch.autograd.grad(observed,params,retain_graph=True,allow_unused=True)
    b=torch.autograd.grad(adversarial,params,allow_unused=True)
    norm=lambda xs:torch.sqrt(sum(v.detach().double().square().sum() for v in xs if v is not None))
    na,nb=norm(a),norm(b)
    if not torch.isfinite(na) or not torch.isfinite(nb) or na<=0:raise RuntimeError('Invalid generator gradients')
    scale=torch.minimum(torch.ones_like(nb),ratio*na/(nb+1e-30))
    for p,ga,gb in zip(params,a,b):
        p.grad=None if ga is None and gb is None else (torch.zeros_like(p) if ga is None else ga.detach())+scale.to(p.dtype)*(torch.zeros_like(p) if gb is None else gb.detach())
    return float(scale*nb/na)

def critic_losses(critic,real,fake,kind,discriminator=True):
    if discriminator:
        r=critic(real);f=critic(fake.detach())
        if kind=='BCE with logits':return .5*(F.binary_cross_entropy_with_logits(r,torch.ones_like(r))+F.binary_cross_entropy_with_logits(f,torch.zeros_like(f)))
        return .5*(F.mse_loss(r,torch.ones_like(r))+F.mse_loss(f,torch.zeros_like(f)))
    f=critic(fake)
    return F.binary_cross_entropy_with_logits(f,torch.ones_like(f)) if kind=='BCE with logits' else F.mse_loss(f,torch.ones_like(f))
