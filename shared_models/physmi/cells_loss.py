"""Quantized, censored observation likelihood."""
def objective(c,xa,xb,y,matrix,sig,alpha,input_b):
 import torch
 import torch.nn.functional as F
 def nll(x):
  mu=torch.einsum('ck,bkhw->bchw',matrix,x)*c['unit'];variance=sig[:,:,None,None].square()+alpha[:,None,None,None]*mu.clamp_min(0)
  sd=variance.double().sqrt();mu=mu.double();z=y.double();lo=(z-.5-mu)/sd;hi=(z+.5-mu)/sd;right=lo>0
  lh=torch.special.log_ndtr(torch.where(right,-lo,hi));ll=torch.special.log_ndtr(torch.where(right,-hi,lo))
  logp=lh+torch.log(-torch.expm1((ll-lh).clamp_max(-1e-14)));logp=torch.where(z==0,torch.special.log_ndtr(hi),logp);logp=torch.where(z==255,torch.special.log_ndtr(-lo),logp)
  return -logp
 def balance(v):
  # Input-conditioned weights, independent of the target noise realization.
  brightness=input_b.amax(1,keepdim=True);v=v.mean(1,keepdim=True);loss=[v.mean()]
  for low,high in [(0,2),(2,16),(16,1e20)]:
   mask=(brightness>=low)&(brightness<high)
   if mask.any():loss.append(v[mask].mean())
  return sum(loss)/len(loss)
 na,nb=nll(xa),nll(xb);main=(2*balance(na)+balance(nb))/3
 cross=.5*(balance(na[:,c['B']])+balance(nb[:,c['A']]));consistent=F.smooth_l1_loss(xa,xb);loss=main+.2*cross+.1*consistent
 assert torch.isfinite(loss),'Finite censored quantized observation likelihood'
 return loss,{'observed_nll':float(main.detach()),'cross_observation_nll':float(cross.detach()),'branch_consistency':float(consistent.detach())}
