"""Configuration, batches, and evaluation utilities."""
from pathlib import Path
import sys
import math
import json
import numpy as np
import torch
sys.path.append(str(Path(__file__).resolve().parents[1]/'shared_models'))
from physmi.models import load_config,build_model
from physmi.discriminator import StCDiscriminator
from simulation import ROOT,read,file_digest

def configuration(model):
    if model not in ('cells','ecoli'):raise ValueError('Choose cells or ecoli')
    c=load_config(model)
    c['recipe']=read(ROOT/'configs'/(model+'.json'))
    c['seed']=c['recipe']['seed']
    c['steps']=sum(p['steps'] for p in c['recipe']['phases'])
    return c

def build(c,device):
    seed=c['seed'];torch.manual_seed(seed)
    if str(device).startswith('cuda'):torch.cuda.manual_seed_all(seed)
    g=build_model(c,weights=False,device=device)
    head=g.core.unmixing_head if c['name']=='cells' else g.unmixing_head
    with torch.no_grad():head.fc_abundance.bias.fill_(math.log(math.expm1(.1))/c['unit'])
    if c['name']=='ecoli':
        torch.manual_seed(seed+3137)
        if str(device).startswith('cuda'):torch.cuda.manual_seed_all(seed+3137)
    d=StCDiscriminator(in_channels=1,ndf=64,n_layers=3).to(device)
    if c['name']=='ecoli':
        torch.manual_seed(seed+9191)
        if str(device).startswith('cuda'):torch.cuda.manual_seed_all(seed+9191)
    return g,d

def learning_rate(c,step):
    if not 1<=step<=c['steps']:raise ValueError('Step outside configured schedule')
    for phase in c['recipe']['phases']:
        if step<=phase['steps']:
            low=phase['learning_rate_end'];high=phase['learning_rate_start']
            return low+(high-low)*.5*(1+math.cos(math.pi*(step-1)/(phase['steps']-1)))
        step-=phase['steps']

def batch(data,manifest,indices,c,device):
    result={}
    required=({'observed28','input_A3','input_B3','unlabelled_prior_DN','sigma_DN','shot_alpha'}
              if c['name']=='cells' else {'observed83','unlabelled_prior_DN'})
    for index in indices:
        row=manifest['splits']['train'][int(index)]
        with np.load(Path(data)/row['file'],allow_pickle=False) as z:
            if set(z.files)!=required:raise ValueError('Training payload must contain observations and independent priors only')
            for key in required:
                value=z[key].copy()
                if key=='unlabelled_prior_DN':value=value/c['unit']
                if key=='observed83':value=value.astype(np.float32)/c['unit']
                result.setdefault(key,[]).append(value)
    result={k:torch.from_numpy(np.stack(v).astype(np.float32)).to(device) for k,v in result.items()}
    return result

def integrate(observed,c):
    return torch.stack([observed[:,group].sum(1) for group in c['A_groups']],1)

def scores(prediction,reference):
    p=np.asarray(prediction,np.float64);r=np.asarray(reference,np.float64)
    error=np.sqrt(np.mean((p-r)**2,(1,2)))
    rms=np.sqrt(np.mean(r*r,(1,2)))
    slope=np.divide((p*r).sum((1,2)),(r*r).sum((1,2)),out=np.full(len(r),np.nan),where=rms>0)
    return [{'RMSE':float(e),'reference_RMS':float(s),'output_RMS':float(np.sqrt(np.mean(v*v))),
             'amplitude_slope':float(a) if np.isfinite(a) else None} for e,s,a,v in zip(error,rms,slope,p)]
