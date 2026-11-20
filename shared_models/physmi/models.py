"""Construct the four published generators without changing their state keys."""
from pathlib import Path
import hashlib
import json
import torch
from torch import nn
import torch.nn.functional as F
from .core import FSR_Unmixing, UnmixingHead
from .static_panel import PanelFSR as StaticPanel
from .live_panel import PanelFSR as LivePanel
from .positive_head import PositiveDyeHead

ROOT=Path(__file__).resolve().parents[1]

def load_config(name_or_path):
    p=Path(name_or_path)
    if not p.is_file():p=ROOT/'configs'/(str(name_or_path)+'.json')
    return json.loads(p.read_text(encoding='utf-8'))

class PositiveOrganelleHead(UnmixingHead):
    def forward(self,features):
        abundance=F.softplus(self.fc_abundance(features)*self.unit)/self.unit
        matrix=torch.cat([self.endmember_dict_signal,self.endmember_dict_background],0)
        return torch.einsum('kc,bkhw->bchw',matrix,abundance),abundance

class ThreeInput(nn.Module):
    def __init__(self,core):
        super().__init__();self.core=core
    def forward(self,image,branch='A'):
        if image.ndim!=4 or image.shape[1]!=3 or branch not in ('A','B'):
            raise ValueError('Expected three measured image channels and A/B branch')
        marker=torch.full_like(image[:,:1],0. if branch=='A' else 1.)
        return self.core(torch.cat([image,marker],1))

def build_model(config,weights=True,device='cpu',weight_path=None):
    c=load_config(config) if not isinstance(config,dict) else config
    matrix=torch.tensor(c['response'],dtype=torch.float32)
    inputs=4 if c['name']=='cells' else c['input_channels']
    k=len(c['components'])
    core=FSR_Unmixing(inputs,c['reconstruction_channels'],32,3,k,
                      matrix.T[:k-1],matrix.T[k-1],True,True)
    if c['name']=='static':model=StaticPanel(core,matrix)
    elif c['name']=='live':model=LivePanel(core)
    elif c['name']=='cells':
        core.unmixing_head.__class__=PositiveOrganelleHead
        core.unmixing_head.unit=c['unit'];model=ThreeInput(core)
        if not weights:
            with torch.no_grad():
                core.unmixing_head.fc_abundance.weight.mul_(.0001)
                core.unmixing_head.fc_abundance.bias.fill_(torch.log(torch.expm1(torch.tensor(.1))).item()/c['unit'])
    elif c['name']=='ecoli':
        core.unmixing_head.__class__=PositiveDyeHead
        core.unmixing_head.softplus_tau=1/255;model=core
        if not weights:
            with torch.no_grad():
                core.unmixing_head.fc_abundance.weight.mul_(.01)
                core.unmixing_head.fc_abundance.bias.fill_(torch.log(torch.expm1(torch.tensor(.1))).item()/255)
    else:raise ValueError('Unknown spectral panel')
    if weights:
        p=Path(weight_path) if weight_path else ROOT/c['weight']
        if not weight_path and hashlib.sha256(p.read_bytes()).hexdigest()!=c['weight_SHA256']:
            raise ValueError('Weight file does not match its spectral configuration')
        state=torch.load(p,map_location='cpu',weights_only=True)
        model.load_state_dict(state.get('model',state),strict=True)
    return model.to(device)

def predict(model,image,c,panel=None,branch='A'):
    if image.ndim!=4 or image.shape[1]!=c['input_channels']:
        raise ValueError('Incorrect number of measured inputs')
    if c['name']=='static':
        return model(image,branch=0 if branch=='A' else 1)
    if c['name']=='live':
        if panel is None:raise ValueError('Supply the experimental staining-panel vector')
        return model(image,panel)
    return model(image,branch) if c['name']=='cells' else model(image)

def prepare_input(array,c,mode='normalized',offset=None):
    import numpy as np
    a=np.asarray(array,dtype=np.float32)
    if a.ndim not in (3,4):raise ValueError('Input must have CHW or TCHW axes')
    axis=a.ndim-3
    if mode=='normalized':
        if a.shape[axis]!=c['input_channels']:raise ValueError('Normalized input must already contain selected A channels')
        return a
    if a.shape[axis]==c['reconstruction_channels']:
        if c['name'] in ('static','live'):a=np.take(a,c['A'],axis=axis)
        else:a=np.stack([np.take(a,g,axis=axis).sum(axis=axis) for g in c['A_groups']],axis=axis)
    elif a.shape[axis]!=c['input_channels']:
        raise ValueError('Raw input has neither the selected nor full channel count')
    if c['name'] in ('static','live'):
        off=np.asarray(offset if offset is not None else np.asarray(c['offset_DN'])[c['A']],np.float32)
        if len(off)!=c['input_channels']:raise ValueError('Provide one offset for each selected channel')
        shape=[1]*a.ndim;shape[axis]=len(off);a=a-off.reshape(shape)
        if c['name']=='live':a=np.maximum(a,0)
    return a/c['unit']
