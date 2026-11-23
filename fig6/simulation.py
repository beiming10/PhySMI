"""Generate independent observation and image-prior samples."""
from pathlib import Path
import json
import hashlib
import math
import copy
import numpy as np
import tifffile
from cells_simulation import generate_cells, independent_inputs
from ecoli_spatial import SpatialSources

ROOT=Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def file_digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):value.update(block)
    return value.hexdigest()

def load_library(root,model):
    pools={name:[[] for _ in range(5)] for name in (['train','holdout'] if model=='cells' else ['train','val','test'])}
    manifest=read(ROOT/'library.json')
    for row in manifest['sources']:
        path=Path(root)/row['file']
        image=tifffile.imread(path)
        if image.shape!=(256,256) or image.dtype!=np.float32 or not np.isfinite(image).all() or image.min()<0 or image.max()<=0:
            raise ValueError('Expected a finite nonnegative 256-pixel texture')
        pools[row[model+'_split']][row['class']].append((row['id'],image))
    return pools

def quantize(signal,noise=0.):
    return np.clip(np.rint(signal+noise),0,255).astype(np.uint8)

class Simulator:
    def __init__(self,model,library,config):
        self.model=model;self.config=config
        self.settings=read(ROOT/'simulation.json')[model]
        self.pools=load_library(library,model)
        self.matrix=np.asarray(config['response'],np.float32)
        if model=='ecoli':self.spatial=EcoliSimulator(self.settings,self.matrix)

    def sample(self,split,index):
        if split not in ('train','val','test'):raise ValueError('Unknown split')
        c=self.config
        if self.model=='cells':
            plan=self.settings['plans'][split]
            if index>=len(plan):raise ValueError('Index exceeds the configured split')
            spec=plan[index]
            pool=self.pools['train' if split=='train' else 'holdout']
            y,p,x,sigma,alpha,meta=generate_cells(c,pool,{'matrix':self.matrix},split,index,
                                               None if split=='train' else spec['profile'],spec['target'])
            (a,b),seeds=independent_inputs(c,x,self.matrix,sigma,alpha,split,index)
            payload={'observed28':y,'input_A3':a,'input_B3':b,'unlabelled_prior_DN':p,
                     'sigma_DN':sigma,'shot_alpha':np.array(alpha,np.float32)}
        else:
            payload,x=self.spatial.sample(self.pools[split],split,index)
        if split!='train':payload['reference_DN']=x
        assert 'reference_DN' not in payload or split!='train'
        return payload

class EcoliSimulator(SpatialSources):
    def __init__(self,settings,matrix):
        self.settings=settings;self.matrix=matrix
        self.pairs=settings['near_pairs'];self.bands=settings['bands']
        self.yy,self.xx=np.mgrid[:128,:128]
        self.background=np.asarray(settings['background23'],np.float32)
        self.sigma=np.asarray(settings['read_sigma23'],np.float32)
        self.effective_sigma=np.sqrt(self.sigma*self.sigma+1/12)
        with np.load(ROOT.parent/'shared_models/calibration/ecoli_noise.npz',allow_pickle=False) as z:
            self.noise={k:z[k].copy() for k in z.files}

    def quality(self,x,y,q,eta):
        checks=[]
        for k in np.flatnonzero(x.max((1,2))>0):
            peak=float(x[k].max());core=x[k]>=.2*peak
            contribution=self.matrix[:,k,None,None]*x[k][None]
            without=quantize(np.maximum(y-contribution,0),eta)
            detect=float(np.any(q[self.bands]!=without[self.bands],axis=0)[core].mean())
            snr=float(np.max(self.matrix[self.bands,k]*peak/self.effective_sigma[self.bands]))
            identities=[]
            for a,b in self.pairs:
                j=b if k==a else a if k==b else None
                if j is not None and np.any(x[j]>0):
                    swapped=quantize(np.maximum(y-contribution+self.matrix[:,j,None,None]*x[k][None],0),eta)
                    identities.append(float(np.any(q[self.bands]!=swapped[self.bands],axis=0)[core].mean()))
            checks.append({'dye':int(k),'passed':detect>=.5 and snr>=2.5 and all(value>=.1 for value in identities)})
        return checks

    def repaired(self,original,prior,split,index):
        s=self.settings
        rng=np.random.default_rng(s['gain_seed']+{'train':100000,'val':200000,'test':300000}[split]+index)
        x=original.copy();signal=np.einsum('mk,khw->mhw',self.matrix,x,optimize=True)
        desired=float(np.clip(rng.lognormal(np.log(s['median_peak']),.4),25,210))
        gain=desired/max(float(signal[self.bands].max()),1e-12) if x.max()>0 else 1.
        x*=gain
        eta=rng.standard_normal((23,128,128),dtype=np.float32)*self.sigma[:,None,None]
        for iteration in range(9):
            signal=np.einsum('mk,khw->mhw',self.matrix,x,optimize=True)
            x*=min(1.,230/max(float(signal.max()),1e-12))
            signal=np.einsum('mk,khw->mhw',self.matrix,x,optimize=True)
            y=signal+self.background[:,None,None]
            checks=self.quality(x,y,quantize(y,eta),eta)
            if all(row['passed'] for row in checks) or iteration==8:break
            for row in checks:
                if not row['passed']:x[row['dye']]*=1.5
        if prior is not None:
            signal=np.einsum('mk,khw->mhw',self.matrix,prior,optimize=True)
            gain=min(desired/max(float(signal[self.bands].max()),1e-12),230/max(float(signal.max()),1e-12)) if prior.max()>0 else 1.
            prior=prior*gain
        return x,prior

    def observe(self,signal,rng):
        a=self.noise
        mu=np.maximum(signal,0);variance=np.empty_like(signal,dtype=np.float32)
        for j in range(len(signal)):variance[j]=np.interp(signal[j],a['grid_signal_DN'],a['variance83'][j])
        sigma=a['read_sigma83'][:,None,None]
        shot=np.maximum(variance-sigma*sigma,.01*mu)
        theta=np.divide(shot,2*mu,out=np.full_like(mu,.5),where=mu>1e-8)
        theta=np.maximum(theta,1e-5);rate=mu/theta
        if float(rate.max())>20000:raise ValueError('Noise event rate exceeds the configured range')
        events=rng.poisson(rate);shot_signal=rng.gamma(events,scale=theta)
        return quantize(shot_signal+a['offset83'][:,None,None],rng.standard_normal(mu.shape)*sigma)

    def sample(self,pool,split,index):
        s=self.settings
        rng=np.random.default_rng(s['spatial_seed']+{'train':100000,'val':200000,'test':300000}[split]+index)
        if split=='test':
            spec=copy.deepcopy(s['test_specs'][index])
            spec.update(sigma=[0.,.7,1.3][index%3],factor=2 if index%4==3 else 1)
        else:spec=self.spec_for(index,rng)
        x,_=self.draw(pool,rng,spec);prior=None
        if split=='train':
            prng=np.random.default_rng(s['spatial_seed']+400000+index)
            prior,_=self.draw(pool,prng,self.spec_for(index+217,prng))
        x,prior=self.repaired(x,prior,split,index)
        matrix=self.noise['matrix83']
        clean=np.einsum('mk,khw->mhw',matrix,x,optimize=True).astype(np.float32)
        gain=min(1.,230/max(float(clean.max()),1e-9));x*=gain;clean*=gain
        rng=np.random.default_rng(s['seed']+{'train':100000,'val':200000,'test':300000}[split]+index)
        payload={'observed83':self.observe(clean,rng)}
        if prior is not None:
            signal=np.einsum('mk,khw->mhw',matrix,prior,optimize=True)
            prior*=min(1.,230/max(float(signal.max()),1e-9))
            payload['unlabelled_prior_DN']=prior
        return payload,x
