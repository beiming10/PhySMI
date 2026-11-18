"""Create a small observation-only example from a user's single-stain textures.

This compact generator illustrates the training data interface. It does not
replay the acquisition-specific image sampling of the released checkpoints.
"""
import argparse
from pathlib import Path
import json
import numpy as np
from scipy.ndimage import zoom,gaussian_filter
from physmi.models import ROOT,load_config

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['static','live','cells','ecoli'],required=True)
    p.add_argument('--library',type=Path,required=True,help='Directory of nonnegative single-channel NPY/TIFF images')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=4);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    files=sorted(p for p in a.library.rglob('*') if p.suffix.lower() in ['.npy','.tif','.tiff'])
    if len(files)<2:raise ValueError('At least two independent source images are required')
    c=load_config(a.model);rng=np.random.default_rng(a.seed);k=len(c['components']);unit=c['unit']
    matrix=np.asarray(c['response'],np.float32)
    if a.model=='ecoli':
        with np.load(ROOT/'calibration/ecoli_noise.npz') as z:matrix=z['matrix83'].copy()
    def draw(exclude=None):
        opts=[p for p in files if p!=exclude];path=opts[int(rng.integers(len(opts)))]
        if path.suffix=='.npy':im=np.load(path,allow_pickle=False)
        else:
            import tifffile
            im=tifffile.imread(path)
        if im.ndim!=2 or im.min()<0 or not np.isfinite(im).all():raise ValueError('Expected a finite nonnegative grayscale texture')
        im=zoom(im.astype(np.float32),(128/im.shape[0],128/im.shape[1]),order=1)
        im=np.rot90(im,int(rng.integers(4)));im=im/(np.sqrt(np.mean(im.astype(float)**2))+1e-12)
        return np.ascontiguousarray(im,dtype=np.float32),path
    a.output.mkdir(parents=True)
    for index in range(a.count):
        x=[];prior=[]
        for component in range(k):
            texture,source=draw();other,_=draw(source);gain=float(np.exp(rng.uniform(-2.,0.)))
            x.append(texture*gain);prior.append(other*gain)
        x=np.stack(x);prior=np.stack(prior)
        # Targets are used only by the forward simulator, not stored in a batch.
        if a.model in ['static','live']:
            code=31 if index%2==0 else int(rng.integers(32));panel=np.array([(code>>j)&1 for j in range(k)],np.float32)
            x*=panel[:,None,None]
        clean=np.einsum('ck,khw->chw',matrix,x)
        scale=(.5 if a.model in ['static','live'] else 64/unit)/max(float(clean.max()),1e-12)
        x*=scale;prior*=scale;clean*=scale
        observed=clean.copy()
        batch={'observed':observed.astype(np.float32),'prior':prior.astype(np.float32),'gan':np.array(True)}
        if a.model in ['static','live']:
            # Independent repeat noise is attached to each measurement channel.
            observed+=rng.normal(0,.002,observed.shape).astype(np.float32)
            batch.update(observed=observed,input_a=observed[c['A']],input_b=observed[c['B']],panel=panel,prior_anchor=np.array(1.,np.float32))
        elif a.model=='cells':
            sigma=np.ones(28,np.float32);shot=.5
            def observe():return np.floor(np.clip(clean*unit+rng.normal(size=clean.shape)*np.sqrt(sigma[:,None,None]**2+shot*clean*unit),0,255)+.5).astype(np.float32)
            observed=observe();ya=observe();yb=observe()
            batch.update(observed=observed,input_a=np.stack([ya[g].sum(0) for g in c['A_groups']])/unit,input_b=yb[c['B']]/unit,sigma_DN=sigma,shot_alpha=np.array(shot,np.float32))
        else:
            # A noiseless quantized smoke-test example. Calibrated noisy training
            # batches are supplied through the same observed83 interface.
            observed=np.floor(np.clip(clean*255,0,255)+.5)/255
            batch.update(observed=observed,input_a=np.stack([observed[g].sum(0) for g in c['A_groups']]))
        np.savez_compressed(a.output/f'example_{index:03d}.npz',**batch)
    (a.output/'example_description.json').write_text(json.dumps({'purpose':'Small observation-only interface example, not a replay of a publication training set','component_GT_in_optimizer':False,'model':a.model,'seed':a.seed,'count':a.count},indent=2))
    print(a.output)
if __name__=='__main__':main()
