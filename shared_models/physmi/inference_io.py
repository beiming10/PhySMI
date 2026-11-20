"""Input/output helpers and optional overlap inference for the published models."""
from pathlib import Path
import json
import numpy as np
import torch
from .models import predict,prepare_input


def read_input(path):
    path=Path(path)
    if path.suffix.lower()=='.npy':return np.load(path,allow_pickle=False)
    if path.suffix.lower() in ('.tif','.tiff'):
        import tifffile
        return tifffile.imread(path)
    raise ValueError('Input must be NPY or TIFF')


def starts(length,tile,stride):
    if tile<=0 or stride<=0 or stride>tile:raise ValueError('Invalid tile or stride')
    if length<=tile:return [0]
    positions=list(range(0,length-tile+1,stride))
    if positions[-1]!=length-tile:positions.append(length-tile)
    return positions


def infer_array(model,array,config,mode='normalized',panel=None,offset=None,device='cpu',
                tile_size=None,tile_overlap=None):
    image=prepare_input(array,config,mode,offset)
    if image.ndim==3:image=image[None]
    settings=config.get('inference',{})
    explicit_full_frame=tile_size==0
    tile_size=settings.get('tile_size',0) if tile_size is None else tile_size
    tile_overlap=settings.get('tile_overlap',0) if tile_overlap is None else tile_overlap
    if explicit_full_frame:tile_overlap=0
    if tile_size<0 or tile_overlap<0 or (tile_size and tile_overlap>=tile_size):
        raise ValueError('Require 0 <= overlap < tile size, or tile size 0 for full-frame inference')
    if not tile_size and tile_overlap:raise ValueError('Full-frame inference has no overlap')
    if config['known_panel']:
        if panel is None:raise ValueError('The live-cell model requires an experimental staining panel')
        panel=torch.as_tensor(panel,device=device,dtype=torch.float32).reshape(1,-1)
        if panel.shape!=(1,len(config['components'])) or not torch.all((panel==0)|(panel==1)):
            raise ValueError('Invalid binary staining panel')
    elif panel is not None:
        raise ValueError('This model does not use a staining-panel input')
    outputs=[]
    with torch.inference_mode():
        for frame in image:
            height,width=frame.shape[-2:]
            if not tile_size or (height<=tile_size and width<=tile_size):
                _,out=predict(model,torch.from_numpy(np.ascontiguousarray(frame[None])).to(device),config,panel)
                value=out[0].cpu().numpy()*config['unit']
            else:
                size_y,size_x=min(tile_size,height),min(tile_size,width)
                stride=tile_size-tile_overlap
                ys=starts(height,size_y,min(stride,size_y));xs=starts(width,size_x,min(stride,size_x))
                window=(np.hanning(size_y+2)[1:-1,None]*np.hanning(size_x+2)[None,1:-1]
                        if tile_overlap else np.ones((size_y,size_x)))
                total=np.zeros((len(config['components']),height,width),np.float64)
                weight=np.zeros((height,width),np.float64)
                for y in ys:
                    for x in xs:
                        tile=np.ascontiguousarray(frame[None,:,y:y+size_y,x:x+size_x])
                        _,out=predict(model,torch.from_numpy(tile).to(device),config,panel)
                        out=out[0].cpu().numpy()*config['unit']
                        if not np.isfinite(out).all():raise RuntimeError('Nonfinite model output')
                        total[:,y:y+size_y,x:x+size_x]+=out*window
                        weight[y:y+size_y,x:x+size_x]+=window
                if not np.all(weight>0):raise RuntimeError('Incomplete tile coverage')
                value=(total/weight).astype(np.float32)
            if not np.isfinite(value).all():raise RuntimeError('Nonfinite model output')
            outputs.append(value)
    return np.ascontiguousarray(np.stack(outputs),dtype=np.float32)


def write_prediction(path,output,config,panel=None):
    path=Path(path)
    if path.suffix.lower() not in ('.npy','.tif','.tiff'):raise ValueError('Output must be NPY or TIFF')
    if path.exists() or path.with_suffix('.json').exists():raise FileExistsError(path)
    if output.ndim!=4 or output.shape[1]!=len(config['components']) or not np.isfinite(output).all():
        raise ValueError('Expected finite T,K,Y,X output')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.suffix.lower()=='.npy':np.save(path,output,allow_pickle=False)
    else:
        import tifffile
        tifffile.imwrite(path,output,photometric='minisblack',metadata={'axes':'TCYX'},compression='deflate')
    path.with_suffix('.json').write_text(json.dumps({
        'components':config['components'],'axes':'T,K,Y,X','units':'network abundance DN',
        'model':config['name'],'weight_SHA256':config['weight_SHA256'],
        'known_panel':panel,'postprocessing':None},indent=2)+'\n',encoding='utf-8')
