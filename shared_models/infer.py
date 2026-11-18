"""Run one of the four fixed spectral-panel models on CHW/TCHW images."""
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from physmi.models import load_config,build_model,predict,prepare_input

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['static','live','cells','ecoli'],required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['raw','normalized'],default='normalized')
    p.add_argument('--panel',help='Known staining panel, e.g. 1,1,1,1,1 or live four-colour 1,1,1,1,0')
    p.add_argument('--offset',help='Selected-channel offsets in acquisition DN; needed when a live acquisition has a different offset')
    p.add_argument('--weights',type=Path)
    p.add_argument('--device',default='cpu')
    a=p.parse_args();torch.set_num_threads(2);c=load_config(a.model)
    if a.output.exists():raise FileExistsError(a.output)
    if a.input.suffix.lower() in ['.tif','.tiff']:
        import tifffile
        image=tifffile.imread(a.input)
    else:image=np.load(a.input,allow_pickle=False)
    image=prepare_input(image,c,a.mode,[float(x) for x in a.offset.split(',')] if a.offset else None)
    if image.ndim==3:image=image[None]
    panel=None
    if c['known_panel']:
        if not a.panel:p.error('--panel is required for static and live models')
        panel=torch.tensor([[float(x) for x in a.panel.split(',')]],device=a.device)
        if panel.shape!=(1,len(c['components'])) or not torch.all((panel==0)|(panel==1)):p.error('Invalid panel')
    model=build_model(c,device=a.device,weight_path=a.weights).eval();outputs=[]
    with torch.inference_mode():
        for frame in image:
            _,out=predict(model,torch.from_numpy(np.ascontiguousarray(frame[None])).to(a.device),c,panel)
            if not torch.isfinite(out).all():raise RuntimeError('Nonfinite prediction')
            outputs.append(out[0].cpu().numpy()*c['unit'])
    a.output.parent.mkdir(parents=True,exist_ok=True);np.save(a.output,np.stack(outputs))
    a.output.with_suffix('.json').write_text(json.dumps({'components':c['components'],'axes':'T,K,Y,X','units':'network abundance DN','model':c['name'],'known_panel':panel.cpu().tolist()[0] if panel is not None else None,'postprocessing':None},indent=2))
    print(a.output)
if __name__=='__main__':main()
