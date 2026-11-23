"""Evaluate all component outputs on held-out simulated observations."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import torch
import tifffile
from common import configuration,build_model,integrate,scores,read

def evaluate(model,c,data,split,device):
    manifest=read(Path(data)/'manifest.json')
    if split not in ('val','test'):raise ValueError('References are restricted to held-out samples')
    was_training=model.training;model.eval();rows=[]
    try:
        for case in manifest['splits'][split]:
            with np.load(Path(data)/case['file'],allow_pickle=False) as z:
                reference=z['reference_DN'].copy()
                if c['name']=='cells':
                    image=torch.from_numpy(z['input_A3'][None].astype(np.float32)/c['unit']).to(device)
                else:
                    observed=torch.from_numpy(z['observed83'][None].astype(np.float32)/c['unit']).to(device)
                    image=integrate(observed,c)
            with torch.inference_mode():
                _,output=model(image,'A') if c['name']=='cells' else model(image)
            prediction=output[0].cpu().numpy()*c['unit']
            if not np.isfinite(prediction).all():raise ValueError('Nonfinite prediction')
            for name,values in zip(c['components'],scores(prediction,reference)):
                rows.append(dict(id=case['id'],component=name,**values))
    finally:model.train(was_training)
    return rows

def write_scores(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['cells','ecoli'],required=True)
    parser.add_argument('--data',type=Path,default=Path('fig6_data'))
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--split',choices=['val','test'],default='test')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args();c=configuration(args.model)
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(2)
    model=build_model(c,weight_path=args.checkpoint,device=args.device)
    rows=evaluate(model,c,args.data/args.model,args.split,args.device)
    write_scores(args.output,rows);print(args.output,flush=True)

if __name__=='__main__':main()
