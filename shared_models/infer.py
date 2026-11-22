"""Run a configured model on channel-first image arrays."""
import argparse
from pathlib import Path
import json
import torch
from physmi.models import load_config,build_model
from physmi.inference_io import read_input,infer_array,write_prediction

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
    p.add_argument('--tile-size',type=int,help='Override the configured tile size; 0 uses a full frame')
    p.add_argument('--tile-overlap',type=int,help='Override overlap in pixels')
    a=p.parse_args();torch.set_num_threads(2);c=load_config(a.model)
    if a.output.exists():raise FileExistsError(a.output)
    image=read_input(a.input)
    panel=None
    if a.panel and not c['known_panel']:
        p.error('--panel is used only by models with staining-panel conditioning')
    if c['known_panel']:
        if not a.panel:p.error('--panel is required for the live-cell model')
        panel=[float(x) for x in a.panel.split(',')]
    model=build_model(c,device=a.device,weight_path=a.weights).eval()
    if a.weights:
        import hashlib
        c=dict(c,weight_SHA256=hashlib.sha256(a.weights.read_bytes()).hexdigest())
    output=infer_array(model,image,c,mode=a.mode,panel=panel,
                       offset=[float(x) for x in a.offset.split(',')] if a.offset else None,
                       device=a.device,tile_size=a.tile_size,tile_overlap=a.tile_overlap)
    write_prediction(a.output,output,c,panel)
    print(a.output)
if __name__=='__main__':main()
