"""Run the supplied input files with their matching models and acquisition settings."""
import argparse,json,re
from pathlib import Path
import torch
from physmi.models import load_config,build_model
from physmi.inference_io import read_input,infer_array,write_prediction


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path(__file__).resolve().parents[1]/'TestDATA')
    p.add_argument('--output',type=Path)
    p.add_argument('--model',choices=['static','live','cells','ecoli'])
    p.add_argument('--id',action='append',default=[],help='Select an example ID; may be repeated')
    p.add_argument('--list',action='store_true')
    p.add_argument('--format',choices=['npy','tif'],default='tif')
    p.add_argument('--device',default='cpu')
    args=p.parse_args();root=args.data.resolve()
    examples=json.loads((root/'examples.json').read_text(encoding='utf-8'))['examples']
    examples=[e for e in examples if (not args.model or e['model']==args.model) and (not args.id or e['id'] in args.id)]
    if not examples:p.error('No matching examples')
    if args.list:
        for e in examples:print(e['id'],e['model'],e['file'])
        return
    if args.output is None:p.error('--output is required unless --list is used')
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(2);models={};records=[]
    for e in examples:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',e['id']) or e['id'] in ['.','..']:
            raise ValueError('Invalid example ID')
        path=(root/e['file']).resolve()
        if not path.is_relative_to(root):raise ValueError('Example is outside the data directory')
        if e['model'] not in models:
            config=load_config(e['model']);model=build_model(config,device=args.device).eval()
            models[e['model']]=(config,model)
        config,model=models[e['model']]
        output=infer_array(model,read_input(path),config,mode=e['mode'],panel=e.get('panel'),
                           offset=e.get('offset'),device=args.device,
                           tile_size=e.get('tile_size'),tile_overlap=e.get('tile_overlap'))
        destination=args.output/e['model']/(e['id']+'.'+args.format)
        write_prediction(destination,output,config,e.get('panel'))
        records.append({'id':e['id'],'model':e['model'],'output':str(destination),
                        'shape':list(output.shape),'weight_SHA256':config['weight_SHA256']})
        print(e['id'],'->',destination,flush=True)
    (args.output/'results.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
