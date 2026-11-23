"""Prepare fixed observation-only training samples and held-out references."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import numpy as np
from common import configuration
from simulation import ROOT,Simulator,read,file_digest
from download import download

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['cells','ecoli'],required=True)
    parser.add_argument('--data',type=Path,default=Path('fig6_data'))
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--train-count',type=int)
    parser.add_argument('--val-count',type=int)
    parser.add_argument('--test-count',type=int)
    args=parser.parse_args()
    defaults={'cells':{'train':2048,'val':40,'test':64},'ecoli':{'train':1024,'val':64,'test':132}}
    counts={key:defaults[args.model][key] if getattr(args,key+'_count') is None else getattr(args,key+'_count') for key in defaults[args.model]}
    if args.workers<1 or any(not 1<=counts[k]<=defaults[args.model][k] for k in counts):parser.error('Invalid sample count or worker count')
    destination=args.data/args.model
    if destination.exists():raise FileExistsError(destination)
    if not args.offline:download(args.data)
    c=configuration(args.model);simulator=Simulator(args.model,args.data,c)
    destination.mkdir(parents=True)
    manifest={'model':args.model,'splits':{},'train_component_targets':False,
              'source_assignment':'library.json','counts':counts,'canonical_counts':counts==defaults[args.model]}
    def one(task):
        split,index=task
        payload=simulator.sample(split,index)
        path=destination/split/f'{split}_{index:04d}.npz'
        np.savez_compressed(path,**payload)
        return {'id':f'{split}_{index:04d}','file':str(path.relative_to(destination)).replace('\\','/'),'sha256':file_digest(path)}
    for split,count in counts.items():
        (destination/split).mkdir()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            rows=list(executor.map(one,[(split,i) for i in range(count)]))
        manifest['splits'][split]=rows
        print(split,len(rows),'samples',flush=True)
    if manifest['canonical_counts']:
        with np.load(ROOT/'orders.npz',allow_pickle=False) as z:order=z[args.model]
    else:
        rng=np.random.default_rng(c['seed']+900000);indices=[]
        while len(indices)<c['steps']*2:indices.extend(int(i) for i in rng.permutation(counts['train']))
        order=np.asarray(indices[:c['steps']*2],np.int32).reshape(c['steps'],2)
    np.save(destination/'order.npy',order,allow_pickle=False)
    manifest['order_sha256']=file_digest(destination/'order.npy')
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Prepared:',destination,flush=True)

if __name__=='__main__':main()
