"""Train a fixed spectral panel from simulated observations and an independent image prior."""
from pathlib import Path
import argparse
import copy
import json
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from common import ROOT,configuration,build,batch,read,learning_rate,integrate,file_digest
from physmi.losses import cells_objective,EcoliLikelihood
from evaluate import evaluate,write_scores

def assign_gradients(observed,adversarial,parameters,ratio=.1):
    parameters=tuple(p for p in parameters if p.requires_grad)
    og=torch.autograd.grad(observed,parameters,retain_graph=True,allow_unused=True)
    ag=torch.autograd.grad(adversarial,parameters,allow_unused=True)
    norm=lambda values:math.sqrt(sum(float(v.detach().double().square().sum()) for v in values if v is not None))
    main,adv=norm(og),norm(ag)
    if not math.isfinite(main) or main<=0 or not math.isfinite(adv):raise RuntimeError('Invalid gradient')
    scale=min(1.,ratio*main/max(adv,1e-30))
    for p,a,b in zip(parameters,og,ag):
        p.grad=None if a is None and b is None else (torch.zeros_like(p) if a is None else a)+(0 if b is None else scale*b)
    return scale*adv/main

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['cells','ecoli'],required=True)
    parser.add_argument('--data',type=Path,default=Path('fig6_data'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--stop-after',type=int)
    parser.add_argument('--validate-every',type=int)
    args=parser.parse_args();c=configuration(args.model)
    if args.output.exists():raise FileExistsError(args.output)
    stop=c['steps'] if args.stop_after is None else args.stop_after
    if not 1<=stop<=c['steps']:parser.error('Stop must lie within the configured training budget')
    every=c['recipe']['validation_interval'] if args.validate_every is None else args.validate_every
    if every<0:parser.error('Validation interval must be nonnegative')
    data=args.data/args.model;manifest=read(data/'manifest.json')
    if manifest['model']!=args.model or manifest['train_component_targets']:raise ValueError('Invalid training manifest')
    if file_digest(data/'order.npy')!=manifest['order_sha256']:raise ValueError('Training order checksum mismatch')
    order=np.load(data/'order.npy',allow_pickle=False)
    if order.shape!=(c['steps'],2):raise ValueError('Expected a two-sample training schedule')
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    random.seed(c['seed']);g,d=build(c,args.device)
    initial_lr=c['recipe']['phases'][0]['learning_rate_start']
    betas=tuple(c['recipe']['Adam_betas'])
    go=torch.optim.Adam((p for p in g.parameters() if p.requires_grad),lr=initial_lr,betas=betas)
    do=torch.optim.Adam(d.parameters(),lr=initial_lr,betas=betas)
    rng=np.random.default_rng(c['seed']+(7 if args.model=='cells' else 23))
    matrix=torch.tensor(c['response'],dtype=torch.float32,device=args.device)
    likelihood=EcoliLikelihood(c,ROOT.parent/'shared_models/calibration').to(args.device) if args.model=='ecoli' else None
    args.output.mkdir(parents=True)
    (args.output/'configuration.json').write_text(json.dumps(c,indent=2)+'\n')
    start=0

    def snapshot(step):
        return {'model':copy.deepcopy(g.state_dict()),'D':copy.deepcopy(d.state_dict()),
                'G_optimizer':copy.deepcopy(go.state_dict()),'D_optimizer':copy.deepcopy(do.state_dict()),
                'step':step,'configuration':c,'torch_rng':torch.get_rng_state().clone(),
                'dataset_manifest_sha256':file_digest(data/'manifest.json'),'order_sha256':manifest['order_sha256'],
                'cuda_rng':[r.clone() for r in torch.cuda.get_rng_state_all()] if str(args.device).startswith('cuda') else None,
                'numpy_rng':copy.deepcopy(rng.bit_generator.state),'python_rng':random.getstate()}

    def restore(state):
        for obj,key in [(g,'model'),(d,'D'),(go,'G_optimizer'),(do,'D_optimizer')]:obj.load_state_dict(state[key])
        torch.set_rng_state(state['torch_rng'].cpu())
        if str(args.device).startswith('cuda') and state['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all([r.cpu() for r in state['cuda_rng']])
        rng.bit_generator.state=state['numpy_rng'];random.setstate(state['python_rng'])
        d.requires_grad_(True);g.train();d.train()

    if args.resume:
        state=torch.load(args.resume,map_location='cpu',weights_only=True)
        if state['configuration']!=c:raise ValueError('Resume configuration differs')
        if state['dataset_manifest_sha256']!=file_digest(data/'manifest.json') or state['order_sha256']!=manifest['order_sha256']:
            raise ValueError('Resume dataset or order differs')
        restore(state);start=int(state['step'])
        if start>=stop:raise ValueError('Resume step must precede the stopping point')

    def checkpoint(step):
        target=args.output/f'checkpoint_{step:06d}.pt'
        temporary=target.with_suffix('.part')
        torch.save(snapshot(step),temporary);temporary.replace(target)
        return target

    with (args.output/'history.jsonl').open('w') as log:
        for step in range(start+1,stop+1):
            previous=snapshot(step-1)
            try:
                lr=learning_rate(c,step)
                for opt in (go,do):
                    for group in opt.param_groups:group['lr']=lr
                go.zero_grad(set_to_none=True);do.zero_grad(set_to_none=True);g.train()
                b=batch(data,manifest,order[step-1],c,args.device)
                if args.model=='cells':
                    ya=b['input_A3']/c['unit'];yb=b['input_B3']/c['unit']
                    _,xa=g(ya,'A');_,xb=g(yb,'B')
                    base,_=cells_objective(c,xa,xb,b['observed28'],matrix,b['sigma_DN'],b['shot_alpha'],b['input_B3'])
                else:
                    _,xa=g(integrate(b['observed83'],c));xb=None
                    base=likelihood(xa,b['observed83'])
                if not torch.isfinite(base):raise RuntimeError('Nonfinite observation objective')
                dl=base.detach()*0;adv=base*0
                if step>c['recipe']['warmup']:
                    cy,cx=[int(v) for v in rng.integers(0,65,2)]
                    fake=(torch.cat([xa[:,:,cy:cy+64,cx:cx+64],xb[:,:,cy:cy+64,cx:cx+64]],0)
                          if xb is not None else xa[:,:,cy:cy+64,cx:cx+64]).reshape(-1,1,64,64)
                    prior=b['unlabelled_prior_DN'][:,:,cy:cy+64,cx:cx+64]
                    real=(torch.cat([prior,prior],0) if xb is not None else prior).reshape(-1,1,64,64)
                    d.train();d.requires_grad_(True);dr=d(real);df=d(fake.detach())
                    dl=.5*(F.mse_loss(dr,torch.ones_like(dr))+F.mse_loss(df,torch.zeros_like(df)))
                    if not torch.isfinite(dl):raise RuntimeError('Nonfinite image-prior objective')
                    dl.backward()
                    if args.model=='ecoli':torch.nn.utils.clip_grad_norm_(d.parameters(),1e10,error_if_nonfinite=True)
                    do.step();d.eval();d.requires_grad_(False)
                    adv=F.mse_loss(d(fake),torch.ones_like(dr))
                ratio=assign_gradients(base,adv,g.parameters(),c['recipe']['adversarial_gradient_cap'])
                norm=torch.nn.utils.clip_grad_norm_(g.parameters(),c['recipe']['generator_gradient_clip'],error_if_nonfinite=True)
                if not math.isfinite(float(norm)) or norm<=0:raise RuntimeError('Invalid generator gradient')
                go.step();d.requires_grad_(True);d.train()
                if any(not torch.isfinite(v).all() for m in (g,d) for v in m.state_dict().values()):raise RuntimeError('Nonfinite updated parameters')
            except BaseException:
                restore(previous);checkpoint(step-1);raise
            del previous
            record={'step':step,'learning_rate':lr,'observation_loss':float(base.detach()),
                    'prior_discriminator_loss':float(dl.detach()),'adversarial_gradient_ratio':ratio}
            log.write(json.dumps(record)+'\n')
            if step==start+1 or step%100==0:log.flush();print(json.dumps(record),flush=True)
            if (every and step%every==0) or step==stop:
                if every and step%every==0:
                    write_scores(args.output/f'validation_{step:06d}.csv',evaluate(g,c,data,'val',args.device))
                path=checkpoint(step);print('Saved:',path,flush=True)
        if stop==c['steps']:
            write_scores(args.output/'test.csv',evaluate(g,c,data,'test',args.device))

if __name__=='__main__':main()
