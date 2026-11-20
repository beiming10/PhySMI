"""Self-supervised training on paired observation batches, without component targets."""
import argparse,copy,json,math
from pathlib import Path
import numpy as np
import torch
from physmi.models import ROOT,load_config,build_model,predict
from physmi.discriminator import StCDiscriminator
from physmi.losses import eight_channel_objective,cells_objective,EcoliLikelihood,capped_gradients,critic_losses

ALLOWED={'observed','input_a','input_b','prior','panel','sigma_DN','shot_alpha','gan','condition_only','prior_anchor'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['static','live','cells','ecoli'],required=True)
    p.add_argument('--data',type=Path,required=True,help='Directory of observation-only NPZ batches')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--seed',type=int,default=20261118)
    p.add_argument('--initialize',choices=['published','fresh'],default='published')
    a=p.parse_args()
    if a.steps<=0:raise ValueError('A positive finite step count is required')
    if a.output.exists():raise FileExistsError(a.output)
    files=sorted(a.data.glob('*.npz'))
    if not files:raise ValueError('No observation batches')
    torch.set_num_threads(2);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
    c=load_config(a.model);model=build_model(c,weights=a.initialize=='published',device=a.device)
    critic=StCDiscriminator(in_channels=1,ndf=64,n_layers=3).to(a.device)
    schedule=c['training'];lr0=schedule.get('LR_start',2e-4);lr1=schedule.get('LR_end',4e-5)
    groups=[{'params':[v for n,v in model.named_parameters() if v.requires_grad and n not in ['branch_film','absence_log_gain']], 'lr_multiplier':1.}]
    if a.model=='static' and c['known_panel']:
        groups.extend([{'params':[model.branch_film],'lr_multiplier':schedule['branch_LR_multiplier']},
                       {'params':[model.absence_log_gain],'lr_multiplier':schedule['absence_LR_multiplier']}])
    for group in groups:group['lr']=lr0*group['lr_multiplier']
    opt=torch.optim.Adam(groups,lr=lr0,betas=(.5,.999));dopt=torch.optim.Adam(critic.parameters(),lr=lr0,betas=(.5,.999))
    likelihood=EcoliLikelihood(c,ROOT/'calibration').to(a.device) if a.model=='ecoli' else None
    matrix=torch.tensor(c['response'],device=a.device,dtype=torch.float32)
    a.output.mkdir(parents=True);history=[]
    original_trainable={n for n,v in model.named_parameters() if v.requires_grad}
    for step in range(1,a.steps+1):
        file=files[(step-1)%len(files)]
        with np.load(file,allow_pickle=False) as z:
            if not set(z.files)<=ALLOWED:raise ValueError('Training batches accept observations and independent priors only: '+str(set(z.files)-ALLOWED))
            b={k:torch.as_tensor(z[k].copy(),device=a.device,dtype=torch.float32) for k in z.files}
        for key in ['observed','input_a','prior']:
            if key not in b:raise ValueError('Missing '+key)
        if b['observed'].ndim==3:
            b={k:(v[None] if k in ['observed','input_a','input_b','prior','panel','sigma_DN'] else v) for k,v in b.items()}
        panel=b.get('panel');cond_only=bool(b.get('condition_only',torch.tensor(False)).item())
        if cond_only and not (a.model=='static' and c['known_panel']):
            raise ValueError('Condition-only batches are not supported by this model')
        for name,v in model.named_parameters():v.requires_grad_(name in original_trainable and (not cond_only or name in ['branch_film','absence_log_gain']))
        # An exception restores the complete pre-batch transaction and stops.
        snapshot=copy.deepcopy([o.state_dict() for o in [model,critic,opt,dopt]])
        rng_snapshot=copy.deepcopy(rng.bit_generator.state);cpu_rng=torch.get_rng_state().clone()
        cuda_rng=torch.cuda.get_rng_state_all() if str(a.device).startswith('cuda') else None
        try:
            model.train();opt.zero_grad(set_to_none=True);dopt.zero_grad(set_to_none=True)
            ra,xa=predict(model,b['input_a'],c,panel,'A');xb=None
            if a.model in ['static','live','cells']:
                rb,xb=predict(model,b['input_b'],c,panel,'B')
            if a.model in ['static','live']:base=eight_channel_objective(ra,rb,xa,xb,b['observed'],b['input_a'],b['input_b'],c)
            elif a.model=='cells':
                base,_=cells_objective(c,xa,xb,b['observed'],matrix,b['sigma_DN'],b['shot_alpha'].reshape(-1),b['input_b'])
            else:base=likelihood(xa,b['observed'])
            use_gan=bool(b.get('gan',torch.tensor(True)).item()) and (a.initialize=='published' or step>schedule.get('warmup',0))
            ratio=0.;dl=base.detach()*0
            if use_gan:
                h,w=xa.shape[-2:];cy,cx=int(rng.integers(h-63)),int(rng.integers(w-63))
                fake=torch.cat([xa,xb],0) if xb is not None else xa
                prior=torch.cat([b['prior'],b['prior']],0) if xb is not None else b['prior']
                anchor=float(b.get('prior_anchor',torch.tensor(1.)).item())
                fake=(fake[:,:,cy:cy+64,cx:cx+64]/anchor).reshape(-1,1,64,64)
                real=(prior[:,:,cy:cy+64,cx:cx+64]/anchor).reshape(-1,1,64,64)
                critic.train();critic.requires_grad_(True)
                dl=critic_losses(critic,real,fake,schedule['GAN_loss']);dl.backward();dopt.step()
                critic.eval();critic.requires_grad_(False)
                adv=critic_losses(critic,real,fake,schedule['GAN_loss'],False)
                ratio=capped_gradients(base,adv,model.parameters())
            else:base.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            if not torch.isfinite(base) or not torch.isfinite(dl) or norm<=0:raise RuntimeError('Invalid training state')
            opt.step()
            if any(not torch.isfinite(v).all() for m in [model,critic] for v in m.state_dict().values()):raise RuntimeError('Nonfinite updated weights')
            lr=lr1+(lr0-lr1)*.5*(1+math.cos(math.pi*step/a.steps))
            for op in [opt,dopt]:
                for g in op.param_groups:g['lr']=lr*g.get('lr_multiplier',1.)
            history.append({'step':step,'observation_loss':float(base.detach()),'critic_loss':float(dl.detach()),'adversarial_gradient_ratio':ratio})
        except BaseException:
            for o,s in zip([model,critic,opt,dopt],snapshot):o.load_state_dict(s)
            rng.bit_generator.state=rng_snapshot;torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
            torch.save({'model':model.state_dict(),'step':step-1},a.output/'stopped_prebatch.pt');raise
        finally:critic.requires_grad_(True)
    torch.save({'model':model.state_dict(),'critic':critic.state_dict(),'optimizer':opt.state_dict(),'critic_optimizer':dopt.state_dict(),'steps':a.steps,'model_name':a.model},a.output/'checkpoint.pt')
    (a.output/'history.json').write_text(json.dumps(history,indent=2));print(a.output/'checkpoint.pt')
if __name__=='__main__':main()
