"""Train a configured reconstruction model."""
import argparse
import copy
import hashlib
import json
import random
from pathlib import Path
import torch
from torch.nn import functional as F
from common import CONFIGURATIONS, Observations, build, config, state_digest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration', choices=CONFIGURATIONS, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int)
    args = parser.parse_args(); c = config(args.configuration)
    steps = c['steps'] if args.steps is None else args.steps
    if steps <= 0 or steps > c['steps']: parser.error('Steps must be between 1 and the configured budget')
    if args.output.exists(): raise FileExistsError(args.output)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    g, d = build(c, args.device, args.data/'checkpoints/initial.pt')
    observations = Observations(args.data, c, with_priors=c['use_prior'])
    go = torch.optim.Adam(g.parameters(), lr=c['learning_rate_initial'], betas=tuple(c['Adam_betas']))
    do = torch.optim.Adam(d.parameters(), lr=c['learning_rate_initial'], betas=tuple(c['Adam_betas']))
    sg = torch.optim.lr_scheduler.CosineAnnealingLR(go, T_max=c['lr_schedule_steps'], eta_min=c['learning_rate_min'])
    sd = torch.optim.lr_scheduler.CosineAnnealingLR(do, T_max=c['lr_schedule_steps'], eta_min=c['learning_rate_min'])
    bce = torch.nn.BCEWithLogitsLoss(); args.output.mkdir(parents=True)
    (args.output/'configuration.json').write_text(json.dumps(c, indent=2)+'\n')
    sequence = hashlib.sha256()

    def update(index, epoch, batch):
        target = observations.batch(epoch, batch).to(args.device)
        inputs = [target[:, c['A']]] + ([target[:, c['B']]] if c['branches'] == 2 else [])
        g.train(); d.train(); d.requires_grad_(True)
        go.zero_grad(set_to_none=True); do.zero_grad(set_to_none=True)
        if c['use_prior']:
            with torch.no_grad(): shape = d[0](torch.randn(1, 1, 128, 128, device=args.device)).shape
            real = torch.ones(4, *shape[1:], device=args.device); fake = torch.zeros_like(real)
            with torch.no_grad(): fakes = [g(v)[1] for v in inputs]
            loss_d = target.sum()*0.
            for branch, abundance in enumerate(fakes):
                for k in range(5):
                    positive = observations.prior_batch(index, branch, k).to(args.device)
                    loss_d = loss_d + (bce(d[0](positive), real) + bce(d[0](abundance[:, k:k+1].detach()), fake)) / len(inputs)
            if not torch.isfinite(loss_d): raise RuntimeError('Nonfinite discriminator loss')
            loss_d.backward(); do.step()
        else: loss_d = target.sum()*0.
        d.requires_grad_(False); outputs = [g(v) for v in inputs]
        rec = sum(F.l1_loss(r, target) for r, x in outputs) / len(outputs)
        cross = target.sum()*0.; consistency = target.sum()*0.
        if len(outputs) == 2:
            cross = .5 * (F.l1_loss(outputs[0][0][:, c['B']], inputs[1]) + F.l1_loss(outputs[1][0][:, c['A']], inputs[0]))
            consistency = F.l1_loss(outputs[0][1], outputs[1][1])
        adversarial = target.sum()*0.
        if c['use_prior']:
            for r, x in outputs:
                for k in range(5): adversarial = adversarial + bce(d[0](x[:, k:k+1]), real) / len(outputs)
        loss = c['lambda_reconstruction']*rec + (c['lambda_denoising']*cross + c['lambda_consistency']*consistency) + c['lambda_prior']*adversarial
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite generator loss')
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in g.parameters()):
            raise RuntimeError('Nonfinite gradient')
        go.step(); d.requires_grad_(True)
        if not all(torch.isfinite(v).all() for m in (g, d) for v in m.state_dict().values()):
            raise RuntimeError('Nonfinite parameters')
        return dict(loss=float(loss.detach()), reconstruction=float(rec.detach()),
                    cross_denoising=float(cross.detach()), consistency=float(consistency.detach()),
                    prior=float(adversarial.detach()), discriminator=float(loss_d.detach()))

    initial = copy.deepcopy([g.state_dict(), d.state_dict(), go.state_dict(), do.state_dict()])
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else None
    python_rng = random.getstate()
    for batch in range(9): update(batch+1, 0, batch)
    for obj, state in zip((g, d, go, do), initial): obj.load_state_dict(state)
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None: torch.cuda.set_rng_state_all(cuda_rng)
    random.setstate(python_rng); d.requires_grad_(True)
    assert state_digest(g) == c['initial_G_SHA256'] and state_digest(d[0]) == c['initial_D_SHA256']
    del initial
    with (args.output/'history.jsonl').open('w') as log:
        for step in range(1, steps+1):
            epoch, batch = divmod(step-1, c['steps_per_epoch'])
            record = update(step, epoch, batch)
            sequence.update(hashlib.sha256(observations.batch(epoch, batch).numpy().tobytes()).hexdigest().encode())
            sg.step()
            if c['use_prior']: sd.step()
            if step == 1 or step % 100 == 0:
                row = dict(step=step, epoch=epoch+1, learning_rate=go.param_groups[0]['lr'], **record)
                log.write(json.dumps(row)+'\n'); log.flush(); print(json.dumps(row), flush=True)
            if step % (c['steps_per_epoch']*10) == 0 or step == steps:
                state = dict(G=g.state_dict(), D=d.state_dict(), G_Adam=go.state_dict(), D_Adam=do.state_dict(),
                             G_schedule=sg.state_dict(), D_schedule=sd.state_dict(), epoch=epoch+1, step=step,
                             input_sequence_sha256=sequence.hexdigest(), configuration=c['name'])
                torch.save(state, args.output/f'checkpoint_{epoch+1:03d}.pt')

if __name__ == '__main__': main()
