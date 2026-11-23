"""Reconstruct components and calculate PSNR, SSIM, and spectral angle."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import tifffile
import torch
from common import CONFIGURATIONS, ROOT, build, config, metrics, read_json

def nnls(matrix, observed):
    matrix = np.asarray(matrix, np.float64)
    y = np.asarray(observed, np.float64).reshape(8, -1)
    best = np.full(y.shape[1], np.inf); solution = np.zeros((5, y.shape[1]))
    for mask in range(1, 32):
        ids = [k for k in range(5) if mask >> k & 1]
        value = np.linalg.lstsq(matrix[:, ids], y, rcond=None)[0]
        valid = (value >= -1e-8).all(0)
        error = ((matrix[:, ids] @ value-y)**2).sum(0)
        pick = valid & (error < best); best[pick] = error[pick]; solution[:, pick] = 0
        for j, k in enumerate(ids): solution[k, pick] = np.maximum(value[j, pick], 0)
    return solution.reshape(5, *observed.shape[-2:])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--configuration', choices=CONFIGURATIONS)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--recompute-nnls', action='store_true')
    parser.add_argument('--save-components', action='store_true')
    args = parser.parse_args()
    if args.checkpoint and not args.configuration: parser.error('Select the checkpoint configuration')
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True); torch.set_num_threads(2)
    cases = read_json(ROOT/'dataset.json')['test']; all_rows = []; summary = []
    choices = [args.configuration] if args.configuration else list(CONFIGURATIONS)
    for name in choices + ([] if args.configuration else ['nnls']):
        c = config('full' if name == 'nnls' else name)
        if name != 'nnls':
            model, _ = build(c, args.device)
            path = args.checkpoint or args.data/'checkpoints'/(name+'.pt')
            state = torch.load(path, map_location='cpu', weights_only=True)
            model.load_state_dict(state['G'], strict=True); model.eval()
        rows = []
        for case in cases:
            ref = tifffile.imread(args.data/case['reference'])
            if name == 'nnls':
                if args.recompute_nnls:
                    obs = tifffile.imread(args.data/case['input'])
                    pred = np.clip(np.rint(nnls(c['response'], obs)), 0, 65535).astype(np.uint16)
                else: pred = tifffile.imread(args.data/case['nnls'])
            else:
                obs = tifffile.imread(args.data/case['input']).astype(np.float32)/c['scale']
                with torch.inference_mode():
                    pred = model(torch.from_numpy(np.ascontiguousarray(obs[c['A']][None])).to(args.device))[1][0].cpu().numpy()*c['scale']
            score = metrics(pred, ref)
            row = dict(configuration=name, id=case['id'], **score)
            rows.append(row); all_rows.append(row)
            if args.save_components:
                p = args.output/name;p.mkdir(exist_ok=True)
                tifffile.imwrite(p/(case['id']+'.tif'), pred.astype(np.float32), photometric='minisblack', metadata={'axes':'CYX'}, compression='deflate')
        means = {key:float(np.mean([r[key] for r in rows])) for key in ('PSNR','SSIM','SAM')}
        result = dict(configuration=name, n=len(rows), epoch=None if name=='nnls' else state.get('epoch'),
                      updates=None if name=='nnls' else state.get('step'), **means,
                      PSNR_sample_SD=float(np.std([r['PSNR'] for r in rows], ddof=1)))
        summary.append(result); print(json.dumps(result), flush=True)
    (args.output/'per_pair.json').write_text(json.dumps(all_rows, indent=2)+'\n')
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    for filename, rows, fields in [('Table1.csv', summary, list(summary[0])),
                                   ('per_pair.csv', all_rows, ['configuration','id','PSNR','SSIM','SAM'])]:
        with (args.output/filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)

if __name__ == '__main__': main()
