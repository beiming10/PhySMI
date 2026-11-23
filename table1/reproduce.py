"""Reproduce the complete checkpoint comparison with one command."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('table1_data'))
    parser.add_argument('--output',type=Path,default=Path('table1_results'))
    parser.add_argument('--device',default='auto')
    parser.add_argument('--offline',action='store_true')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    device=('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if not args.offline:
        subprocess.run([sys.executable,str(ROOT/'download.py'),'--evaluation-only','--output',str(args.data)],check=True)
    subprocess.run([sys.executable,str(ROOT/'evaluate.py'),'--data',str(args.data),'--output',str(args.output),'--device',device],check=True)
    rows=json.loads((args.output/'summary.json').read_text())
    expected={r['configuration']:r for r in csv.DictReader((ROOT/'expected_results.csv').open())}
    print('\nConfiguration                           PSNR (dB)     SSIM    SAM (deg)')
    for row in rows:
        name=row['configuration']
        print(f'{name:40s} {row["PSNR"]:8.2f}    {row["SSIM"]:.3f}      {row["SAM"]:.2f}')
        limits={'PSNR':.02,'SSIM':.001,'SAM':.02}
        for key,tolerance in limits.items():
            if abs(row[key]-float(expected[name][key]))>tolerance:
                raise RuntimeError(f'{name} {key} differs from the recorded result; inspect the numerical environment')
    print('\nSaved:',args.output/'Table1.csv')
    print('Per-pair scores:',args.output/'per_pair.csv')

if __name__=='__main__':main()
