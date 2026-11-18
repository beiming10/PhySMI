"""Single-component controls with all model outputs retained."""
import argparse,json
from pathlib import Path
import numpy as np

def single_component_statistics(prediction,reference,active):
    p=np.asarray(prediction,np.float64);t=np.asarray(reference,np.float64)
    if p.shape!=t.shape or p.ndim!=3 or p.min()<0:raise ValueError('Expected matching nonnegative K,Y,X arrays')
    if any(np.any(t[j]!=0) for j in range(len(t)) if j!=active):raise ValueError('Reference is not single-component')
    sums=p.sum((1,2));signal=t[active].sum()
    if sums.sum()<=0 or signal<=0:raise ValueError('Empty signal denominator')
    x=t[active].ravel();center=x-x.mean();den=center@center
    slopes=[float(center@(y.ravel()-y.mean())/den) if den>0 else None for y in p]
    return {'off_target_fraction_percent':float(100*(sums.sum()-sums[active])/sums.sum()),
            'target_integrated_gain':float(sums[active]/signal),
            'output_signal_fraction_percent':(100*sums/sums.sum()).tolist(),
            'spatial_regression_slopes':slopes,
            'definitions':{'off_target_fraction_percent':'All non-target output intensity divided by all output intensity over the entire FOV',
                           'spatial_regression_slopes':'Centered regression of each output on the simulated component image; removes only the fitted constant when estimating a slope. Does not change the prediction.'}}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('prediction',type=Path);p.add_argument('reference',type=Path);p.add_argument('--active',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=single_component_statistics(np.load(a.prediction),np.load(a.reference),a.active)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False));print(a.output)
if __name__=='__main__':main()
