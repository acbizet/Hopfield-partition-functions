"""Small independent enumeration comparison; run from any working directory."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/positive-lattice'))
from positive import positive, bounded_cutoffs
from verify import reference

W = np.array([[1,2],[-2,0],[0,1],[1,-1],[-1,2]])
b = np.array([.12,-.3,.2,.09,-.11])
a = np.array([.4,-.2,.7,-.6,.1])
beta = .7
bounds = bounded_cutoffs(W,beta,1e-10)
actual = positive(W,b,beta,bounds['K'],source=a)
exact = reference(W,b,beta,a)
errors = {key:abs(actual[key]-exact[key]) for key in ('logZ','mean_overlap','chi')}
loss = exact['logZ']-actual['logZ']
if not (-2e-12 <= loss <= bounds['logZ_tail_bound']+2e-12):
    raise RuntimeError('Truncation comparison failed')
if max(errors.values()) >= 1e-9:
    raise RuntimeError('Observable comparison failed')
print(json.dumps(dict(status='passed',spin_states=2**len(b),positive=actual,
                     enumeration=exact,absolute_errors=errors,bounds=bounds),indent=2))
