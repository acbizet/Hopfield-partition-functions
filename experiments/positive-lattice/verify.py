"""Independent small-spin checks of positivity, derivatives and tail bounds."""
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from positive import positive,bounded_cutoffs

def reference(W,b,beta,source):
    n=len(b);s=2*((np.arange(2**n)[:,None]>>np.arange(n))&1)-1
    h=s@W;lw=beta*(s@b+(h*h).sum(axis=1)/(2*n));lz=logsumexp(lw);p=np.exp(lw-lz)
    q=s@source;mu=np.dot(p,q)
    return dict(logZ=float(lz),mean_overlap=float(mu/n),chi=float(beta*np.dot(p,(q-mu)**2)/n))

def main():
    cases=[dict(W=[[1,2],[-2,0],[0,1],[1,-1],[-1,2]],b=[.12,-.3,.2,.09,-.11],beta=.7),
           dict(W=[[1,2],[-2,0],[0,1],[1,-1],[-1,2]],b=[.12,-.3,.2,.09,-.11],beta=12.)]
    out=[]
    for case in cases:
        W=np.array(case['W']);b=np.array(case['b']);beta=case['beta'];a=np.array([.4,-.2,.7,-.6,.1])
        ref=reference(W,b,beta,a);choice=bounded_cutoffs(W,beta,1e-10);calc=positive(W,b,beta,choice['K'],source=a)
        # Exact truncation errors are one-sided; allow double-rounding scale.
        loss=ref['logZ']-calc['logZ']
        assert -2e-12<=loss<=choice['logZ_tail_bound']+2e-12
        assert abs(ref['mean_overlap']-calc['mean_overlap'])<1e-9
        assert abs(ref['chi']-calc['chi'])<1e-9
        out.append(dict(model=case,reference=ref,positive=calc,bounds=choice,logZ_loss=loss))
    Path(__file__).with_name('verification.json').write_text(json.dumps(out,indent=2)+'\n')
    print('Both independent checks passed.')

if __name__=='__main__':main()
