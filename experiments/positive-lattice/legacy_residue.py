"""Evaluate the explicit resummed residue formula vs the original integral.

Integer W only. Parameter substitution, exponentials/cosh and summation are
timed; derivation and choice of truncation/grid settings are not. Both paths
compute logZ only and keep model parameters inside the timed numerical work.
"""
import json,itertools,math,time,platform
from pathlib import Path
import numpy as np
from functools import lru_cache
from scipy.special import roots_hermitenorm,logsumexp

PI_TEXT='3.1415926535897932384626433832795028841971693993751'
TOL=1e-6


@lru_cache(maxsize=64)
def cached_rule(q):
    return roots_hermitenorm(q)


def formula(W,b,beta,K,wide=False):
    dtype=np.longdouble if wide else np.float64
    pi=dtype(PI_TEXT); W=np.asarray(W,dtype=int); b=np.asarray(b,dtype=dtype)
    n,r=W.shape; beta=dtype(beta); kd=np.arange(-K,K+1,dtype=int)
    parity=W.sum(axis=0)%2; phase_offset=int(((parity-W.sum(axis=0))//2).sum())
    odd=W.sum(axis=1)%2; inner=len(kd)
    outer=itertools.product(*([kd]*(r-1)))
    batch=max(1,32768//inner); total=dtype(0)
    while True:
        chunk=list(itertools.islice(outer,batch))
        if not chunk: break
        if r==1: ix=kd[:,None]
        else:
            other=np.array(chunk,dtype=int)
            ix=np.concatenate([np.tile(kd,len(chunk))[:,None],np.repeat(other,inner,axis=0)],axis=1)
        x=(parity+2*ix).astype(dtype)/n
        u=beta*(b+x@W.T.astype(dtype))
        eu=np.exp(u)
        matter=np.where(odd,eu-1/eu,eu+1/eu)
        sign=np.where((ix.sum(axis=1)+phase_offset)%2,-1,1)
        values=sign*np.exp(-n*beta*(x*x).sum(axis=1)/2+r*n*pi*pi/(8*beta))*np.prod(matter,axis=1)
        total+=values.reshape(-1,inner).sum(axis=1,dtype=dtype).sum(dtype=dtype)
    # Known analytic resummation for this specialization. NOT a fitted correction.
    S=dtype(0)
    for ell in range(1,1000):
        term=np.exp(-n*pi*pi*ell*(ell+1)/(2*beta)); S+=term
        if term<dtype('1e-24'): break
    if total<=0 or not np.isfinite(total): return None
    return float(np.log(total)+r*(np.log(np.sqrt(n*beta/(2*pi))/n)-np.log1p(S)))


def integrate(W,b,beta,q):
    W=np.asarray(W,dtype=float); b=np.asarray(b,dtype=float); n,r=W.shape
    nodes,weights=cached_rule(q)
    indices=np.stack([a.ravel() for a in np.meshgrid(*([np.arange(q)]*r),indexing='ij')],axis=1)
    logs=[]
    for start in range(0,len(indices),32768):
        ix=indices[start:start+32768]
        m=nodes[ix]/np.sqrt(n*beta)
        u=beta*(b+m@W.T)
        lp=np.log(weights[ix]).sum(axis=1)-r/2*np.log(2*np.pi)+np.logaddexp(u,-u).sum(axis=1)
        logs.append(logsumexp(lp))
    return float(logsumexp(logs))


def exact(W,b,beta):
    W=np.asarray(W); b=np.asarray(b); n,r=W.shape
    x=((np.arange(2**n,dtype=np.uint64)[:,None]>>np.arange(n,dtype=np.uint64))&1).astype(float)*2-1
    field=x@W
    return float(logsumexp(beta*(x@b+np.sum(field*field,axis=1)/(2*n))))


def passing(a,z,ref):
    return a is not None and z is not None and abs(a-ref)<=TOL and abs(z-ref)<=TOL and abs(a-z)<=TOL


def time_call(fn):
    t=time.perf_counter(); result=fn(); warm=time.perf_counter()-t
    batch=min(100,max(1,math.ceil(.025/max(warm,1e-9))))
    times=[]
    for _ in range(5):
        t=time.perf_counter()
        for _ in range(batch): result=fn()
        times.append((time.perf_counter()-t)/batch)
    return dict(logZ=result,median_seconds=float(np.median(times)),
                q25_seconds=float(np.quantile(times,.25)),q75_seconds=float(np.quantile(times,.75)),
                samples_seconds=times,calls_per_batch=batch)


def main():
    # Load the EXACT same integer model parameters as the preceding residue test.
    cases=json.loads(Path(__file__).with_name('models.json').read_text())
    roots_hermitenorm(4)  # Shared library setup outside both methods' timings.
    results=[]
    for c in cases:
        W=np.array(c['W']); b=np.array(c['b']); beta=c['beta']; n,r=W.shape; ref=exact(W,b,beta)
        scan=[]; settings=[]; scan_times={}
        for wide in (False,True):
            start=time.perf_counter(); prev=None
            for K in range(2,max(c['cutoffs'])+2):
                value=formula(W,b,beta,K,wide)
                scan.append(dict(method='formula',K=K,wide=wide,logZ=value))
                if prev is not None and passing(prev[1],value,ref):
                    settings.append(dict(K=prev[0],wide=wide,logZ=prev[1],validation_logZ=value,
                                         validation_K=K,terms=(2*prev[0]+1)**r))
                    break
                prev=(K,value)
            scan_times['formula_wide' if wide else 'formula_float64']=time.perf_counter()-start
        start=time.perf_counter(); prev=None; quad_setting=None
        for q in (4,6,8,10,12,14,16,20,24,28,32):
            if q**r>35_000_000: break
            value=integrate(W,b,beta,q)
            scan.append(dict(method='integral',q=q,logZ=value))
            if prev is not None and passing(prev[1],value,ref):
                quad_setting=dict(q=prev[0],logZ=prev[1],validation_q=q,validation_logZ=value,nodes=prev[0]**r)
                break
            prev=(q,value)
        scan_times['integral']=time.perf_counter()-start
        if not settings or quad_setting is None: raise RuntimeError('Target not reached; inspect candidates')
        candidates=[]
        for setting in settings:
            timed=time_call(lambda:formula(W,b,beta,setting['K'],setting['wide']))
            assert timed['logZ'] is not None and abs(timed['logZ']-ref)<=TOL
            candidates.append(dict(**setting,timing=timed))
        best=min(candidates,key=lambda z:z['timing']['median_seconds'])
        quad_timing=time_call(lambda:integrate(W,b,beta,quad_setting['q']))
        assert abs(quad_timing['logZ']-ref)<=TOL
        row=dict(N=n,r=r,beta=beta,W=W.tolist(),b=b.tolist(),reference_logZ=ref,target=TOL,
                 formula=best,formula_precision_candidates=candidates,
                 integral=dict(**quad_setting,timing=quad_timing),selection_seconds=scan_times,scan=scan,
                 formula_to_integral_time_ratio=best['timing']['median_seconds']/quad_timing['median_seconds'])
        results.append(row)
        Path(__file__).with_name('results.json').write_text(json.dumps(dict(cases=results,
                formula='Z = [sqrt(N beta/2pi)/(N Psi)]^r sum_k I0(z_1k1,...,z_rkr)',
                psi='1+sum_l>=1 exp[-N pi^2 l(l+1)/(2 beta)]',
                precision=dict(python=platform.python_version(),numpy=np.__version__,scipy=__import__('scipy').__version__)),indent=2)+'\n')
        print(n,r,beta,'K',best['K'],'wide',best['wide'],'q',quad_setting['q'],
              'times',best['timing']['median_seconds'],quad_timing['median_seconds'],
              'ratio',row['formula_to_integral_time_ratio'],flush=True)


if __name__=='__main__': main()
