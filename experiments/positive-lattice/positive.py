"""Exact positive integer-weight Hopfield lattice representation.

Z = G^(-r) sum_k I((p+2k)/N), G=sum_j exp(-2*beta*j*j/N).
Finite cutoffs approximate the infinite identity. This is an alternative
Gaussian-summation representation, not direct complex-residue evaluation.
"""
import itertools,json,math,time
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.special import roots_hermitenorm,logsumexp


def log_normalizer(n,beta):
    a=2*beta/n
    # Jacobi transformation: choose the faster of two positive series.
    exponent=a if a>=np.pi else np.pi*np.pi/a
    vals=[]
    for j in range(1,10000):
        t=np.exp(-exponent*j*j);vals.append(t)
        # Successive tail terms decrease at least geometrically.
        tail=2*np.exp(-exponent*(j+1)**2)/(1-np.exp(-exponent*(2*j+3)))
        if tail<1e-18:break
    return np.log1p(2*sum(vals))+(0 if a>=np.pi else .5*np.log(np.pi/a))


def coordinate_tail_bound(n,beta,absolute_column_sum,parity,K):
    """Uniform omitted mixture mass for a symmetric coordinate cutoff.

    This evaluates an analytic bound in ordinary floating point; it is not
    an outward-rounded interval certificate. No spin enumeration is needed.
    """
    a=2*beta/n;center=(int(absolute_column_sum)-int(parity))//2
    first_right=K+1-int(parity)-center;first_left=K+1+center
    if min(first_left,first_right)<=0:return 1.
    def tail(m):return np.exp(-a*m*m)/(-np.expm1(-a*(2*m+1)))
    return min(1.,float((tail(first_right)+tail(first_left))*np.exp(-log_normalizer(n,beta))))


def bounded_cutoffs(W,beta,tolerance):
    W=np.asarray(W,dtype=int);n,r=W.shape;p=W.sum(axis=0)%2;A=np.abs(W).sum(axis=0)
    target=-np.expm1(-tolerance);cutoffs=[];bounds=[]
    for absolute,parity in zip(A,p):
        K=1
        while coordinate_tail_bound(n,beta,absolute,parity,K)>target/r:K+=1
        cutoffs.append(K);bounds.append(coordinate_tail_bound(n,beta,absolute,parity,K))
    return dict(K=cutoffs,relative_Z_tail_bound=sum(bounds),logZ_tail_bound=float(-np.log1p(-sum(bounds))),
                per_coordinate_bounds=bounds,scope='analytic truncation bound evaluated in float64; rounding excluded')


def batches(axes,batch=32768):
    shape=tuple(len(x) for x in axes);total=math.prod(shape)
    for start in range(0,total,batch):
        ix=np.stack(np.unravel_index(np.arange(start,min(start+batch,total)),shape),axis=1)
        yield np.stack([ax[ix[:,a]] for a,ax in enumerate(axes)],axis=1),ix


def positive(W,b,beta,K,source=None):
    W=np.asarray(W);b=np.asarray(b,dtype=float)
    if not np.isfinite(W).all() or not np.equal(W,np.rint(W)).all():raise ValueError('integer W required')
    if beta<=0 or not np.isfinite(beta):raise ValueError('finite beta > 0 required')
    n,r=W.shape;parity=W.astype(np.int64).sum(axis=0)%2
    if np.isscalar(K):K=[int(K)]*r
    # Symmetric real intervals; odd-parity axes have a half-step offset.
    axes=[(p+2*np.arange(-k,k+1-p))/n for k,p in zip(K,parity)]
    if any(len(ax)==0 for ax in axes):raise ValueError('cutoffs must retain points')
    logs=[];batch_means=[]
    for x,ix in batches(axes):
        u=beta*(b+x@W.T)
        lw=-n*beta*(x*x).sum(axis=1)/2+np.logaddexp(u,-u).sum(axis=1)
        lz=float(logsumexp(lw));logs.append(lz)
        if source is not None:
            a=np.asarray(source,dtype=float);th=np.tanh(u)
            t=th@a
            # sech^2(u) via exponentials avoids 1-tanh^2 cancellation.
            e=np.exp(-2*np.abs(u));v=(4*e/(1+e)**2)@(a*a)
            probs=np.exp(lw-lz);mu=np.dot(probs,t)
            batch_means.append((mu,float(np.dot(probs,(t-mu)**2+v))))
    total=float(logsumexp(logs))
    result=dict(logZ=total-r*log_normalizer(n,beta),terms=math.prod(map(len,axes)),K=list(K),precision='float64')
    if source is not None:
        weights=np.exp(np.array(logs)-total);means=np.array([x[0] for x in batch_means]);mean=np.dot(weights,means)
        # Law of total variance, including conditional spin fluctuations.
        variance=np.dot(weights,np.array([x[1] for x in batch_means])+(means-mean)**2)
        result.update(mean_overlap=float(mean/n),chi=float(beta*variance/n))
    return result


@lru_cache(maxsize=32)
def cached_rule(q):return roots_hermitenorm(q)


def gauss_hermite(W,b,beta,q):
    """Original integral, sharing batching and log-integrand code structure."""
    W=np.asarray(W);b=np.asarray(b);n,r=W.shape
    nodes,weights=cached_rule(q);axes=[nodes/np.sqrt(n*beta)]*r;logw=np.log(weights);logs=[]
    for x,ix in batches(axes):
        u=beta*(b+x@W.T)
        lw=logw[ix].sum(axis=1)-r/2*np.log(2*np.pi)+np.logaddexp(u,-u).sum(axis=1)
        logs.append(float(logsumexp(lw)))
    return dict(logZ=float(logsumexp(logs)),terms=q**r,q=q,precision='float64')


def timed(fn):
    start=time.perf_counter();r=fn();warm=time.perf_counter()-start
    batch=min(30,max(1,math.ceil(.025/max(warm,1e-8))));samples=[]
    for _ in range(5):
        start=time.perf_counter()
        for __ in range(batch):r=fn()
        samples.append((time.perf_counter()-start)/batch)
    return dict(value=r,median_seconds=float(np.median(samples)),samples_seconds=samples,calls_per_batch=batch)


def main():
    # The bundled model and reference are unchanged from the preceding rank-5 benchmark.
    root=Path(__file__).parent
    prior=json.loads((root/'rank5_input.json').read_text())
    W=np.array(prior['W']);b=np.array(prior['b']);beta=prior['beta'];ref=prior['reference_logZ']
    result=dict(model=prior,scan=[],comparisons=[],critical=[])
    def save():(root/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    for K in range(2,13):
        z=positive(W,b,beta,K);z['error']=abs(z['logZ']-ref);result['scan'].append(z)
        print('positive scan',K,z,flush=True);save()
        if K>2 and all(x['error']<1e-12 for x in result['scan'][-2:]):break
    import legacy_residue
    for tol in (1e-3,1e-6,1e-9):
        selected=next(a for a,z in zip(result['scan'],result['scan'][1:]) if a['error']<=tol and z['error']<=tol and abs(z['logZ']-a['logZ'])<=tol)
        K=selected['K'][0]
        pos=timed(lambda:positive(W,b,beta,K));pos['absolute_logZ_error']=abs(pos['value']['logZ']-ref)
        old=next(c for c in prior['prior_settings'] if c['tolerance']==tol)
        q=old['q'];cached_rule(q);fine=gauss_hermite(W,b,beta,old['validation_q'])
        gh=timed(lambda:gauss_hermite(W,b,beta,q));gh['absolute_logZ_error']=abs(gh['value']['logZ']-ref)
        assert gh['absolute_logZ_error']<=tol and abs(fine['logZ']-ref)<=tol
        res=timed(lambda:legacy_residue.formula(W,b,beta,old['K'],old['wide']))
        res['absolute_logZ_error']=abs(res['value']-ref)
        assert res['absolute_logZ_error']<=tol
        row=dict(tolerance=tol,positive=pos,gauss_hermite=gh,legacy_residue=res,
                 speedup_over_residue=res['median_seconds']/pos['median_seconds'],
                 speedup_over_gauss_hermite=gh['median_seconds']/pos['median_seconds'])
        result['comparisons'].append(row);save();print('comparison',row,flush=True)
    rows=np.array([[1,1,1],[-1,1,1],[1,-1,1],[1,1,-1]])
    W=np.repeat(rows,[6,2,2,2],axis=0);b=np.zeros(12);a=W@np.ones(3)/np.sqrt(3)
    cref=2.873524158356024125814297772083539917198
    for K in (12,16,20,24):
        z=positive(W,b,.6,K,source=a);z.update(chi_error=abs(z['chi']-cref))
        result['critical'].append(z);save();print('critical',z,flush=True)
    result['critical_timing']=timed(lambda:positive(W,b,.6,24,source=a));save()
    W=np.array(prior['W']);b=np.array(prior['b']);beta=prior['beta']
    result['bounded_cutoff_evaluations']=[]
    for tolerance in (1e-3,1e-6,1e-9):
        choice=bounded_cutoffs(W,beta,tolerance);value=positive(W,b,beta,choice['K'])
        error=abs(value['logZ']-ref)
        assert error<=choice['logZ_tail_bound']+1e-13
        result['bounded_cutoff_evaluations'].append(dict(tolerance=tolerance,choice=choice,value=value,actual_error=error));save()


if __name__=='__main__':main()
