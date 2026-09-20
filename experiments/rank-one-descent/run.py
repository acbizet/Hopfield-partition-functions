"""Reproduce a restricted contour-descent benchmark and independent baselines.

Python 3.10+, NumPy, SciPy. Run with OPENBLAS_NUM_THREADS=1.
No user-supplied full Bethe solver is available: see report.md for scope.
"""
import json, math, time, statistics
from decimal import Decimal as D, localcontext
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp, roots_hermitenorm, expit, gammaln

PI = D('3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647')


def timed(fn, repeats=3):
    measurements=[]
    for _ in range(repeats):
        start=time.perf_counter(); result=fn(); measurements.append(time.perf_counter()-start)
    return result, statistics.median(measurements)


def exact(W,b,beta):
    n,r=W.shape
    states=((np.arange(2**n, dtype=np.uint64)[:,None] >> np.arange(n,dtype=np.uint64)) & 1).astype(float)*2-1
    s=states@W
    scores=beta*(states@b+np.sum(s*s,axis=1)/(2*n))
    lz=logsumexp(scores); p=np.exp(scores-lz)
    return dict(logZ=float(lz), gb=beta*(p@states), gW=beta/n*(states.T@(p[:,None]*s)))


def grouped(n,beta):
    k=np.arange(n+1); s=n-2*k
    return float(logsumexp(gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1)+beta*s*s/(2*n)))


def quadrature1(W,b,beta):
    n=len(b); w=W[:,0]
    # All real stationary points lie in [-mean|W|, mean|W|].
    center=np.mean(abs(w))
    def logf(x): return -n*beta*x*x/2+np.logaddexp(beta*(b+w*x),-beta*(b+w*x)).sum()
    grid=np.linspace(-center-1,center+1,2001)
    u=beta*(b[None,:]+grid[:,None]*w[None,:])
    peak=float(np.max(-n*beta*grid*grid/2+np.logaddexp(u,-u).sum(axis=1)))
    # Split around the possible mode region so infinite-interval quadrature
    # does not miss displaced modes. Error estimates are quadrature estimates.
    cuts=[-np.inf,-center-1,0.,center+1,np.inf]
    vals=[quad(lambda x: np.exp(logf(x)-peak),lo,hi,epsabs=1e-11,epsrel=1e-11,limit=300) for lo,hi in zip(cuts,cuts[1:])]
    z=sum(x[0] for x in vals)
    return dict(logZ=float(.5*np.log(n*beta/(2*np.pi))+peak+np.log(z)), estimated_relative_quad_error=sum(x[1] for x in vals)/z)


def hermite(W,b,beta,order):
    n,r=W.shape
    q,w=roots_hermitenorm(order)
    meshes=np.meshgrid(*([np.arange(order)]*r),indexing='ij')
    ix=np.stack([m.ravel() for m in meshes],axis=1)
    points=q[ix]; logs=np.log(w[ix]).sum(axis=1)-r/2*np.log(2*np.pi)
    accum=[]; grad_b=[]; grad_w=[]
    # Bounded batches avoid a nodes-by-N memory blowup.
    for start in range(0,len(ix),20000):
        v=points[start:start+20000]; m=v/np.sqrt(n*beta)
        u=beta*(b+m@W.T); t=np.tanh(u)
        lp=logs[start:start+20000]+np.logaddexp(u,-u).sum(axis=1)
        top=float(lp.max()); p=np.exp(lp-top); z=p.sum()
        accum.append(top+np.log(z))
        grad_b.append(beta*(p@t)/z)
        grad_w.append(beta*(t.T@(p[:,None]*m))/z)
    lz=float(logsumexp(accum)); weights=np.exp(np.array(accum)-lz)
    return dict(logZ=lz, gb=np.einsum('k,ki->i',weights,grad_b),gW=np.einsum('k,kij->ij',weights,grad_w),nodes=order**r)


def ais(W,b,beta,seed,particles=512,steps=256):
    # Uniform base, linear inverse-temperature schedule, one full Gibbs sweep
    # per bridge. Weight update precedes the target-invariant transition.
    rng=np.random.default_rng(seed); n,r=W.shape
    x=rng.choice([-1.,1.],size=(particles,n)); s=x@W; lw=np.zeros(particles)
    for t in range(1,steps+1):
        lw+=beta/steps*(x@b+np.sum(s*s,axis=1)/(2*n))
        for i in range(n):
            excluded=s-x[:,i,None]*W[i]
            field=b[i]+excluded@W[i]/n
            new=np.where(rng.random(particles)<expit(2*beta*t/steps*field),1.,-1.)
            s+=(new-x[:,i])[:,None]*W[i]; x[:,i]=new
    lnorm=logsumexp(lw); p=np.exp(lw-lnorm)
    return dict(logZ=float(n*np.log(2)+lnorm-np.log(particles)),
                gb=beta*(p@x),gW=beta/n*(x.T@(p[:,None]*s)),ess=float(1/(p@p)))


def residue_two_levels(W,b,beta,dps=75,extra_cut=0):
    """B0+B1=B0/2 for the reconstructed LOWER-STRIP contour convention.

    Uses only residues and model parameters, never a reference Z. Integer
    rank-one weights, arbitrary real biases. B1 includes inherited poles.
    Arbitrary precision real arithmetic avoids destructive complex cancellation.
    No gradients or multidimensional descent claimed.
    """
    n=len(b)
    if any(int(v)!=v for v in W): raise ValueError('Integer rank-one W required')
    with localcontext() as ctx:
        ctx.prec=dps
        beta=D(str(beta)); bb=[D(str(v)) for v in b]; ww=[int(v) for v in W]
        nn=D(n); h=2/nn; parity=sum(ww)%2
        A=sum(abs(v) for v in ww)
        # Conservative fixed cutoff from an absolute envelope. Enlarging it
        # and increasing precision are checked separately; not a certified tail.
        constant=nn*PI*PI/(8*beta)+beta*sum(abs(v) for v in bb)+nn*D(2).ln()+beta*D(A*A)/(2*nn)
        L=D(A)/nn+(2*(D(45)*D(10).ln()+constant)/(nn*beta)).sqrt()+D(extra_cut)
        K=int(L/h)+3; values=[]
        for k in range(-K,K+1):
            x=(D(parity)+2*D(k))/nn
            value=(-nn*beta*x*x/2+nn*PI*PI/(8*beta)).exp()
            for wi,bi in zip(ww,bb):
                u=beta*(bi+D(wi)*x); e=u.exp()
                value*=e-1/e if wi%2 else e+1/e
            # Full integer sum fixes the phase when sum(W) differs from parity.
            phase=k+(parity-sum(ww))//2
            values.append(value if phase%2==0 else -value)
        total=sum(values,D(0)); value=(nn*beta/(2*PI)).sqrt()*total/nn
        condition=sum(abs(v) for v in values)/abs(total)
        a=(-nn*PI*PI/(2*beta)).exp()
        S=D(0)
        for l in range(1,10000):
            term=a**(l*(l+1)); S+=term
            if term<D('1e-65'): break
        return dict(logZ=float(value.ln()),logZ_decimal=str(value.ln()),
                    predicted_relative_Z_error=float(S),predicted_logZ_error=math.log1p(float(S)),
                    predicted_relative_Z_error_decimal=str(S),
                    descent_ratio=float(S/(1+2*S)),residue_terms=len(values),
                    cancellation_digits=float(condition.log10()),precision=dps,
                    B0_logZ=float(value.ln()+D(2).ln()))


def gradient_error(a,b):
    return max(float(np.max(abs(a['gb']-b['gb']))),float(np.max(abs(a['gW']-b['gW']))))


def residue_float64(W,b,beta):
    n=len(b); w=np.array(W,dtype=int); A=sum(abs(w)); parity=int(sum(w))%2
    constant=n*np.pi**2/(8*beta)+beta*sum(abs(b))+n*np.log(2)+beta*A*A/(2*n)
    L=A/n+np.sqrt(2*(45*np.log(10)+constant)/(n*beta)); K=int(L/(2/n))+3
    k=np.arange(-K,K+1); x=(parity+2*k)/n
    u=beta*(b[None,:]+x[:,None]*w[None,:])
    factors=np.where(w[None,:]%2,2*np.sinh(u),2*np.cosh(u))
    v=np.exp(-n*beta*x*x/2+n*np.pi**2/(8*beta))*np.prod(factors,axis=1)
    v*=np.where((k+(parity-int(sum(w)))//2)%2, -1.,1.)
    z=np.sqrt(n*beta/(2*np.pi))*sum(v)/n
    return dict(logZ=float(np.log(z)) if z>0 and np.isfinite(z) else None,
                status='positive_result' if z>0 and np.isfinite(z) else 'nonpositive_or_nonfinite_result')


def check_descendants(beta):
    n=4; a=np.exp(-n*np.pi**2/(2*beta)); t=a*a
    def f(x,j):
        I=np.exp(-n*beta*x*x/2+np.logaddexp(beta*x,-beta*x)*n)
        r=a*np.exp(-1j*np.pi*n*x)
        f1=-(1-t)*r/((1-r)*(r-t))
        f2=-(1-t)**2*r*r*(r*r-t*(1+t)*r+t)/((1-r)*(r-t)*(t*r*r-2*r+t)*(r*r-2*r+t*t))
        return float((I*(1 if j==0 else f1 if j==1 else f2)).real)
    # A broad, explicitly split interval; Gaussian tails beyond it negligible
    # for these N=4, beta>=4 validation cases.
    cuts=np.linspace(-5,5,201)
    vals=[sum(quad(lambda x:f(x,j),lo,hi,epsabs=1e-5,epsrel=1e-11)[0] for lo,hi in zip(cuts,cuts[1:])) for j in range(3)]
    S=sum(a**(l*(l+1)) for l in range(1,30))
    return dict(beta=beta,E0_over_Z=vals[1]/vals[0],E1_over_Z=vals[2]/vals[0],
                predicted_E0_over_Z=-(1+2*S),predicted_E1_over_Z=-S,
                ratio=abs(vals[2]/vals[1]))


def run():
    rows=[]
    for n,beta in [(4,.8),(4,1.),(4,1.2),(4,4.),(4,8.),(4,16.),(16,1.),(64,1.)]:
        W=np.ones((n,1)); b=np.zeros(n)
        ref,te=timed(lambda:grouped(n,beta))
        q,tq=timed(lambda:quadrature1(W,b,beta))
        bet,tb=timed(lambda:residue_two_levels(W[:,0],b,beta))
        stable=residue_two_levels(W[:,0],b,beta,dps=100,extra_cut=2)
        gh,tgh=timed(lambda:hermite(W,b,beta,64))
        gh2=hermite(W,b,beta,128)
        rows.append(dict(model='uniform_rank1',N=n,r=1,beta=beta,reference_logZ=ref,
                         quadrature_logZ_error=abs(q['logZ']-ref),quadrature_seconds=tq,
                         reference_seconds=te,bethe_seconds=tb,bethe_abs_logZ_error=abs(bet['logZ']-ref),
                         hermite64_seconds=tgh,hermite64_logZ_error=abs(gh['logZ']-ref),
                         hermite64_to128_change=abs(gh2['logZ']-gh['logZ']),
                         precision_cutoff_logZ_change=abs(stable['logZ']-bet['logZ']),**bet))
    W=np.array([[1.],[2.],[-1.],[0.],[1.],[-2.],[2.],[1.]])
    b=np.array([.1,-.2,.05,.15,-.1,.3,-.05,.2]); beta=3.
    ref,te=timed(lambda:exact(W,b,beta)); q,tq=timed(lambda:quadrature1(W,b,beta))
    bet,tb=timed(lambda:residue_two_levels(W[:,0],b,beta))
    stable=residue_two_levels(W[:,0],b,beta,dps=100,extra_cut=2)
    gh,tgh=timed(lambda:hermite(W,b,beta,64)); gh2=hermite(W,b,beta,128)
    rows.append(dict(model='integer_heterogeneous_rank1',N=8,r=1,beta=beta,reference_logZ=ref['logZ'],
                     quadrature_logZ_error=abs(q['logZ']-ref['logZ']),quadrature_seconds=tq,
                     reference_seconds=te,bethe_seconds=tb,bethe_abs_logZ_error=abs(bet['logZ']-ref['logZ']),
                     hermite64_seconds=tgh,hermite64_logZ_error=abs(gh['logZ']-ref['logZ']),
                     hermite64_to128_change=abs(gh2['logZ']-gh['logZ']),
                     precision_cutoff_logZ_change=abs(stable['logZ']-bet['logZ']),**bet))
    bethe_gb=[]; h=1e-5
    for i in range(len(b)):
        plus=b.copy(); minus=b.copy(); plus[i]+=h; minus[i]-=h
        pp=residue_two_levels(W[:,0],plus,beta)['logZ_decimal']
        mm=residue_two_levels(W[:,0],minus,beta)['logZ_decimal']
        bethe_gb.append(float((D(pp)-D(mm))/(2*D(str(h)))))
    grad_check=dict(model='integer_heterogeneous_rank1',method='Centered differences of residue logZ, step=1e-5',
                    max_abs_bias_gradient_error=float(np.max(abs(np.array(bethe_gb)-ref['gb']))),
                    unrestricted_W_gradient='Not supported by this integer-weight contour specialization')
    baselines=[]; models=[]
    for r,orders in [(1,(32,64)),(2,(20,40)),(5,(8,12))]:
        rng=np.random.default_rng(4100+r); n=12; beta=1.
        W=rng.normal(0,.65,size=(n,r)); b=rng.normal(0,.15,size=n)
        models.append(dict(N=n,r=r,beta=beta,W=W.tolist(),b=b.tolist()))
        ref,te=timed(lambda:exact(W,b,beta))
        for order in orders:
            result,secs=timed(lambda:hermite(W,b,beta,order))
            baselines.append(dict(model=f'generic_rank{r}',N=n,r=r,beta=beta,method=f'Gauss-Hermite order {order}',
                                  reference_logZ=ref['logZ'],abs_logZ_error=abs(result['logZ']-ref['logZ']),
                                  max_abs_gradient_error=gradient_error(result,ref),seconds=secs,reference_seconds=te,nodes=result['nodes']))
        runs=[]; runtimes=[]
        for seed in range(5):
            result,secs=timed(lambda:ais(W,b,beta,9000+seed),repeats=1)
            runtimes.append(secs); runs.append(result)
        errors=[v['logZ']-ref['logZ'] for v in runs]
        baselines.append(dict(model=f'generic_rank{r}',N=n,r=r,beta=beta,method='AIS 512 particles x 256 bridges, 5 seeds',
                              reference_logZ=ref['logZ'],logZ_RMSE=float(np.sqrt(np.mean(np.array(errors)**2))),
                              min_abs_logZ_error=min(abs(v) for v in errors),max_abs_logZ_error=max(abs(v) for v in errors),
                              mean_max_abs_gradient_error=float(np.mean([gradient_error(v,ref) for v in runs])),
                              median_seconds=statistics.median(runtimes),reference_seconds=te,
                              mean_ess=float(np.mean([v['ess'] for v in runs]))))
    for row in rows:
        nn=row['N']
        ww=np.ones(nn) if row['model']=='uniform_rank1' else np.array([1,2,-1,0,1,-2,2,1])
        bb=np.zeros(nn) if row['model']=='uniform_rank1' else np.array([.1,-.2,.05,.15,-.1,.3,-.05,.2])
        fp,tf=timed(lambda:residue_float64(ww,bb,row['beta']))
        row['float64_residue_seconds']=tf
        row['float64_residue_status']=fp['status']
        row['float64_residue_abs_logZ_error']=abs(fp['logZ']-row['reference_logZ']) if fp['logZ'] is not None else None
    result=dict(scope='Reconstructed lower-strip rank-one descent; higher-rank baselines only',
                bethe=rows,baselines=baselines,models=models,bias_gradient_check=grad_check,
                descendant_integral_checks=[check_descendants(beta) for beta in (4.,8.,16.)],
                versions=dict(numpy=np.__version__,scipy=__import__('scipy').__version__),
                timing='CPU wall time, median of 3 warm-process calls except AIS: 5 independent seeds; one BLAS thread; research Python implementations')
    Path(__file__).with_name('results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': run()
