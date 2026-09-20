"""Successive one-variable residue elimination for integer-weight Ising models.

No numerical quadrature is used by this solver. Pole positions are analytical.
Each coordinate uses the first two descent contributions B0+B1=B0/2,
including inherited poles. The specialization and its limitations are explicit.
"""
import itertools, json, math, time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp, log_ndtr

PI=np.longdouble('3.1415926535897932384626433832795028841971693993751')


def validate(W,b,beta):
    W=np.asarray(W); b=np.asarray(b)
    if W.ndim!=2 or b.shape!=(W.shape[0],) or W.shape[1]<1:
        raise ValueError('W must be N by r and b length N')
    if not np.isfinite(W).all() or not np.equal(W,np.round(W)).all():
        raise ValueError('This analytical-pole specialization requires integer W')
    if not np.isfinite(b).all() or not np.isfinite(beta) or beta<=0:
        raise ValueError('Finite biases and beta>0 required')
    return W.astype(int),b.astype(np.longdouble),np.longdouble(beta)


def root_cutoffs(W,b,beta,relative_tail_target=1e-12):
    """Gaussian-envelope bound for omitted residues, relative to a lower bound on Z.

    The bound is an analytic absolute-tail bound, evaluated in ordinary
    floating point, NOT an outward-rounded numerical certificate.
    """
    n,r=W.shape; beta=float(beta); b=np.array(b,dtype=float)
    c=n*beta/2; spacing=2/n; mu=np.abs(W).sum(axis=0)/n
    parity=W.sum(axis=0)%2
    pref=math.sqrt(n*beta/(2*math.pi))/n
    log_const=(r*math.log(pref)+n*math.log(2)+beta*sum(abs(b))
               +r*n*math.pi**2/(8*beta)+beta/(2*n)*sum(np.abs(W).sum(axis=0)**2))
    log_lower=n*math.log(2)+beta*np.sum(W*W)/(2*n)
    # sum of each lattice Gaussian <= 1+integral/spacing; two centers bound |x|.
    full=2*(1+math.sqrt(math.pi/c)/spacing)
    K=0
    while True:
        L=(2*K+2-parity)/n
        if np.all(L>mu):
            distance=L-mu
            # erfc(t) = 2 Phi(-sqrt(2)t), evaluated stably in log space.
            le=math.log(2)+log_ndtr(-np.sqrt(2*c)*distance)
            tail=np.log(2)+np.logaddexp(-c*distance*distance,
                    math.log(math.sqrt(math.pi)/(2*spacing*math.sqrt(c)))+le)
            log_bound=log_const+(r-1)*math.log(full)+logsumexp(tail)-log_lower
            if log_bound<=math.log(relative_tail_target):
                return [K]*r,float(math.exp(log_bound))
        K+=1
        if K>10000: raise RuntimeError('Residue tail cutoff exceeded budget')


def residue_nodes(W,beta,cutoffs):
    n,r=W.shape
    return [(W[:,a].sum()%2+2*np.arange(-cutoffs[a],cutoffs[a]+1,dtype=np.longdouble))/n
            -1j*PI/(2*beta) for a in range(r)]


def original_integrand(z,W,b,beta):
    z=np.asarray(z,dtype=np.clongdouble)
    return np.exp(-len(b)*beta*np.sum(z*z)/2)*np.prod(2*np.cosh(beta*(b+W@z)))


def build_successive_integrals(W,b,beta,cutoffs):
    """Literal F0 -> F1 -> ... -> Fr. Each returned callable has one fewer argument.

    F_(a+1)(rest) = (sqrt(N beta/2pi)/N) sum_roots F_a(root,rest).
    These are residue contributions, not sampled quadrature weights.
    """
    W,b,beta=validate(W,b,beta); n,r=W.shape
    nodes=residue_nodes(W,beta,cutoffs)
    factor=np.sqrt(n*beta/(2*PI))/n
    stages=[lambda rest:original_integrand(rest,W,b,beta)]
    for a in range(r):
        previous=stages[-1]; poles=nodes[a]
        def next_stage(rest,previous=previous,poles=poles):
            return factor*np.sum(np.array([previous(np.concatenate(([pole],rest))) for pole in poles],
                                          dtype=np.clongdouble),dtype=np.clongdouble)
        stages.append(next_stage)
    return stages


def batched_residue_sum(W,b,beta,cutoffs,batch_size=32768):
    """Batched evaluation of the same nested sums; only analytic pole indices.

    Complex phases at these poles reduce exactly to real signs and sinh/cosh.
    Summing axis zero first is exposed in the partial sums for each outer tuple.
    No Hermite, Legendre, adaptive integration, or real-domain grid is used.
    """
    n,r=W.shape
    ks=[np.arange(-K,K+1,dtype=int) for K in cutoffs]
    parity=W.sum(axis=0)%2
    phase_offset=int(np.sum((parity-W.sum(axis=0))//2))
    odd=np.sum(W,axis=1)%2
    inner=ks[0]; roots_per_inner=len(inner)
    outer=itertools.product(*ks[:0:-1])
    groups=max(1,batch_size//roots_per_inner)
    total=np.longdouble(0); abs_total=np.longdouble(0); evaluations=0
    while True:
        chunk=list(itertools.islice(outer,groups))
        if not chunk: break
        if r==1:
            index=inner[:,None]
        else:
            rest=np.asarray(chunk,dtype=int)[:,::-1]
            index=np.concatenate((np.tile(inner,len(chunk))[:,None],
                                  np.repeat(rest,roots_per_inner,axis=0)),axis=1)
        x=(parity+2*index).astype(np.longdouble)/n
        u=beta*(b+x@W.T.astype(np.longdouble))
        eu=np.exp(u)
        matter=np.where(odd,eu-1/eu,eu+1/eu)
        phase=np.where((index.sum(axis=1)+phase_offset)%2,-1,1)
        values=phase*np.exp(-n*beta*np.sum(x*x,axis=1)/2+r*n*PI*PI/(8*beta))*np.prod(matter,axis=1)
        if not np.isfinite(values).all(): raise FloatingPointError('Residue terms exceed numerical range')
        # First eliminate m1 for each fixed tuple of the other residue locations.
        first_stage=values.reshape(-1,roots_per_inner).sum(axis=1,dtype=np.longdouble)
        total+=first_stage.sum(dtype=np.longdouble)
        abs_total+=np.abs(values).sum(dtype=np.longdouble)
        evaluations+=len(values)
    pref=(np.sqrt(n*beta/(2*PI))/n)**r
    Z=pref*total
    if Z<=0: raise FloatingPointError('Cancellation produced nonpositive partition estimate')
    return Z,dict(pole_combinations=evaluations,cancellation_digits=float(np.log10(abs_total/abs(total))))


def solve(W,b,beta,relative_tail_target=1e-12,extra_poles=0,max_terms=50_000_000):
    start=time.perf_counter(); W,b,beta=validate(W,b,beta); n,r=W.shape
    cutoffs,tail=root_cutoffs(W,b,beta,relative_tail_target)
    cutoffs=[k+extra_poles for k in cutoffs]
    terms=math.prod(2*k+1 for k in cutoffs)
    if terms>max_terms:
        return dict(status='residue_budget_exceeded',N=n,r=r,beta=float(beta),pole_combinations=terms,cutoffs=cutoffs)
    Z,diag=batched_residue_sum(W,b,beta,cutoffs)
    # Closed-form specialization identity for the infinite nested residue sum.
    S=sum(math.exp(-n*math.pi**2*l*(l+1)/(2*float(beta))) for l in range(1,100))
    return dict(status='computed',logZ=float(np.log(Z)),Z=str(Z),N=n,r=r,beta=float(beta),
                seconds=time.perf_counter()-start,cutoffs=cutoffs,**diag,
                analytical_logZ_truncation=r*math.log1p(S),
                estimated_relative_residue_tail_bound=tail,
                arithmetic='numpy longdouble',decimal_precision=int(np.finfo(np.longdouble).precision),
                certified_numerical_error=False)


def exact(W,b,beta):
    n,r=W.shape
    x=((np.arange(2**n,dtype=np.uint64)[:,None]>>np.arange(n,dtype=np.uint64))&1).astype(float)*2-1
    field=x@W
    return float(logsumexp(beta*(x@b+np.sum(field*field,axis=1)/(2*n))))


def exact_first_partial(W,b,beta,y):
    n,r=W.shape
    x=((np.arange(2**n,dtype=np.uint64)[:,None]>>np.arange(n,dtype=np.uint64))&1).astype(float)*2-1
    field=x@W
    return float(np.exp(logsumexp(beta*(x@b+field[:,0]**2/(2*n)+field[:,1:]@y))
                        -n*beta*np.dot(y,y)/2))


def main():
    cases=[]
    for n,r,beta in [(4,1,1.),(4,2,1.),(4,2,4.),(6,3,2.),(8,5,2.)]:
        rng=np.random.default_rng(7200+n+r)
        W=rng.integers(-1,2,size=(n,r)); b=rng.normal(0,.15,n)
        if r==2:
            W=np.array([[1,1],[1,-1],[-1,1],[0,1]])
        assert np.linalg.matrix_rank(W)==r
        ref_start=time.perf_counter(); ref=exact(W,b,beta); ref_time=time.perf_counter()-ref_start
        result=solve(W,b,beta)
        record=dict(W=W.tolist(),b=b.tolist(),reference_logZ=ref,reference_seconds=ref_time,**result)
        if result['status']=='computed':
            record['actual_abs_logZ_error']=abs(result['logZ']-ref)
            if r<=3:
                refined=solve(W,b,beta,extra_poles=2)
                record['cutoff_refinement_change']=abs(refined['logZ']-result['logZ'])
            if r==2:
                stages=build_successive_integrals(W,b,beta,result['cutoffs'])
                trace=[]
                for y in (-1.,-.5,0.,.5,1.):
                    value=stages[1](np.array([y])); truth=exact_first_partial(W,b,beta,np.array([y]))
                    trace.append(dict(m2=y,F1_real=float(value.real),F1_imag=float(value.imag),
                                      exact_partial=truth,relative_error=float(value.real/truth-1)))
                sequential=stages[2](np.array([]))
                record['literal_sequential_logZ']=float(np.log(sequential.real))
                record['literal_vs_batched_logZ_difference']=abs(record['literal_sequential_logZ']-result['logZ'])
                record['first_integration_trace']=trace
        cases.append(record)
        Path(__file__).with_name('results.json').write_text(json.dumps(cases,indent=2)+'\n')
        print(n,r,beta,result['status'],result.get('pole_combinations'),result.get('seconds'),record.get('actual_abs_logZ_error'),flush=True)


if __name__=='__main__': main()
