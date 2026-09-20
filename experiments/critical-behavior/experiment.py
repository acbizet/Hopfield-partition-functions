"""Correlated Hopfield critical fluctuations: independent sums and residues.

Run: OPENBLAS_NUM_THREADS=1 python experiment.py
Dependencies: numpy, scipy, mpmath. All residue observables use analytic
field derivatives, never numerical integration or finite differences.
"""
from pathlib import Path
import itertools, json, math, sys, time
import numpy as np
from scipy.special import gammaln, gamma, gammainc
from scipy.integrate import quad
import mpmath as mp

OUT=Path(__file__).parent
BETA_C=3/5
A=123/2500
B=2/25
D=219/15625

def quartic_moment(k):
    return A**(-k/4)*gamma((k+1)/4)/gamma(1/4)

LEADING=BETA_C*quartic_moment(2)
CONSTANT=BETA_C*(D*(quartic_moment(8)-quartic_moment(2)*quartic_moment(6))
                -B*(quartic_moment(4)-quartic_moment(2)**2))-1

def grouped_exact(n,beta,histogram=False):
    """Full finite spin sum compressed only by four equal-row groups.

    Group sizes N/2,N/6,N/6,N/6, rows +++,-++,+-+,++-.
    This is independent of the residue formula and uses no truncation.
    """
    assert n%6==0
    start=time.perf_counter()
    n0,n1=n//2,n//6
    s=np.arange(-n1,n1+1,2,dtype=float)
    ix=np.stack([a.ravel() for a in np.meshgrid(*([np.arange(n1+1)]*3),indexing='ij')],axis=1)
    ss=s[ix]; ts=ss.sum(axis=1); sq=(ss*ss).sum(axis=1)
    lc1=gammaln(n1+1)-gammaln(np.arange(n1+1)+1)-gammaln(n1-np.arange(n1+1)+1)
    lc=lc1[ix].sum(axis=1)
    peak=-np.inf; z=v2=v4=0.; mass=np.zeros(4*n+1)
    for j in range(n0+1):
        s0=2*j-n0; t=s0+ts
        norm2=3*t*t-4*t*ts+4*sq
        lw=lc+gammaln(n0+1)-gammaln(j+1)-gammaln(n0-j+1)+beta*norm2/(2*n)
        newpeak=max(peak,float(lw.max())); factor=np.exp(peak-newpeak)
        z*=factor;v2*=factor;v4*=factor
        if histogram:mass*=factor
        w=np.exp(lw-newpeak); q=3*s0+ts; m2=q*q/(3*n*n)
        z+=w.sum();v2+=np.dot(w,m2);v4+=np.dot(w,m2*m2)
        if histogram:mass+=np.bincount((q+2*n).astype(int),weights=w,minlength=4*n+1)
        peak=newpeak
    row=dict(N=n,beta=beta,logZ=float(peak+np.log(z)),chi=float(beta*n*v2/z),
             scaled_overlap_second=float(np.sqrt(n)*v2/z),
             scaled_overlap_fourth=float(n*v4/z),
             binder=float(1-(v4/z)/(3*(v2/z)**2)),
             grouped_states=(n0+1)*(n1+1)**3,seconds=time.perf_counter()-start)
    if histogram:
        row['distribution_q']=np.arange(-2*n,2*n+1).tolist()
        row['distribution_prob']=(mass/z).tolist()
    return row

def brute_force(n,beta):
    rows=np.array([[1,1,1],[-1,1,1],[1,-1,1],[1,1,-1]])
    w=np.repeat(rows,[n//2,n//6,n//6,n//6],axis=0)
    spin=2*((np.arange(2**n)[:,None]>>np.arange(n))&1)-1
    overlaps=spin@w; m=overlaps.sum(axis=1)/(math.sqrt(3)*n)
    lw=beta*(overlaps*overlaps).sum(axis=1)/(2*n);p=np.exp(lw-lw.max());z=p.sum()
    return dict(logZ=float(lw.max()+np.log(z)),chi=float(beta*n*np.dot(p,m*m)/z))

def grouped_high_precision(n=12,dps=100):
    mp.mp.dps=dps;beta=mp.mpf(3)/5;n0,n1=n//2,n//6
    z=mp.mpf(0);v2=mp.mpf(0)
    for js in itertools.product(range(n0+1),*([range(n1+1)]*3)):
        s0=2*js[0]-n0;ss=[2*j-n1 for j in js[1:]];t=s0+sum(ss)
        norm2=sum((t-2*s)**2 for s in ss)
        mult=math.comb(n0,js[0])*math.prod(math.comb(n1,j) for j in js[1:])
        w=mult*mp.exp(beta*norm2/(2*n));q=3*s0+sum(ss)
        z+=w;v2+=w*q*q/(3*n*n)
    return dict(N=n,dps=dps,logZ=mp.nstr(mp.log(z),80),chi=mp.nstr(beta*n*v2/z,80))

def pole_tail_relative_bound(n,beta,k):
    """Analytic envelope / Jensen lower bound on Z, evaluated at high precision.

    Not an interval-arithmetic certificate for the implementation.
    """
    n=mp.mpf(n);beta=mp.mpf(beta);c=n*beta/2;h=2/n;ell=2*(k+1)/n
    if ell<=1:return mp.inf
    tail=2*(mp.exp(-c*(ell-1)**2)+mp.sqrt(mp.pi)/(2*h*mp.sqrt(c))*mp.erfc(mp.sqrt(c)*(ell-1)))
    full=2*(1+mp.sqrt(mp.pi)/(h*mp.sqrt(c)))
    psi=mp.fsum(mp.exp(-n*mp.pi**2*j*(j+1)/(2*beta)) for j in range(30))
    pref=(mp.sqrt(n*beta/(2*mp.pi))/(n*psi))**3
    return pref*mp.exp(3*n*mp.pi**2/(8*beta)+3*n*beta/2-3*beta/2)*3*tail*full**2

def residue_observable(n=12,k=40,dps=80):
    """Explicit residue sum and exact first/second field derivatives.

    At each pole, all four matter factors are powers of 2*sinh.
    We differentiate the products polynomially, so removable zero singularities
    need no coth division. Permutation symmetry reduces work but no spin
    expansion, numerical quadrature, or reference normalization is used.
    """
    start=time.perf_counter();mp.mp.dps=dps
    assert n%12==0  # even group counts and even parity offsets
    beta=mp.mpf(3)/5;rt3=mp.sqrt(3)
    n0,n1=n//2,n//6
    def group(count,field_coefficient):
        out={};v=beta*field_coefficient
        for j in range(-3*k,3*k+1):
            u=2*beta*j/n;e=mp.exp(u);s=e-1/e;c=e+1/e
            g=s**count
            g1=count*v*c*s**(count-1)
            g2=count*v*v*g+count*(count-1)*v*v*c*c*s**(count-2)
            out[j]=(g,g1,g2)
        return out
    large=group(n0,rt3);small=group(n1,1/rt3)
    gaussian={j:mp.exp(-2*beta*j*j/n) for j in range(-k,k+1)}
    z=mp.mpf(0);z1=mp.mpf(0);z2=mp.mpf(0);az=mp.mpf(0);az2=mp.mpf(0);calls=0
    for ks in itertools.combinations_with_replacement(range(-k,k+1),3):
        t=sum(ks);p,p1,p2=large[t]
        for ka in ks:
            g,g1,g2=small[t-2*ka]
            p,p1,p2=p*g,p1*g+p*g1,p2*g+2*p1*g1+p*g2
        mult=1 if ks[0]==ks[2] else (3 if ks[0]==ks[1] or ks[1]==ks[2] else 6)
        weight=mult*gaussian[ks[0]]*gaussian[ks[1]]*gaussian[ks[2]]*(-1 if t%2 else 1)
        term=weight*p;term2=weight*p2
        z+=term;z1+=weight*p1;z2+=term2;az+=abs(term);az2+=abs(term2);calls+=1
    psi=mp.fsum(mp.exp(-n*mp.pi**2*j*(j+1)/(2*beta)) for j in range(30))
    logpref=3*(mp.log(mp.sqrt(n*beta/(2*mp.pi))/n)-mp.log(psi))+3*n*mp.pi**2/(8*beta)
    logz=mp.log(z)+logpref
    chi=(z2/z-(z1/z)**2)/(beta*n)
    delta0=pole_tail_relative_bound(n,beta,k)
    # |Z'' tail| <= beta^2 * (sum |a_i|)^2 * |Z tail envelope|.
    # At h=0, Z'=0 by symmetry, and sum |a_i|=2N/sqrt(3).
    chi_tail=(4*beta*n/3+abs(chi))*delta0
    return dict(N=n,beta=float(beta),K=k,dps=dps,pole_tuples=(2*k+1)**3,evaluated_symmetric_tuples=calls,
                logZ=mp.nstr(logz,40),chi=mp.nstr(chi,40),
                first_derivative_ratio=mp.nstr(z1/z,8),
                cancellation_digits_Z=float(mp.log10(az/abs(z))),
                cancellation_digits_Z_second_derivative=float(mp.log10(az2/abs(z2))),
                relative_Z_tail_bound=mp.nstr(delta0,8),absolute_chi_tail_bound=mp.nstr(chi_tail,8),
                seconds=time.perf_counter()-start)

def window_prediction(s):
    shift=max(0,s*s/(16*A))
    z=quad(lambda t:math.exp(s*t*t/2-A*t**4-shift),-np.inf,np.inf,epsabs=1e-11)[0]
    z2=quad(lambda t:t*t*math.exp(s*t*t/2-A*t**4-shift),-np.inf,np.inf,epsabs=1e-11)[0]
    return BETA_C*z2/z

def main():
    result=dict(model=dict(rank=3,correlation=1/3,beta_c=BETA_C,
        pattern_rows=[[1,1,1],[-1,1,1],[1,-1,1],[1,1,-1]],group_fractions=[1/2,1/6,1/6,1/6]),
        coefficients=dict(quartic=A,transverse=B,sextic=D,leading=LEADING,constant=CONSTANT),critical=[],window=[],residues=[])
    def save(): (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    brute=brute_force(12,BETA_C);group=grouped_exact(12,BETA_C)
    assert abs(brute['logZ']-group['logZ'])<1e-12 and abs(brute['chi']-group['chi'])<1e-12
    result['brute_force_check']=dict(brute=brute,grouped=group)
    for n in (12,24,48,96,192,384):
        row=grouped_exact(n,BETA_C,histogram=(n==192))
        row['leading_prediction']=LEADING*math.sqrt(n)
        row['corrected_prediction']=row['leading_prediction']+CONSTANT
        row['corrected_relative_error']=abs(row['corrected_prediction']/row['chi']-1)
        if 'distribution_prob' in row:
            mass=np.array(row['distribution_prob']);x=np.array(row['distribution_q'])/(np.sqrt(3)*n**.75)
            cdf=.5+np.sign(x)*gammainc(.25,A*x**4)/2
            right=mass.cumsum();left=right-mass
            row['quartic_limit_KS_distance']=float(max(abs(right-cdf).max(),abs(left-cdf).max()))
        result['critical'].append(row);save()
        print('critical',n,row['chi'],row['corrected_prediction'],row['seconds'],flush=True)
    for s in (-1.,0.,1.):
        prediction=window_prediction(s)
        for n in (48,96,192):
            row=grouped_exact(n,BETA_C+s/math.sqrt(n))
            row.update(s=s,predicted_chi_over_sqrtN=prediction,chi_over_sqrtN=row['chi']/math.sqrt(n))
            result['window'].append(row);save()
        print('window',s,prediction,flush=True)
    mp.mp.dps=60;k=1
    while pole_tail_relative_bound(12,mp.mpf(3)/5,k)>mp.mpf('1e-14'):k+=1
    print('residue selected K',k,flush=True)
    for cutoff,dps in ((k,80),(k+2,100)):
        row=residue_observable(12,cutoff,dps);result['residues'].append(row);save()
        print('residue',row,flush=True)
        assert abs(float(row['logZ'])-brute['logZ'])<1e-12
        assert abs(float(row['chi'])-brute['chi'])<1e-12
    result['high_precision_reference']=grouped_high_precision();save()
    for row in result['residues']:
        assert abs(mp.mpf(row['chi'])-mp.mpf(result['high_precision_reference']['chi']))<mp.mpf('1e-35')
    print('done',flush=True)

if __name__=='__main__':main()
