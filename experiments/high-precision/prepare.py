"""High-precision reference, nodes, and untimed truncation calibration.

Biases are exactly the original binary64 values, not reinterpreted rounded
decimal measurements. The factorized 256-spin diagnostic is used only for
selecting truncation settings; the timed C++ evaluators visit all tensor points.
"""
import itertools,json,sys,time,math
from pathlib import Path
import mpmath as mp
ROOT=Path(__file__).parent

def main():
 raw=json.loads((ROOT/'model.json').read_text());W=raw['W'];n=len(W);r=len(W[0]);mp.mp.dps=100
 b=[mp.mpf(float(v).as_integer_ratio()[0])/float(v).as_integer_ratio()[1] for v in raw['b']];beta=mp.mpf(raw['beta'])
 def spin_data():
  out=[]
  for sigma in itertools.product((-1,1),repeat=n):
   h=tuple(sum(W[i][a]*sigma[i] for i in range(n)) for a in range(r))
   coeff=mp.exp(beta*mp.fdot(b,sigma));out.append((h,coeff))
  return out
 spins=spin_data();ref=mp.log(mp.fsum(coeff*mp.exp(beta*sum(t*t for t in h)/(2*n)) for h,coeff in spins))
 ref100=mp.nstr(ref,100);mp.mp.dps=140
 b=[mp.mpf(float(v).as_integer_ratio()[0])/float(v).as_integer_ratio()[1] for v in raw['b']]
 spins=spin_data();ref140=mp.log(mp.fsum(coeff*mp.exp(beta*sum(t*t for t in h)/(2*n)) for h,coeff in spins))
 assert abs(ref140-mp.mpf(ref100))<mp.mpf('1e-98')
 (ROOT/'input.txt').write_text('\n'.join([str(raw['beta'])]+[mp.nstr(v,110) for v in b]+[' '.join(map(str,row)) for row in W])+'\n')
 result=dict(reference_logZ=mp.nstr(ref140,120),reference_100_digits=ref100,
             bias_interpretation='exact original IEEE binary64 values',scans={'positive':[],'gh':[]},settings=[],rule_setup_seconds={})
 mp.mp.dps=85;spins=spin_data();ref=+ref140
 parity=[sum(row[a] for row in W)%2 for a in range(r)]
 hs=[sorted(set(h[a] for h,c in spins)) for a in range(r)]
 G=mp.fsum(mp.exp(-2*beta*j*j/n) for j in range(-24,25))
 def diagnose(method,order,rule=None):
  moments=[]
  for a in range(r):
   if method=='positive':
    nodes=[mp.mpf(parity[a]+2*k)/n for k in range(-order,order+1-parity[a])]
    weights=[mp.exp(-n*beta*x*x/2)/G for x in nodes]
   else:
    nodes=[x*mp.sqrt(2/(n*beta)) for x in rule[0]];weights=[w/mp.sqrt(mp.pi) for w in rule[1]]
   moments.append({h:mp.fsum(w*mp.exp(beta*h*x) for x,w in zip(nodes,weights)) for h in hs[a]})
  z=mp.log(mp.fsum(coeff*mp.fprod(moments[a][h[a]] for a in range(r)) for h,coeff in spins))
  row=dict(order=order,logZ=mp.nstr(z,75),absolute_error=mp.nstr(abs(z-ref),30))
  result['scans'][method].append(row);print('calibration',method,order,row['absolute_error'],flush=True)
  (ROOT/'calibration.json').write_text(json.dumps(result,indent=2)+'\n')
  return abs(z-ref)
 for K in range(9,21):
  err=diagnose('positive',K)
  if K>9 and err<mp.mpf('1e-35') and mp.mpf(result['scans']['positive'][-2]['absolute_error'])<mp.mpf('1e-32'):break
 (ROOT/'rules').mkdir(exist_ok=True)
 for q in range(17,65):
  t=time.perf_counter();X,V=mp.gauss_quadrature(q,'hermite');secs=time.perf_counter()-t
  assert abs(mp.fsum(V)-mp.sqrt(mp.pi))<mp.mpf('1e-78')
  (ROOT/'rules'/f'{q}.txt').write_text('\n'.join(mp.nstr(x,80)+' '+mp.nstr(w,80) for x,w in zip(X,V))+'\n')
  result['rule_setup_seconds'][str(q)]=secs
  err=diagnose('gh',q,(X,V))
  if q>17 and err<mp.mpf('1e-35') and mp.mpf(result['scans']['gh'][-2]['absolute_error'])<mp.mpf('1e-32'):break
 for tol,precision in [('1e-9','double'),('1e-12','double'),('1e-15','longdouble'),('1e-18','longdouble'),('1e-24','quad'),('1e-30','quad')]:
  row=dict(tolerance=tol,precision=precision)
  for method in ('positive','gh'):
   scan=result['scans'][method]
   idx=next(i for i in range(len(scan)-1) if mp.mpf(scan[i]['absolute_error'])<mp.mpf(tol) and mp.mpf(scan[i+1]['absolute_error'])<mp.mpf(tol))
   row[method]=dict(order=scan[idx]['order'],finer_order=scan[idx+1]['order'],calibrated_truncation_error=scan[idx]['absolute_error'])
  result['settings'].append(row)
 (ROOT/'calibration.json').write_text(json.dumps(result,indent=2)+'\n')
 print('settings',json.dumps(result['settings']),flush=True)

if __name__=='__main__':main()
