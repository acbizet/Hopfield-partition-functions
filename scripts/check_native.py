"""Check native double kernels and time a 100-digit enumeration baseline."""
import itertools,json,statistics,subprocess,time
from pathlib import Path
import mpmath as mp
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'experiments/high-precision'
mp.mp.dps=100
raw=json.loads((D/'model.json').read_text())
n=len(raw['W']);r=len(raw['W'][0]);beta=mp.mpf(raw['beta'])
b=[mp.mpf(float(v).as_integer_ratio()[0])/float(v).as_integer_ratio()[1] for v in raw['b']]
def enumerate_logz():
    terms=[]
    for s in itertools.product((-1,1),repeat=n):
        h=[sum(raw['W'][i][a]*s[i] for i in range(n)) for a in range(r)]
        terms.append(mp.exp(beta*(mp.fdot(b,s)+mp.mpf(sum(t*t for t in h))/(2*n))))
    return mp.log(mp.fsum(terms))
ref=enumerate_logz();times=[]
for _ in range(3):
    start=time.perf_counter();enumerate_logz();times.append(time.perf_counter()-start)
record=json.loads((ROOT/'results/recorded/high-precision/results.json').read_text())
if abs(ref-mp.mpf(record['reference_logZ'])) > mp.mpf('1e-95'):
    raise RuntimeError('Fresh enumeration disagrees with stored reference')
out={'reference_logZ':mp.nstr(ref,100),'enumeration':{'precision_digits':100,'states':2**n,'median_seconds':statistics.median(times),'samples_seconds':times},'native':{}}
for method,order in [('positive',9),('gh',18)]:
    rule='-' if method=='positive' else str(D/'rules'/f'{order}.txt')
    value=json.loads(subprocess.check_output([str(D/'evaluate'),str(D/'input.txt'),rule,'double',method,str(order),'3'],text=True))
    error=abs(mp.mpf(value['logZ'])-ref)
    if error > mp.mpf('1e-9'):raise RuntimeError(f'{method} failed: {error}')
    value['absolute_logZ_error']=mp.nstr(error,30);out['native'][method]=value
out['note']='Local smoke check; timings are not a new full precision sweep. Enumeration uses Python and 100 digits; kernels use C++ binary64, so this is not a matched-implementation speed ranking.'
print(json.dumps(out,indent=2))
