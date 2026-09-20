"""Measure the full tensor evaluators against a 140-digit spin reference."""
import json,subprocess,sys,time
from pathlib import Path
import mpmath as mp
ROOT=Path(__file__).resolve().parent;mp.mp.dps=100

def main():
 c=json.loads((ROOT/'calibration.json').read_text());ref=mp.mpf(c['reference_logZ'])
 out=dict(reference_logZ=c['reference_logZ'],rows=[],timing='median of 3 evaluations after one full warm evaluation; 1 CPU thread; cached GH rules',
          kernel='shared native C++ full tensor traversal; model-dependent exponential preparation timed; no spin expansion')
 def save():(ROOT/'results.json').write_text(json.dumps(out,indent=2)+'\n')
 def call(precision,method,order,repeats):
  print('START',precision,method,order,'repeats',repeats,flush=True)
  args=[str(ROOT/'evaluate'),str(ROOT/'input.txt'),'-' if method=='positive' else str(ROOT/'rules'/f'{order}.txt'),precision,method,str(order),str(repeats)]
  x=json.loads(subprocess.check_output(args,text=True));err=abs(mp.mpf(x['logZ'])-ref)
  x.update(precision=precision,method=method,order=order,absolute_logZ_error=mp.nstr(err,30))
  print('DONE',method,order,x['median_seconds'],'error',x['absolute_logZ_error'],flush=True)
  return x
 for setting in c['settings']:
  tol=mp.mpf(setting['tolerance']);precision=setting['precision'];row=dict(tolerance=setting['tolerance'],precision=precision,adjustments=[])
  for method in ('positive','gh'):
   order=setting[method]['order'];result=call(precision,method,order,3)
   while mp.mpf(result['absolute_logZ_error'])>tol:
    row['adjustments'].append(result);order+=1
    result=call(precision,method,order,3)
    if len(row['adjustments'])>4:raise RuntimeError('Precision or convergence failure; inspect recorded values')
   row[method]=result
   # Finer mathematical tensor values already calibrated at 85 digits.
   # Verify the actual native computation at the next order for sub-quad targets.
   # At quad, use a single higher-order evaluation (plus its warm call) at the
   # tightest target only, to avoid duplicate expensive convergence checks.
   if precision!='quad' or setting['tolerance']=='1e-30':
    finer=call(precision,method,order+1,1)
    row[method]['finer_check']=finer
    if mp.mpf(finer['absolute_logZ_error'])>tol:raise RuntimeError('Finer actual evaluation did not retain accuracy')
   save()
  row['speedup_positive_over_gh']=row['gh']['median_seconds']/row['positive']['median_seconds']
  out['rows'].append(row);save();print('ROW',json.dumps(row),flush=True)
 print('COMPLETE',flush=True)

if __name__=='__main__':main()
