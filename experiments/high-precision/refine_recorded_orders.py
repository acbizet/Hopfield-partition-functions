"""Finish the recorded run's even-order scan by checking intermediate orders.

The current prepare.py/benchmark.py already scan consecutive orders, so this
script is only needed to reproduce this initial run's post-scan refinement.
"""
import json,subprocess,sys
from pathlib import Path
import mpmath as mp
ROOT=Path(__file__).resolve().parent;mp.mp.dps=100

def main():
 path=ROOT/'results.json';r=json.loads(path.read_text());c=json.loads((ROOT/'calibration.json').read_text());ref=mp.mpf(r['reference_logZ'])
 assert len(r['rows'])==6
 for row in r['rows']:
  if row['tolerance'] not in ('1e-15','1e-18'):continue
  order=25 if row['tolerance']=='1e-15' else 29
  args=[str(ROOT/'evaluate'),str(ROOT/'input.txt'),str(ROOT/'rules'/f'{order}.txt'),row['precision'],'gh',str(order),'3']
  x=json.loads(subprocess.check_output(args,text=True));x.update(method='gh',order=order,precision=row['precision'],absolute_logZ_error=mp.nstr(abs(mp.mpf(x['logZ'])-ref),30))
  assert mp.mpf(x['absolute_logZ_error'])<mp.mpf(row['tolerance'])
  row['superseded_even_order_gh']=row['gh'];row['gh']=x
  # The superseded accepted even order is also the finer actual check.
  x['finer_check']=dict(row['superseded_even_order_gh'])
  row['speedup_positive_over_gh']=x['median_seconds']/row['positive']['median_seconds']
  print('refined',row['tolerance'],x['median_seconds'],x['absolute_logZ_error'],row['speedup_positive_over_gh'],flush=True)
 path.write_text(json.dumps(r,indent=2)+'\n')

if __name__=='__main__':main()
