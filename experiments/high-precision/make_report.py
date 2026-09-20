"""Insert measured values and package the reproducible comparison."""
import json,zipfile,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
 r=json.loads((ROOT/'results.json').read_text());assert len(r['rows'])==6
 p=ROOT/'README.md';s=p.read_text()
 labels={'double':'binary64 (~16 digits)','longdouble':'extended (~19 digits)','quad':'binary128 (~34 digits)'}
 lines=['| Absolute log-Z target | Working precision | Positive formula (s) | Gauss-Hermite (s) | GH / positive time |',
        '|---:|---|---:|---:|---:|']
 for row in r['rows']:
  lines.append(f"| {row['tolerance']} | {labels[row['precision']]} | {row['positive']['median_seconds']:.6f} | {row['gh']['median_seconds']:.6f} | {row['speedup_positive_over_gh']:.3f} |")
 lines+=['','A ratio above one means the positive formula was faster. The large jump in absolute time on entering binary128 reflects software quadruple-precision arithmetic as well as the larger sums.']
 s=s.replace('<!-- RESULTS -->','\n'.join(lines))
 lines=['| Target | Lattice K | Lattice points | GH order q | GH points | Actual lattice log-Z error | Actual GH log-Z error |',
        '|---:|---:|---:|---:|---:|---:|---:|']
 for row in r['rows']:
  a,b=row['positive'],row['gh']
  lines.append(f"| {row['tolerance']} | {a['order']} | {a['points']:,} | {b['order']} | {b['points']:,} | {float(a['absolute_logZ_error']):.3e} | {float(b['absolute_logZ_error']):.3e} |")
 lines+=['','The initial run checked even GH orders. Intermediate odd orders were then tested: q=25 replaces q=26 at 1e-15, and q=29 replaces q=30 at 1e-18. The superseded timings are retained in results.json for transparency. The current preparation and benchmark scripts search consecutive orders directly. At 1e-18, q=28 failed its actual-error check despite having a mathematical truncation error slightly below the requested tolerance; rounding pushed it over the limit.']
 s=s.replace('<!-- ACCURACY -->','\n'.join(lines))
 last=r['rows'][-1];a,b=last['positive'],last['gh']
 text=f"On this fixed example, a speed advantage emerges as the required error decreases. At 1e-30, the positive formula needs {a['points']:,} points, while Gauss-Hermite needs {b['points']:,}: about {b['points']/a['points']:.2f} times as many. The measured runtime advantage is {last['speedup_positive_over_gh']:.2f} times. The difference is primarily the number of tensor points, because both paths use the same arithmetic and evaluation structure.\n\nThe Gaussian lattice tail decays rapidly with the squared cutoff distance. At this precision, the positive formula retains 30 or 31 points per direction, compared with 40 quadrature nodes per direction. Raising these per-direction counts to the fifth power explains the sizable total difference. This is an explanation of the observed case, not a universal asymptotic complexity result.\n\nThe advantage need not increase monotonically at every tolerance: cutoff orders change in integer steps, arithmetic modes change at precision thresholds, and the achieved errors can be substantially smaller than their respective targets."
 s=s.replace('<!-- INTERPRETATION -->',text);assert '<!--' not in s;p.write_text(s)
 import mpmath as mp
 (ROOT/'requirements.txt').write_text('mpmath=='+mp.__version__+'\n')
 names=['README.md','evaluate.cpp','prepare.py','benchmark.py','refine_recorded_orders.py','make_report.py','model.json','input.txt','calibration.json','results.json','environment.json','requirements.txt']
 target=ROOT.parent/'high-precision-hopfield.zip'
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
  for name in names:z.write(ROOT/name,'high-precision-hopfield/'+name)
  for path in sorted((ROOT/'rules').glob('*.txt')):z.write(path,'high-precision-hopfield/rules/'+path.name)
 with zipfile.ZipFile(target) as z:assert z.testzip() is None
 print('Saved',target,'bytes',target.stat().st_size)
 for row in r['rows']:print(row['tolerance'],row['positive']['median_seconds'],row['gh']['median_seconds'],row['speedup_positive_over_gh'])

if __name__=='__main__':main()
