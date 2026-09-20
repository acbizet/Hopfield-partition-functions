"""Regenerate figures from preserved measurements, not new timings."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
x=json.loads((R/'results/recorded/high-precision/results.json').read_text())['rows']
fig,ax=plt.subplots(figsize=(8,4.3),layout='constrained')
digits=[int(-np.log10(float(a['tolerance']))) for a in x]
for key,label,color in [('positive','Positive lattice','#007f82'),('gh','Tensor Gauss-Hermite','#bc5b35')]:
 ax.plot(digits,[a[key]['median_seconds'] for a in x],'o-',label=label,color=color,lw=2)
ax.set_yscale('log');ax.set_xticks(digits);ax.set_xlabel('Accuracy target: absolute error in log Z <= 10^(-d), d shown')
ax.set_ylabel('Median evaluation time (seconds, log scale)');ax.set_title('Recorded rank-five benchmark | N = 8, beta = 2',loc='left',fontweight='bold')
ax.grid(axis='y',alpha=.2);ax.legend(frameon=False)
fig.savefig(R/'figures/precision-runtime.png');plt.close(fig)
x=json.loads((R/'results/recorded/critical-behavior/results.json').read_text())['critical']
fig,ax=plt.subplots(figsize=(8,4.3),layout='constrained')
xx=np.sqrt([a['N'] for a in x]);ax.plot(xx,[a['chi'] for a in x],'o',color='#007f82',label='Grouped finite-state reference')
ax.plot(xx,[a['corrected_prediction'] for a in x],'-',color='#bc5b35',label='Analytic finite-size prediction')
ax.set_xlabel('Square root of N');ax.set_ylabel('Susceptibility');ax.set_title('Correlated rank-three model | c = 1/3, beta = 0.6',loc='left',fontweight='bold');ax.grid(alpha=.2);ax.legend(frameon=False)
fig.savefig(R/'figures/critical-susceptibility.png');plt.close(fig)
print('Two figures regenerated from recorded results.')
