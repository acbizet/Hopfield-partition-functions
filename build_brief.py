"""Build a four-page scientific brief from preserved measurements."""
from pathlib import Path
import json
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
R=Path(__file__).resolve().parents[1]
INK=colors.HexColor('#183047');TEAL=colors.HexColor('#007f82');LIGHT=colors.HexColor('#eef5f6')
S={
 'title':ParagraphStyle('title',fontName='Helvetica-Bold',fontSize=25,leading=29,textColor=INK,spaceAfter=15),
 'h':ParagraphStyle('h',fontName='Helvetica-Bold',fontSize=17,leading=21,textColor=INK,spaceAfter=13),
 'sub':ParagraphStyle('sub',fontName='Helvetica',fontSize=11,leading=16,textColor=TEAL,spaceAfter=15),
 'p':ParagraphStyle('p',fontName='Helvetica',fontSize=10,leading=15,textColor=INK,spaceAfter=10),
 'small':ParagraphStyle('small',fontName='Helvetica',fontSize=8.5,leading=12,textColor=INK,spaceAfter=8),
 'cell':ParagraphStyle('cell',fontName='Helvetica',fontSize=9,leading=12,textColor=INK),
 'head':ParagraphStyle('head',fontName='Helvetica-Bold',fontSize=9,leading=12,textColor=colors.white)}
story=[]
def p(s,k='p'):story.append(Paragraph(s,S[k]))
def table(headers,rows,widths):
 data=[[Paragraph(str(v),S['head']) for v in headers]]+[[Paragraph(str(v),S['cell']) for v in row] for row in rows]
 t=Table(data,colWidths=widths,repeatRows=1)
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),INK),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,LIGHT]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8)]))
 story.extend([t,Spacer(1,12)])
def page(title):story.append(PageBreak());p(title,'h')
p('Hopfield partition functions','title')
p('Positive lattice representations, complex residues and critical observables','sub')
p('Scientific research brief | Prepared 20 September 2026','small')
p('<b>Research question.</b> When do analytic representations of structured binary models lead to stable, accurate and computationally useful normalization and observable calculations?')
p('The work studies N binary spins, r integer-weight patterns, real fields and positive inverse temperature beta. The energy retains the diagonal self interaction. Detailed equations, code and exact inputs accompany this brief in the repository.')
table(['Result','Evidence and boundary'],[
('Positive lattice identity','Completing the Gaussian square gives an exact infinite positive sum for Z. Integer overlap parity fixes the lattice. Finite cutoffs and rounding still matter.'),
('High-precision comparison','A recorded N = 8, r = 5 example gives a 4.09x speed ratio over the tested tensor Gauss-Hermite evaluator at a 10^-30 absolute error target in log Z.'),
('Critical observables','A correlated rank-three model reproduces a quartic critical law and square-root susceptibility scaling, with an explicit constant correction.'),
('Descent limitations','A suppressed descendant does not generally locate the physical transition or certify retrieval reliability.')],[130,365])
p('<b>Interpretation.</b> These are research prototypes and controlled small-model comparisons. The timing advantage belongs to the positive reformulation. At N = 8, enumeration has only 256 states and is inexpensive. This work does not establish industrial performance or publication priority.')
p('The package was developed through an AI-assisted research session. Original numerical records are preserved separately from local packaging checks. No author name or affiliation has been inferred.','small')
page('Why the positive representation is stable')
p('Expand the product of cosh factors in the Hubbard-Stratonovich integrand. Each spin configuration contributes a Gaussian centered at h/N, where h = transpose(W) times the spin vector. Integer weights imply that every center has the same coordinate parity.')
p('If p is the vector of column-sum parities, use lattice points x = (p + 2k)/N. Translating any completed Gaussian to its center shifts k by an integer vector. Every spin configuration therefore has exactly the same lattice normalization G to the power r, where G is the sum over integer j of exp(-2 beta j squared / N).')
p('<b>Identity.</b> Z equals the sum of the original integrand over this lattice, divided by G to the power r. The proof expands spin states; the numerical lattice evaluator does not.')
p('<b>Conditioning.</b> Every term is positive, so sign cancellation in Z disappears. Log-domain evaluation and stable variance accumulation address additional numerical hazards. Positivity alone does not remove floating-point error or guarantee a specified final accuracy.')
p('<b>Truncation.</b> The normalized lattice is a mixture of discrete Gaussians with centers in known intervals. Coordinate tail bounds give an omitted-mass bound delta and a log-normalization bound -log(1-delta), for delta below one. The implementation evaluates analytic bounds in ordinary floating point; it is not an outward-rounded certificate.')
p('<b>Field derivatives.</b> Means and susceptibility can be expressed as positive lattice expectations of tanh and sech-squared factors. A stable variance decomposition avoids subtracting two large raw second moments. This does not justify differentiating the integer-weight identity with respect to arbitrary real weights.')
p('<b>Scaling.</b> The present evaluator visits a Cartesian product of lattice points. Its cost remains exponential in the number of auxiliary coordinates. Adaptive region bounds and tensor compression are research directions, not completed capabilities.')
p('Detailed proof: docs/mathematics.md. Explicit tail and derivative formulas: experiments/positive-lattice/README.md. Classical theta definitions and transformations: https://dlmf.nist.gov/20.2 and https://dlmf.nist.gov/20.7.','small')
page('Recorded precision-runtime comparison')
x=json.loads((R/'results/recorded/high-precision/results.json').read_text())['rows']
rows=[]
for a in x:
 q=a['positive']['median_seconds'];g=a['gh']['median_seconds'];rows.append([a['tolerance'],f'{q:.5f}',f'{g:.5f}',f'{g/q:.2f}x'])
table(['Error target','Positive (s)','Gauss-Hermite (s)','Time ratio'],rows,[100,120,155,120])
story.append(Image(str(R/'figures/precision-runtime.png'),width=495,height=266))
p('Same fixed N = 8, r = 5, beta = 2 model. Each time is the median of three evaluations after warming, using one CPU thread and matched native C++ evaluation structure. Working precision increases from binary64 to extended precision to binary128.','small')
p('Model-dependent preparation is included. Compilation, node generation, reference computation and setting selection are excluded. The independent reference was checked at 100 and 140 digits. Original binary64 fields are interpreted as exact rationals.','small')
p('At 10^-30, the lattice has 25.11 million points versus 102.40 million quadrature points. Errors were approximately 8.8e-34 and 3.0e-31. The advantage is principally fewer points. A new local smoke check validates the 10^-9 settings; this package does not present a new full timing sweep.','small')
page('Critical behavior and next research steps')
p('Three bipolar patterns with equal pairwise correlation c have collective Gram eigenvalue 1 + 2c. For fixed positive c, the collective critical inverse temperature is 1/(1 + 2c). At c = 1/3, this is 0.6.')
p('The predicted susceptibility is 0.9142635791469034 times sqrt(N), minus 0.3444428364160962, with an O(N^-1/2) remainder. The coefficients come from the analytic quartic expansion and transverse-mode correction, not a fit.')
story.append(Image(str(R/'figures/critical-susceptibility.png'),width=495,height=266))
p('Grouped finite-state sums check the prediction for N from 12 to 384. The relative error of the corrected prediction decreases from about 1.77% to 0.0192%. At N = 12, high-precision residues also agree with an independent positive reference. This validates consistency with mean-field critical behavior; it is not a new universality claim.','small')
p('<b>Next steps.</b> Establish useful size/rank regimes; compare with the strongest structure-aware alternatives; investigate compression and broader weights; develop derivative accuracy control; and test a real inference workload where normalization is costly.')
p('<b>Review and reproduction.</b> Start with examples/quick_check.py, then scripts/check_native.py. Full benchmark and critical calculations are documented in docs/reproduction.md. Historical snapshots are in results/recorded; the packaging validation record is docs/validation.md.','small')
def footer(c,d):
 c.setStrokeColor(TEAL);c.line(50,42,545,42);c.setFont('Helvetica',8);c.setFillColor(INK);c.drawString(50,28,'HOPFIELD PARTITION FUNCTIONS | RESEARCH PROTOTYPE');c.drawRightString(545,28,str(d.page))
SimpleDocTemplate(str(R/'docs/scientific-brief.pdf'),pagesize=(595.28,841.89),rightMargin=50,leftMargin=50,topMargin=47,bottomMargin=55,title='Hopfield partition functions: scientific brief',author='').build(story,onFirstPage=footer,onLaterPages=footer)
print('Scientific brief built.')
