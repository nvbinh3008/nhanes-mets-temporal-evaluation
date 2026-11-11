from phaseb_common import *
from phaseb_metrics import replicate_weights
from scipy.special import expit
from sklearn.preprocessing import SplineTransformer
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor,Color


def main():
 R,O,PRIVATE=execution_paths()
 def fit(X,y,w,start=None):
  norm=w/w.sum();beta=np.zeros(X.shape[1]) if start is None else start.copy();penalty=np.eye(len(beta))*1e-6;penalty[0,0]=0
  for _ in range(50):
   p=expit(X@beta);g=X.T@(norm*(y-p))-penalty@beta;H=X.T@((norm*p*(1-p))[:,None]*X)+penalty+np.eye(len(beta))*1e-10
   try:step=np.linalg.solve(H,g)
   except np.linalg.LinAlgError:return None
   step=np.clip(step,-2,2);beta+=step
   if np.max(np.abs(step))<1e-7:return beta
  return None
 pdf=canvas.Canvas(str(O/'CALIBRATION_PLOT_WITH_CI.pdf'),pagesize=(595,790));allrows=[];details=[]
 for v in ['day1_primary','two_day_average']:
  d=primary(preds(v));p=d.calibrated_probability.to_numpy(float);y=d.outcome.to_numpy(float);w=d.analysis_weight.to_numpy(float)
  # Fixed unsupervised knots, without temporal outcome tuning. Fit on individual observations.
  order=np.argsort(p);positions=(np.cumsum(w[order])-.5*w[order])/w.sum();knots=np.interp(np.linspace(0,1,4),positions,p[order]);spline=SplineTransformer(degree=3,knots=knots[:,None],include_bias=False,extrapolation='constant');X=np.c_[np.ones(len(p)),spline.fit_transform(p[:,None])];grid=np.linspace(np.interp(.02,positions,p[order]),np.interp(.98,positions,p[order]),151);G=np.c_[np.ones(len(grid)),spline.transform(grid[:,None])];beta=fit(X,y,w);assert beta is not None;point=expit(G@beta);curves=[]
  for ww in replicate_weights(d):
   b=fit(X,y,ww,start=beta)
   if b is not None:curves.append(expit(G@b))
  assert len(curves)>=1900
  lo,hi=np.quantile(curves,[.025,.975],axis=0)
  for g,a,b,c in zip(grid,point,lo,hi):allrows.append(dict(view=v,predicted_probability=g,observed_smooth=a,ci_lower=b,ci_upper=c,valid=len(curves),planned=2000))
  details.append(dict(view=v,knots=knots.tolist(),coefficients=beta.tolist(),valid=len(curves),planned=2000,method='Unbinned weighted logistic cubic B-spline, 4 fixed weighted-quantile knots; tiny slope ridge 1e-6; 2000 Rao-Wu PSU replicates; pointwise 95% percentile band; 2nd-98th weighted percentile display; conditional on original labels, frozen predictions and chosen smoother'))
  # Vector scientific figure, one view per page.
  pdf.setFillColor(HexColor('#172c40'));pdf.setFont('Helvetica-Bold',17);pdf.drawString(55,747,'Survey-weighted calibration')
  pdf.setFont('Helvetica',11);pdf.drawString(55,726,'Day 1 primary' if v=='day1_primary' else 'Two-day sensitivity view');pdf.setFont('Helvetica',9);pdf.drawString(55,710,f'NHANES 2015-2018 | n = {len(d):,} | original outcome labels')
  left,bottom,width,height=75,325,450,330
  def xx(x):return left+x*width
  def yy(y):return bottom+y*height
  pdf.setStrokeColor(HexColor('#ccd4db'));pdf.setLineWidth(.5)
  for z in np.linspace(0,1,6):pdf.line(xx(z),bottom,xx(z),bottom+height);pdf.line(left,yy(z),left+width,yy(z));pdf.setFillColor(HexColor('#35495a'));pdf.setFont('Helvetica',8);pdf.drawCentredString(xx(z),bottom-15,f'{z:.1f}');pdf.drawRightString(left-8,yy(z)-3,f'{z:.1f}')
  pdf.setStrokeColor(HexColor('#7f8991'));pdf.setDash(4,3);pdf.line(xx(0),yy(0),xx(1),yy(1));pdf.setDash()
  path=pdf.beginPath();path.moveTo(xx(grid[0]),yy(lo[0]))
  for a,b in zip(grid[1:],lo[1:]):path.lineTo(xx(a),yy(b))
  for a,b in zip(grid[::-1],hi[::-1]):path.lineTo(xx(a),yy(b))
  path.close();pdf.setFillColor(HexColor('#d4e9f3'));pdf.drawPath(path,stroke=0,fill=1)
  path=pdf.beginPath();path.moveTo(xx(grid[0]),yy(point[0]))
  for a,b in zip(grid[1:],point[1:]):path.lineTo(xx(a),yy(b))
  pdf.setStrokeColor(HexColor('#086b98'));pdf.setLineWidth(2);pdf.drawPath(path)
  pdf.setFillColor(HexColor('#172c40'));pdf.setFont('Helvetica',10);pdf.drawCentredString(300,288,'Frozen predicted probability');pdf.saveState();pdf.translate(29,490);pdf.rotate(90);pdf.drawCentredString(0,0,'Estimated observed probability');pdf.restoreState()
  hist,edges=np.histogram(p,bins=np.linspace(0,1,21),weights=w);hist=hist/w.sum();hb=134;hh=105;mx=max(hist)*1.1
  pdf.setFont('Helvetica-Bold',10);pdf.drawString(75,259,'Weighted distribution of frozen predictions')
  for a,b,z in zip(edges[:-1],edges[1:],hist):pdf.setFillColor(HexColor('#769eaf'));pdf.rect(xx(a)+1,hb,(b-a)*width-2,z/mx*hh,stroke=0,fill=1)
  pdf.setFillColor(HexColor('#35495a'));pdf.setFont('Helvetica',8)
  for z in np.linspace(0,1,6):pdf.drawCentredString(xx(z),hb-13,f'{z:.1f}')
  pdf.drawString(75,105,'Each bar shows a survey-weighted probability mass; bins have width 0.05.')
  notes=['Blue: weighted logistic cubic B-spline; pale blue: pointwise 95% PSU bootstrap band.',f'Valid replicates: {len(curves)}/2,000. Dashed diagonal: ideal calibration.', 'Curve shown over the 2nd-98th weighted percentiles; full histogram shown below.', 'Diagnostic fitted using temporal outcomes. No frozen model probabilities were updated.', 'Bands condition on fixed predictions, original labels and fixed smoothing specification.']
  for i,line in enumerate(notes):pdf.drawString(55,83-i*12,line)
  pdf.showPage();print(now(),'CALIBRATION DONE',v,flush=True)
 pdf.save();csv(allrows,'CALIBRATION_SMOOTH_CURVE_WITH_CI.csv');logmeta('CALIBRATION_METHOD.json',dict(views=details))
 import pypdfium2 as pdfium
 document=pdfium.PdfDocument(str(O/'CALIBRATION_PLOT_WITH_CI.pdf'))
 for i,p in enumerate(document):p.render(scale=1.5).to_pil().save(PRIVATE/f'calibration_page_{i+1}.png')


if __name__ == "__main__":
 main()
