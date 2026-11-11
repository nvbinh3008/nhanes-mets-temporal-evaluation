from phaseb_common import *
from scipy.special import expit,logit
from scipy.optimize import brentq
import itertools,time
B=2000;SEED=20261108
NAMES=['prevalence','mean_prediction','log_loss','brier','average_precision','roc_auc','calibration_intercept','calibration_slope','calibration_in_the_large','observed_expected_ratio','ece','sensitivity_at_frozen_threshold','specificity_at_frozen_threshold','sensitivity_at_0_5','specificity_at_0_5']
class Metric:
 def __init__(self,y,p,threshold):
  self.y=np.asarray(y,float);self.p=np.clip(np.asarray(p,float),1e-6,1-1e-6);self.x=logit(self.p);self.t=threshold
  self.order=np.argsort(self.p,kind='stable');self.ends=np.r_[np.flatnonzero(np.diff(self.p[self.order])!=0),len(self.p)-1];self.loss=-self.y*np.log(self.p)-(1-self.y)*np.log1p(-self.p);self.bs=(self.y-self.p)**2
  self.quantile_order=np.argsort(self.p)
 def __call__(self,w):
  w=np.asarray(w,float);sw=w.sum();y=self.y;p=self.p
  if sw<=0 or w@y<=0 or w@(1-y)<=0:return np.full(len(NAMES),np.nan)
  q=w@y/sw;pm=w@p/sw;wp=w[self.order]*y[self.order];wn=w[self.order]*(1-y[self.order]);cp=np.cumsum(wp)[self.ends];cn=np.cumsum(wn)[self.ends];pos=np.diff(np.r_[0,cp]);neg=np.diff(np.r_[0,cn]);auc=np.sum(pos*(cn-neg*.5))/(cp[-1]*cn[-1])
  # AP = sum recall increments * precision at descending score tie groups.
  descending_pos=pos[::-1];descending_total=(pos+neg)[::-1];denom=np.cumsum(descending_total);prec=np.divide(np.cumsum(descending_pos),denom,out=np.zeros_like(denom),where=denom>0);ap=np.sum(descending_pos/cp[-1]*prec)
  a=0.;b=1.;norm=w/sw
  for _ in range(40):
   fit=expit(a+b*self.x);r=norm*(y-fit);v=norm*fit*(1-fit);g=np.array([r.sum(),r@self.x]);H=np.array([[v.sum(),v@self.x],[v@self.x,v@(self.x*self.x)]])
   try:step=np.linalg.solve(H,g)
   except np.linalg.LinAlgError:a=b=np.nan;break
   step=np.clip(step,-2,2);a+=step[0];b+=step[1]
   if np.max(np.abs(step))<1e-9:break
  residual=norm*(y-expit(a+b*self.x))
  if max(abs(residual.sum()),abs(residual@self.x))>1e-7:a=b=np.nan
  alpha=0.
  for _ in range(30):
   fit=expit(self.x+alpha);g=norm@(y-fit);h=norm@(fit*(1-fit))
   if h<=1e-12:alpha=np.nan;break
   step=np.clip(g/h,-2,2);alpha+=step
   if abs(step)<1e-10:break
  good=w[self.quantile_order]>0;ws=w[self.quantile_order][good];ps=p[self.quantile_order][good];edges=np.interp(np.linspace(0,1,11),(np.cumsum(ws)-.5*ws)/sw,ps);edges[0]=0;edges[-1]=1;edges=np.maximum.accumulate(edges);bins=np.searchsorted(edges[1:-1],p,side='right');ece=np.abs(np.bincount(bins,weights=w*(y-p),minlength=10)).sum()/sw
  def sens(t):return np.array([w@(y*(p>=t))/(w@y),w@((1-y)*(p<t))/(w@(1-y))])
  return np.r_[q,pm,w@self.loss/sw,w@self.bs/sw,ap,auc,a,b,alpha,q/pm,ece,sens(self.t),sens(.5)]
def design(d):
 keys=d[['stratum','psu']].astype(str);psu_codes,uni=pd.factorize(pd.MultiIndex.from_frame(keys));groups=[]
 for s,g in pd.DataFrame(list(uni),columns=['s','p']).groupby('s',sort=True):groups.append(g.index.to_numpy())
 if any(len(g)<2 for g in groups):raise RuntimeError('Singleton design stratum: explicit failure')
 return psu_codes,groups
def replicate_weights(d,B=B):
 codes,groups=design(d);rng=np.random.default_rng(SEED);base=d.analysis_weight.to_numpy(float)
 for r in range(B):
  factors=np.zeros(max(codes)+1)
  for g in groups:
   n=len(g);counts=np.bincount(rng.integers(0,n,n-1),minlength=n);factors[g]=n/(n-1)*counts
  yield base*factors[codes]
def evaluate(d,mask=None):
 if mask is None:mask=np.ones(len(d),bool)
 sub=d.loc[mask];th=float(sub.frozen_threshold.iloc[0]) if 'frozen_threshold' in sub else .5;m=Metric(sub.outcome,sub.calibrated_probability,th);point=m(sub.analysis_weight);samples=np.empty((B,len(NAMES)))
 for i,w in enumerate(replicate_weights(d)):samples[i]=m(w[mask])
 return point,samples
def rows(meta,point,samples):
 out=[]
 for j,k in enumerate(NAMES):
  valid=samples[:,j][np.isfinite(samples[:,j])];lo,hi=np.quantile(valid,[.025,.975]) if len(valid)>=.95*B else [np.nan,np.nan]
  out.append(dict(**meta,metric=k,estimate=point[j],ci_lower=lo,ci_upper=hi,standard_error=np.std(valid,ddof=1),valid=len(valid),planned=B,seed=SEED,method='Rao-Wu n_h-1, ultimate-cluster WR; fixed predictions; domain weights retain full design'))
 return out
def run():
 R,O,PRIVATE=execution_paths()
 output=[];contrasts=[];paired=[];subcontrast=[]
 cache=PRIVATE/'metric_replicates';cache.mkdir(exist_ok=True)
 # Check fast metrics against library definitions, including ties, before production.
 from sklearn.metrics import roc_auc_score,average_precision_score
 from model_pipeline.metrics import expected_calibration_error,calibration_intercept_slope
 y=np.array([1,0,1,0,1]);p=np.array([.2,.2,.5,.8,.8]);w=np.array([2,3,1,2,4]);m=Metric(y,p,.5)(w);assert abs(m[5]-roc_auc_score(y,p,sample_weight=w))<1e-12;assert abs(m[4]-average_precision_score(y,p,sample_weight=w))<1e-12;assert abs(m[10]-expected_calibration_error(y,p,w,10))<1e-12
 for v in ['day1_primary','two_day_average']:
  temp=preds(v);prim=primary(temp);dev=primary(preds(v,'development'));dev['frozen_threshold']=prim.frozen_threshold.iloc[0]
  for tag,d in [('development',dev),('temporal',prim)]:
   pt,sp=evaluate(d);np.savez(cache/f'{v}_{tag}.npz',point=pt,replicates=sp);output+=rows(dict(view=v,scope=tag,representation='R5',algorithm='boost_monotonic',group_variable='all',group='all',n=len(d),suppressed=False),pt,sp)
  # Independent periods use independent streams: development uses seed+1 for contrast.
  global SEED
  SEED+=1;dp,ds=evaluate(dev);SEED-=1;tp,ts=evaluate(prim)
  contrasts+=rows(dict(view=v,scope='temporal_minus_development',representation='R5',algorithm='boost_monotonic',group_variable='all',group='all',n=len(prim),suppressed=False),tp-dp,ts-ds)
  for cyc,g in prim.groupby('cycle_id'):
   g=g.reset_index(drop=True);pt,sp=evaluate(g);output+=rows(dict(view=v,scope=cyc,representation='R5',algorithm='boost_monotonic',group_variable='all',group='all',n=len(g),suppressed=False),pt,sp)
  rep=pd.read_parquet(R/f'data/hei2020/representations/temporal_test/r5_{v}.parquet');extra=pd.concat([pd.read_parquet(R/f'data/processed/{c}/person_level_derived.parquet') for c in ['2015_2016','2017_2018']]);extra=extra[['cycle_id','seqn','race_ethnicity_expanded']];grouped=prim.merge(rep[['cycle_id','seqn','sex','race_ethnicity_legacy','age_years']],on=['cycle_id','seqn'],validate='one_to_one').merge(extra,on=['cycle_id','seqn'],validate='one_to_one');grouped['age_band']=pd.cut(grouped.age_years,[19,39,59,79,np.inf],labels=['20-39','40-59','60-79','80+']).astype(str)
  names={'sex':{1:'Male',2:'Female'},'race_ethnicity_legacy':{1:'Mexican American',2:'Other Hispanic',3:'Non-Hispanic White',4:'Non-Hispanic Black',5:'Other/multiracial'},'race_ethnicity_expanded':{1:'Mexican American',2:'Other Hispanic',3:'Non-Hispanic White',4:'Non-Hispanic Black',6:'Non-Hispanic Asian',7:'Other/multiracial'}}
  for var in ['sex','race_ethnicity_legacy','age_band','race_ethnicity_expanded']:
   collected={}
   for level,g in grouped.groupby(var,observed=True):
    label=names.get(var,{}).get(level,str(level));mask=grouped[var].eq(level).to_numpy();cases=g.outcome.sum();psus=g.psu.nunique();suppress=len(g)<300 or cases<100 or psus<30
    if suppress:
     output.append(dict(view=v,scope='temporal_subgroup',representation='R5',algorithm='boost_monotonic',group_variable=var,group=label,n=len(g),cases=cases,psus=psus,suppressed=True,metric='SUPPRESSED',estimate=np.nan,valid=0,planned=B));continue
    pt,sp=evaluate(grouped,mask);output+=rows(dict(view=v,scope='temporal_subgroup',representation='R5',algorithm='boost_monotonic',group_variable=var,group=label,n=len(g),cases=cases,psus=psus,suppressed=False),pt,sp);collected[label]=(pt,sp)
   # Contrasts share replicate weights across domains; no non-overlap-of-CIs inference.
   for a,b in itertools.combinations(collected,2):
    pa,sa=collected[a];pb,sb=collected[b];subcontrast+=rows(dict(view=v,scope='direct_subgroup_difference',group_variable=var,group=a+' minus '+b,n=len(grouped),suppressed=False),pa-pb,sa-sb)
  # All pooled temporal representation/algorithm comparators use original outcomes.
  for (representation,algorithm),g in temp.groupby(['representation','algorithm']):
   if representation=='R5' and algorithm=='boost_monotonic':continue
   g=g.reset_index(drop=True);pt,sp=evaluate(g);output+=rows(dict(view=v,scope='temporal',representation=representation,algorithm=algorithm,group_variable='all',group='all',n=len(g),suppressed=False),pt,sp)
  # Label sensitivity, keep original cohort and original probabilities.
  from nhanes_etl.derive import derive_cycle_variables
  labs=[]
  for c in ['2015_2016','2017_2018']:
   h=pd.read_parquet(R/f'data/processed/{c}/person_level_harmonized.parquet');h['fasting_glucose_mg_dl']=.9776*h.fasting_glucose_mg_dl+.4994;h=derive_cycle_variables(h);labs.append(h[['cycle_id','seqn','mets_prevalent']])
  sens=prim.merge(pd.concat(labs),on=['cycle_id','seqn'],validate='one_to_one');sens['outcome']=sens.mets_prevalent.astype(float);pt,sp=evaluate(sens);output+=rows(dict(view=v,scope='backward_glucose_sensitivity',representation='R5',algorithm='boost_monotonic',group_variable='all',group='all',n=len(sens),suppressed=False),pt,sp)
  csv(output+contrasts,'CORRECTED_ALL_METRICS_AND_CI.csv');csv(subcontrast,'DIRECT_SUBGROUP_CONTRASTS.csv');print(now(),'DONE METRICS',v,flush=True)
 # Paired sample keys sorted, one shared PSU draw per replicate.
 a=primary(preds('day1_primary'));b=primary(preds('two_day_average'));d=a.merge(b[['cycle_id','seqn','calibrated_probability','frozen_threshold']],on=['cycle_id','seqn'],suffixes=('','_two'),validate='one_to_one').sort_values(['cycle_id','seqn']).reset_index(drop=True);pt,sp=evaluate(d);paired+=rows(dict(view='day1_on_completers',scope='paired_sample',n=len(d)),pt,sp);two=d.copy();two['calibrated_probability']=d.calibrated_probability_two;two['frozen_threshold']=d.frozen_threshold_two;pt2,sp2=evaluate(two);paired+=rows(dict(view='two_day_on_completers',scope='paired_sample',n=len(d)),pt2,sp2);paired+=rows(dict(view='two_day_minus_day1_same_completers',scope='paired_difference',n=len(d)),pt2-pt,sp2-sp);csv(paired,'TWO_DAY_PAIRED_RESULTS.csv')
 logmeta('METRICS_EXECUTION.json',dict(replicates=B,seed=SEED,calibration='Newton weighted logistic intercept/slope; offset logistic CITL; no development refitting',metrics=NAMES,fast_metric_tests='AUC/AP/ECE exact tie-fixture checks passed',checkpoint='Both views and paired complete'))
if __name__=='__main__':run()
