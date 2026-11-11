from phaseb_common import *
from phaseb_metrics import Metric,evaluate,rows,NAMES,replicate_weights


def main():
 R,O,PRIVATE=execution_paths()
 out=[];checks=[];response=[]
 for v in ['day1_primary','two_day_average']:
  original=primary(preds(v)).sort_values(['cycle_id','seqn']).reset_index(drop=True)
  for alg in ['boost_monotonic','logistic']:
   path=PRIVATE/f'covariate_{v}_{alg}.temporal.parquet';cov=pd.read_parquet(path);d=original.merge(cov[['cycle_id','seqn','outcome','calibrated_probability','frozen_threshold']],on=['cycle_id','seqn'],suffixes=('','_cov'),validate='one_to_one').sort_values(['cycle_id','seqn']).reset_index(drop=True);assert len(d)==len(original) and np.array_equal(d.outcome,d.outcome_cov)
   a=d.copy();a['calibrated_probability']=d.calibrated_probability_cov;a['frozen_threshold']=d.frozen_threshold_cov;pa,sa=evaluate(a);pb,sb=evaluate(d)
   out+=rows(dict(view=v,algorithm=alg,contrast='covariate_only_absolute_temporal',n=len(d),classification='Post-review exploratory; development-only nested tuning'),pa,sa)
   out+=rows(dict(view=v,algorithm=alg,contrast='covariate_only_minus_original_R5_monotonic',n=len(d),classification='Paired fixed-prediction temporal contrast; positive loss difference favors original R5'),pa-pb,sa-sb)
   oof=pd.read_parquet(PRIVATE/f'covariate_{v}_{alg}.oof.parquet');m=Metric(oof.outcome,oof.calibrated_probability,.5);point=m(oof.analysis_weight)
   for k,value in zip(NAMES,point):out.append(dict(view=v,algorithm=alg,contrast='covariate_only_nested_development_OOF',n=len(oof),metric=k,estimate=value,ci_lower=np.nan,ci_upper=np.nan,classification='OOF point estimates; uncertainty not refit-propagated'))
   checks.append(dict(view=v,algorithm=alg,records=len(d),outcome_match=True,temporal_outcome_used_for_selection=False))
  # Paired comparison with original frozen weighted logistic regression baseline.
  temp=preds(v);lr=temp[temp.representation.eq('R5')&temp.algorithm.eq('logistic')].sort_values(['cycle_id','seqn']).reset_index(drop=True);assert np.array_equal(lr.seqn,original.seqn);pa,sa=evaluate(lr);pb,sb=evaluate(original)
  out+=rows(dict(view=v,algorithm='original_R5_logistic',contrast='original_R5_logistic_minus_original_R5_monotonic',n=len(lr),classification='Original frozen comparator; corrected paired uncertainty'),pa-pb,sa-sb)
  # Selection weight sensitivities on the original sample, no added participants.
  for stage,tag in [('day1_plus_endpoint','primary_response_IPW'),('day2_given_primary','day2_response_IPW')]:
   if stage=='day2_given_primary' and v!='two_day_average':continue
   w=pd.read_parquet(PRIVATE/f'response_weights_temporal_test_{stage}.parquet');d=original.merge(w[['cycle_id','seqn','sensitivity_weight_2yr','response']],on=['cycle_id','seqn'],validate='one_to_one');assert d.response.eq(1).all();d['analysis_weight']=d.sensitivity_weight_2yr/2;pt,sp=evaluate(d.reset_index(drop=True));response+=rows(dict(view=v,scope=tag,n=len(d),classification='Post-review MAR-dependent IPW sensitivity; propensity estimates fixed in bootstrap'),pt,sp)
 csv(out,'COVARIATE_ONLY_EXPLORATORY_RESULTS.csv');csv(response,'RESPONSE_WEIGHT_SENSITIVITY_RESULTS.csv');logmeta('COMPARISON_VALIDATION.json',dict(checks=checks,bootstrap='2000 Rao-Wu shared PSU draws per paired comparison; seed 20261108',outcome='Original labels only',private_predictions_not_released=True));print(now(),'COMPARISONS COMPLETED')


if __name__ == "__main__":
 main()
