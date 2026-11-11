from phaseb_common import *
import yaml,dataclasses,joblib
from model_pipeline.data import load_representation
from model_pipeline.nested import nested_validate_one,fit_final_one,stable_seed
from threadpoolctl import threadpool_limits


def main():
 R,O,PRIVATE=execution_paths()
 threadpool_limits(1)
 config=yaml.safe_load((R/'config/modeling.yaml').read_text());config['_project_root']=str(R)
 # Use the same R5 split/task identity, candidates and calibration protocol, removing diet only.
 cov=['age_years','sex','race_ethnicity_legacy','education_adult','family_poverty_income_ratio','smoking_status','ltpa_mvpa_equivalent_min_week','alcohol_g_day1']
 records=[]
 for v in ['day1_primary','two_day_average']:
  dev=load_representation(config,'development','R5',v);tmp=load_representation(config,'temporal_test','R5',v)
  dev=dataclasses.replace(dev,features=cov,categorical_features=[c for c in dev.categorical_features if c in cov])
  for alg in ['boost_monotonic','logistic']:
   stamp=PRIVATE/f'covariate_{v}_{alg}'
   if stamp.with_suffix('.done.json').exists():print('SKIP completed',stamp,flush=True);continue
   print(now(),'START',v,alg,flush=True)
   oof,search=nested_validate_one(dev,alg,config);oof.to_parquet(stamp.with_suffix('.oof.parquet'));search.to_csv(O/f'COVARIATE_NESTED_SEARCH_{v}_{alg}.csv',index=False)
   model,finalsearch,internal=fit_final_one(dev,alg,config)
   # Fitting/selection above receives DEVELOPMENT ONLY; temporal outcomes are first accessed in evaluation below.
   prob=model.predict_proba(tmp.frame[cov]);d=tmp.frame[['cycle_id','seqn',tmp.target_column,tmp.weight_column,tmp.stratum_column,tmp.psu_column]].copy();d.columns=['cycle_id','seqn','outcome','analysis_weight','stratum','psu'];d['calibrated_probability']=prob;d['frozen_threshold']=model.threshold;d['algorithm']=alg;d['view']=v
   d.to_parquet(stamp.with_suffix('.temporal.parquet'));joblib.dump(model,stamp.with_suffix('.joblib'))
   finalsearch.to_csv(O/f'COVARIATE_FINAL_SEARCH_{v}_{alg}.csv',index=False)
   payload=dict(view=v,algorithm=alg,classification='New post-review exploratory; development-only nested selection',features=model.features,expanded=model.transformed_feature_names,constraints=model.transformed_constraints,calibration=model.calibrator.method,params=model.selected_params,metadata=model.metadata,threshold=model.threshold,nested_predictions_sha256=sha(stamp.with_suffix('.oof.parquet')),temporal_predictions_sha256=sha(stamp.with_suffix('.temporal.parquet')),completed_utc=now())
   stamp.with_suffix('.done.json').write_text(json.dumps(payload,indent=2,default=str));(O/f'COVARIATE_MODEL_SPEC_{v}_{alg}.json').write_text(json.dumps(payload,indent=2,default=str));print(now(),'DONE',v,alg,flush=True)


if __name__ == "__main__":
 main()
