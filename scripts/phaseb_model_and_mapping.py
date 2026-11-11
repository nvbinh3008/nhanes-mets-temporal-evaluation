from phaseb_common import *
from nhanes_etl.derive import derive_cycle_variables
from model_pipeline.nested import stable_seed
from model_pipeline.utils import sha256_tree
from phaseb_metrics import evaluate,rows
import joblib,warnings,importlib.metadata


def main():
 R,O,PRIVATE=execution_paths()
 checks=[];expanded=[];imputers=[];seeds=[]
 for v in ['day1_primary','two_day_average']:
  bundle=R/f'models/survey_aware/{v}/r5/boost_monotonic/model.joblib';spec=json.loads((R/f'reports/modeling/frozen/{v}/MODEL_SPEC_FROZEN.json').read_text());reg=[r for r in spec['registry'] if r['representation']=='R5' and r['algorithm']=='boost_monotonic'][0];assert sha(bundle)==reg['bundle_sha256']
  with warnings.catch_warnings(record=True) as ww:model=joblib.load(bundle)
  rep=pd.read_parquet(R/f'data/hei2020/representations/temporal_test/r5_{v}.parquet');pr=primary(preds(v));d=rep.merge(pr[['cycle_id','seqn','calibrated_probability']],on=['cycle_id','seqn'],validate='one_to_one');p=model.predict_proba(d[model.features]);assert np.array_equal(p,d.calibrated_probability.to_numpy())
  source=sha256_tree(R/'model_pipeline');checks.append(dict(view=v,bundle_sha256=sha(bundle),bundle_hash_match=True,development_file_hash_match=sha(R/f'data/hei2020/representations/development/r5_{v}.parquet')==reg['development_sha256'],model_source_tree_sha256=source,source_tree_matches_frozen=source==spec['model_pipeline_source_sha256'],n=len(d),probabilities_exactly_equal=True,max_absolute_difference=0.,fitting_primary_performed=False,loading_warnings=[str(w.message) for w in ww]))
  logmeta(f'MODEL_OBJECT_METADATA_{v}.json',dict(features=model.features,expanded=model.transformed_feature_names,constraints=model.transformed_constraints,metadata=model.metadata,params=model.selected_params,calibration=model.calibrator.method,threshold=model.threshold,max_absolute_prediction_difference=0.,warnings=[str(w.message) for w in ww]))
  for i,(name,direction) in enumerate(zip(model.transformed_feature_names,model.transformed_constraints)):expanded.append(dict(view=v,index_zero_based=i,expanded_feature=name,constraint=direction))
  for name,trans,columns in model.preprocessor.transformers_:
   if name=='remainder':continue
   imp=trans.named_steps['imputer']
   for c,value in zip(columns,imp.statistics_):imputers.append(dict(view=v,block=name,feature=c,strategy=imp.strategy,fitted_value=str(value)))
   if name=='cat':
    for c,categories in zip(columns,trans.named_steps['onehot'].categories_):imputers.append(dict(view=v,block='onehot',feature=c,strategy='fitted category order',fitted_value=json.dumps(categories.tolist(),default=str)))
  # Enumerate every seed formula used for R5 final and outer/inner/calibration fold tasks.
  for alg in ['boost_monotonic','logistic','boost_unconstrained']:
   for phase,base in [('nested',stable_seed(20260715,'R5',v,'nested_split')),('final',stable_seed(20260715,'final','R5',v))]:
    seeds.append(dict(view=v,algorithm=alg,phase=phase,task='split_seed',seed=base))
    searches=[('full',base)] if phase=='final' else [(f'outer_{j}',stable_seed(base,'outer',j)) for j in range(5)]
    for label,s in searches:
     seeds.append(dict(view=v,algorithm=alg,phase=phase,task=label+'/inner_split',seed=stable_seed(s,'inner_split')))
     for k in range(5 if alg=='logistic' else 6):
      for fold in range(4):seeds.append(dict(view=v,algorithm=alg,phase=phase,task=f'{label}/candidate_{k}/inner_fit_{fold}',seed=stable_seed(s,alg,k,fold)))
      for method in ['none','sigmoid','isotonic']:
       for selected in [False,True]:seeds.append(dict(view=v,algorithm=alg,phase=phase,task=f'{label}/candidate_{k}/{"selected_calibration" if selected else "calibration"}/{method}',seed=stable_seed(s,'selected_calibration' if selected else 'calibration',k,method)))
    if phase=='final':seeds.append(dict(view=v,algorithm=alg,phase=phase,task='refit',seed=stable_seed(base,'refit',alg)))
    else:
     for j in range(5):seeds.append(dict(view=v,algorithm=alg,phase=phase,task=f'outer_{j}/refit',seed=stable_seed(base,'outer_refit',j,alg)))
 csv(expanded,'FITTED_EXPANDED_FEATURE_ORDER.csv');csv(imputers,'FITTED_IMPUTATION_AND_CATEGORIES.csv');csv(seeds,'DERIVED_SEED_REGISTRY.csv');logmeta('MODEL_RECONSTRUCTION_CHECKS.json',dict(checks=checks,packages={n:importlib.metadata.version(n) for n in ['numpy','pandas','pyarrow','scikit-learn','scipy','joblib']},python=sys.version))
 # Verify the documented DID070 mapping omission and quantify its actual consequences.
 raw=R/'data/raw/2007_2008/DIQ_E.XPT';rx=pd.read_sas(raw,format='xport')[['SEQN','DID070']];h=pd.read_parquet(R/'data/processed/2007_2008/person_level_harmonized.parquet');old=derive_cycle_variables(h);fixed=h.merge(rx,left_on='seqn',right_on='SEQN',how='left',validate='one_to_one');fixed['current_diabetes_pills']=fixed.DID070.where(fixed.DID070.isin([1,2]));new=derive_cycle_variables(fixed)
 base=old.age_years.ge(20)&old.mec_exam_status.eq(2)&~(old.sex.eq(2)&old.pregnancy_status.eq(1))&old.day1_recall_status.eq(1)&old.energy_kcal_day1.gt(0)&old.fasting_weight_2yr.gt(0)&old.fasting_duration_hours.ge(8)&old.fasting_duration_hours.lt(24);retained=base&old.mets_prevalent.notna();r=[]
 for label,mask in [('all_harmonized',pd.Series(True,index=old.index)),('eligible_before_endpoint',base),('original_retained',retained)]:
  for c in ['glucose_lowering_treatment','mets_elevated_fasting_glucose','mets_prevalent']:
   both=mask&old[c].notna()&new[c].notna();r.append(dict(scope=label,variable=c,n=int(mask.sum()),changed_known_status=int((both&old[c].ne(new[c]).fillna(False)).sum()),became_classifiable=int((mask&old[c].isna()&new[c].notna()).sum()),became_missing=int((mask&old[c].notna()&new[c].isna()).sum()),raw_did070_yes=int((mask&fixed.DID070.eq(1)).sum())))
 csv(r,'DID070_MAPPING_OMISSION_IMPACT.csv');logmeta('DID070_EVIDENCE.json',dict(raw_path=str(raw),raw_sha256=sha(raw),source='https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/DIQ_E.htm',dictionary_claim='current_diabetes_pills unavailable in 2007-2008',verified='DID070 exists in actual raw XPT; official cross-reference confirms renamed variable',changed_original_retained_labels=int((retained&old.mets_prevalent.ne(new.mets_prevalent).fillna(False)).sum()),correction_applied_to_original_files=False))
 print(json.dumps(r,indent=2));print('MODEL AUDIT COMPLETED')


if __name__ == "__main__":
 main()
