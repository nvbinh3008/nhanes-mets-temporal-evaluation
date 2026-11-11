from phaseb_common import *
from nhanes_etl.config import Settings,DictionaryConfig
from nhanes_etl.medications import map_medications
from nhanes_etl.derive import derive_cycle_variables
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold


def main():
 R,O,PRIVATE=execution_paths()
 settings=Settings.from_yaml(R/'config/pipeline.yaml');dictionary=DictionaryConfig.load(settings.dictionary)
 components=['mets_abdominal_obesity','mets_elevated_triglycerides','mets_low_hdl','mets_elevated_blood_pressure','mets_elevated_fasting_glucose']
 prevalence=[];causes=[];selection=[];missing=[];med=[];bounds=[];allframes=[]
 for cyc in ['2007_2008','2009_2010','2011_2012','2013_2014','2015_2016','2017_2018']:
  d=pd.read_parquet(R/f'data/processed/{cyc}/person_level_derived.parquet');h=pd.read_parquet(R/f'data/processed/{cyc}/person_level_harmonized.parquet');cycle=cyc.replace('_','-');allframes.append(d)
  eligible=(d.age_years>=20)&d.mec_exam_status.eq(2)&~(d.sex.eq(2)&d.pregnancy_status.eq(1))&(d.fasting_weight_2yr>0)&d.fasting_duration_hours.ge(8)&d.fasting_duration_hours.lt(24)
  diet=d.day1_recall_status.eq(1)&d.energy_kcal_day1.gt(0);endpoint=d.mets_prevalent.notna();retained=eligible&diet&endpoint
  c=h.copy()
  if cyc>='2015_2016':c['fasting_glucose_mg_dl']=.9776*c.fasting_glucose_mg_dl+.4994
  sens=derive_cycle_variables(c)
  # Measurement-only phenotype sensitivity isolates treatment-defined positivity; it is NOT the original MetS definition.
  measured=pd.DataFrame(index=d.index)
  measured[components[0]]=d[components[0]];measured[components[1]]=(d.triglycerides_mg_dl>=150).where(d.triglycerides_mg_dl.notna());measured[components[2]]=d[components[2]];measured[components[3]]=(d.mean_sbp.ge(130)|d.mean_dbp.ge(85)).where(d.mean_sbp.notna()&d.mean_dbp.notna());measured[components[4]]=d.fasting_glucose_mg_dl.ge(100).where(d.fasting_glucose_mg_dl.notna());measured['mets_prevalent']=(measured[components].sum(axis=1)>=3).where(measured[components].notna().all(axis=1))
  for name,x in [('original',d),('backward_glucose',sens),('measurement_only',measured)]:
   for component in components+['mets_prevalent']:
    m=retained&x[component].notna();w=d.loc[m,'fasting_weight_2yr'];prevalence.append(dict(cycle=cycle,phenotype=name,component=component,n=int(m.sum()),weighted_prevalence=np.average(x.loc[m,component].astype(float),weights=w),cohort='Original complete-case cohort; missing measurement-only statuses excluded per component'))
   change=(d[components].astype(float)!=x[components].astype(float))&d[components].notna()&x[components].notna()
   for component in components:causes.append(dict(cycle=cycle,sensitivity=name,cause=component,changed_in_original_cohort=int((retained&change[component]).sum())))
   causes.append(dict(cycle=cycle,sensitivity=name,cause='MetS_label',changed_in_original_cohort=int((retained&d.mets_prevalent.ne(x.mets_prevalent).fillna(False)&x.mets_prevalent.notna()).sum())))
  before=eligible&diet;ex=before&~endpoint;positive=d[components].eq(1).sum(axis=1);unknown=d[components].isna().sum(axis=1);knownpos=positive.ge(3);knownneg=(positive+unknown).lt(3);amb=~knownpos&~knownneg;w=d.fasting_weight_2yr
  for scope,mask in [('before_complete_endpoint',before),('excluded_endpoint',ex)]:
   denom=w[mask].sum();bounds.append(dict(cycle=cycle,scope=scope,n=int(mask.sum()),determinable_positive=int((mask&knownpos).sum()),determinable_negative=int((mask&knownneg).sum()),ambiguous=int((mask&amb).sum()),weighted_prevalence_lower=w[mask&knownpos].sum()/denom,weighted_prevalence_upper=w[mask&~knownneg].sum()/denom))
  # Actual medication code and actual mapping file, not a similarly named ZIP snapshot.
  rx=pd.read_parquet(R/f'data/staged/{cyc}/rxq_rx.parquet');flags,matches=map_medications(rx,dictionary,settings,cycle);stored=pd.read_parquet(R/f'data/processed/{cyc}/medication_flags.parquet');mm=flags.merge(stored,on='seqn',suffixes=('_new','_stored'),validate='one_to_one')
  for flag in ['glucose_lowering_rx_flag','triglyceride_specific_rx_flag','antihypertensive_rx_flag','other_lipid_lowering_rx_flag']:
   bad=int((mm[flag+'_new']!=mm[flag+'_stored']).sum());med.append(dict(cycle=cycle,flag=flag,rows=len(mm),stored_rows=len(stored),mismatches=bad,positive=int(flags[flag].sum()),map_sha256=sha(settings.drug_id_map),dictionary_sha256=sha(settings.dictionary)))
  for stage,source,response in [('dietary_reliability',eligible,diet),('endpoint_complete',before,endpoint),('day2_complete',retained,d.day2_recall_status.eq(1)&d.energy_kcal_day2.gt(0))]:
   for group,mask in [('retained',source&response),('excluded',source&~response)]:
    for var in ['age_years','sex','race_ethnicity_legacy','education_adult','family_poverty_income_ratio','smoking_status','ltpa_mvpa_equivalent_min_week','alcohol_g_day1']:
     good=mask&d[var].notna();ww=w[good]
     if var not in ['sex','race_ethnicity_legacy','education_adult','smoking_status']:
      x=d.loc[good,var].astype(float);mean=np.average(x,weights=ww) if len(x) else np.nan;sd=np.sqrt(np.average((x-mean)**2,weights=ww)) if len(x) else np.nan
     if var in ['sex','race_ethnicity_legacy','education_adult','smoking_status']:
      for level in sorted(d.loc[mask,var].dropna().unique()):selection.append(dict(cycle=cycle,stage=stage,group=group,variable=var,level=level,n=int(mask.sum()),observed=int(good.sum()),missing=int((mask&~good).sum()),weighted_estimate=float(w[mask&d[var].eq(level)].sum()/ww.sum()) if ww.sum() else np.nan,weighted_sd=np.nan,statistic='Proportion among observed'))
     else:selection.append(dict(cycle=cycle,stage=stage,group=group,variable=var,level='',n=int(mask.sum()),observed=int(good.sum()),missing=int((mask&~good).sum()),weighted_estimate=mean,weighted_sd=sd,statistic='Mean; population weighted SD'))
 for v in ['day1_primary','two_day_average']:
  for split in ['development','temporal_test']:
   rep=pd.read_parquet(R/f'data/hei2020/representations/{split}/r5_{v}.parquet');features=json.loads((O/f'MODEL_OBJECT_METADATA_{v}.json').read_text())['features']
   for scope,g in list(rep.groupby('cycle_id'))+[(split,rep)]:
    for c in features:missing.append(dict(cycle_or_period=scope,view=v,variable=c,n=len(g),missing=int(g[c].isna().sum()),fraction=g[c].isna().mean()))
 csv(prevalence,'COMPONENT_PREVALENCE_BY_CYCLE.csv');csv(causes,'ENDPOINT_SENSITIVITY_CHANGED_COUNTS.csv');csv(bounds,'PARTIAL_ENDPOINT_SELECTION_BOUNDS.csv');csv(med,'MEDICATION_MAPPING_RECONSTRUCTION.csv');csv(selection,'MISSINGNESS_AND_SELECTION.csv');csv(missing,'PREDICTOR_MISSINGNESS_BY_CYCLE_PERIOD.csv')
 dictionary.variables_long[dictionary.variables_long.normalized_variable.isin(['waist_cm','sex','triglycerides_mg_dl','hdl_mg_dl','fasting_glucose_mg_dl','current_antihypertensive_questionnaire','self_reported_hypertension','current_insulin_use','current_diabetes_pills','self_reported_diabetes','rx_generic_drug_id','rx_generic_name','rx_use_past_month','fasting_hours_reported','fasting_minutes_reported']+['sbp_reading_'+str(i) for i in range(1,5)]+['dbp_reading_'+str(i) for i in range(1,5)])].to_csv(O/'ENDPOINT_VARIABLE_CYCLE_MAP.csv',index=False,encoding='utf-8-sig')
 pd.read_csv(settings.drug_id_map).to_csv(O/'ACTUAL_DRUG_ID_MAP.csv',index=False,encoding='utf-8-sig')
 # Cross-fitted response propensity sensitivity: original predictions untouched; descriptive MAR-dependent analysis.
 full=pd.concat(allframes,ignore_index=True);cov=['age_years','sex','race_ethnicity_legacy','education_adult','family_poverty_income_ratio'];cats=['sex','race_ethnicity_legacy','education_adult'];nums=[c for c in cov if c not in cats];weightrows=[];propout=[]
 for period,df in full.groupby('cycle_role'):
  if period not in ['development','temporal_test']:continue
  eligible=df.age_years.ge(20)&df.mec_exam_status.eq(2)&~(df.sex.eq(2)&df.pregnancy_status.eq(1))&df.fasting_weight_2yr.gt(0)&df.fasting_duration_hours.ge(8)&df.fasting_duration_hours.lt(24)
  diet=df.day1_recall_status.eq(1)&df.energy_kcal_day1.gt(0);endpoint=df.mets_prevalent.notna();primarymask=eligible&diet&endpoint
  for stage,base,response in [('day1_plus_endpoint',eligible,diet&endpoint),('day2_given_primary',primarymask,df.day2_recall_status.eq(1)&df.energy_kcal_day2.gt(0))]:
   d=df.loc[base].copy().reset_index(drop=True);y=response[base].astype(int).to_numpy();w=d.fasting_weight_2yr.to_numpy(float);prob=np.zeros(len(d));splits=StratifiedGroupKFold(5,shuffle=True,random_state=20261107).split(d,y,d.psu_pool)
   for tr,te in splits:
    pre=ColumnTransformer([
     ('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums),
     ('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cats)
    ])
    mod=Pipeline([('pre',pre),('fit',LogisticRegression(C=1,max_iter=2000))]);mod.fit(d.loc[tr,cov],y[tr],fit__sample_weight=w[tr]/w[tr].mean());prob[te]=mod.predict_proba(d.loc[te,cov])[:,1]
   clipped=np.clip(prob,.05,.95);adjusted=w/clipped
   # Stabilize total respondent weight to base population separately by cycle.
   for cyc in d.cycle_id.unique():
    m=d.cycle_id.eq(cyc).to_numpy();observed=m&(y==1);adjusted[observed]*=w[m].sum()/adjusted[observed].sum()
   for cyc in d.cycle_id.unique():
    m=d.cycle_id.eq(cyc).to_numpy();propout.append(dict(period=period,stage=stage,cycle=cyc,eligible=int(m.sum()),responders=int((m&(y==1)).sum()),min_propensity=prob[m].min(),max_propensity=prob[m].max(),propensity_clipped=int(((prob<.05)|(prob>.95))[m].sum()),assumption='MAR conditional on age sex race education PIR; 5-fold PSU cross-fitting; not causal correction'))
   out=d[['cycle_id','seqn']].copy();out['response']=y;out['response_probability']=prob;out['sensitivity_weight_2yr']=adjusted;out.to_parquet(PRIVATE/f'response_weights_{period}_{stage}.parquet')
 csv(propout,'RESPONSE_WEIGHT_DIAGNOSTICS.csv');logmeta('ENDPOINT_SELECTION_EXECUTION.json',dict(actual_drug_map=str(settings.drug_id_map),drug_map_sha256=sha(settings.drug_id_map),mapping_reconstruction='All cycles replayed from staged prescription rows',endpoint_sensitivities=['backward glucose on original cohort','measurement-only with classifiability restriction'],selection_adjustment='Exploratory cross-fitted response IPW; clipped to [.05,.95]; base weights calibrated to base weight total by cycle',state='Completed; source files read-only'))
 print(now(),'ENDPOINT AND SELECTION COMPLETED',flush=True)


if __name__ == "__main__":
 main()
