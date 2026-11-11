"""Descriptive SD only, from the locked Day 1 cohort representations.

Requires access to the original archive. Never modifies inputs or opens a model.
Writes aggregate rows and hashes only, outside the source archive.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = {
    'R5': ['age_years','family_poverty_income_ratio','ltpa_mvpa_equivalent_min_week','alcohol_g_day1'],
    'R2': ['hei_saturated_fats','hei_total_fruits','hei_dairy'],
    'R4': ['total_violation_burden'],
}
KEYS=['cycle_id','seqn']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def moments(x,w):
    x=np.asarray(x,float);w=np.asarray(w,float)
    valid=np.isfinite(x)&np.isfinite(w)&(w>0)
    x,w=x[valid],w[valid]
    if len(x)<2: raise ValueError('Insufficient valid values')
    mean=np.average(x,weights=w)
    variance=np.sum(w*(x-mean)**2)/(w.sum()-(w@w)/w.sum())
    return int(valid.sum()),float(mean),float(np.sqrt(variance))

def derive(root):
    spec=json.loads((root/'reports/modeling/frozen/day1_primary/MODEL_SPEC_FROZEN.json').read_text(encoding='utf-8-sig'))
    lock=json.loads((root/'reports/modeling/temporal/day1_primary/TEMPORAL_EVALUATION_LOCK.json').read_text(encoding='utf-8-sig'))
    inputs=[];stats={}
    for split,weight,expected_n in [('development','weight_development_2007_2014',8399),('temporal_test','weight_test_2015_2018',3894)]:
        frames=[]
        for rep,features in FEATURES.items():
            rel=Path(f'data/hei2020/representations/{split}/{rep.lower()}_day1_primary.parquet')
            p=root/rel
            if split=='development':
                hashes={r['development_sha256'] for r in spec['registry'] if r['representation']==rep}
                assert len(hashes)==1,(rep,hashes)
                expected=hashes.pop()
            else:expected=lock['temporal_files'][rep]['sha256']
            actual=sha(p);assert actual==expected,rel
            d=pd.read_parquet(p,columns=KEYS+features+([weight] if rep=='R5' else []));assert len(d)==expected_n
            frames.append(d);inputs.append(dict(file=rel.as_posix(),sha256=actual,registry_sha256=expected,n=len(d)))
        d=frames[0].merge(frames[1],on=KEYS,validate='one_to_one').merge(frames[2],on=KEYS,validate='one_to_one')
        assert len(d)==expected_n
        stats[split]={x:moments(d[x],d[weight]) for f in FEATURES.values() for x in f}
    rows=[]
    for x,(dn,dm,ds) in stats['development'].items():
        tn,tm,ts=stats['temporal_test'][x]
        rows.append(dict(variable=x,development_observed_n=dn,development_mean=dm,development_sd=ds,temporal_observed_n=tn,temporal_mean=tm,temporal_sd=ts,smd=(tm-dm)/np.sqrt((ds**2+ts**2)/2)))
    return rows,inputs

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path,help='New aggregate output CSV outside the source archive')
    a=parser.parse_args();root=a.archive.resolve();out=a.output.resolve()
    if out.is_relative_to(root):parser.error('Output must be outside the source archive')
    if out.exists():parser.error('Choose a new output path; existing aggregate evidence is not overwritten')
    rows,inputs=derive(root);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with out.with_suffix('.provenance.json').open('x',encoding='utf-8') as f:
        json.dump(dict(method='Survey-weighted descriptive SD: sqrt(sum(w*(x-mean)^2)/(sum(w)-sum(w^2)/sum(w)))',cohorts='Existing Day 1 development and pooled temporal cohort; R2/R4/R5 one-to-one join',input_hashes=inputs,no_model_fitting_or_prediction=True,analysis='New descriptive SD only; existing means and SMDs cross-checked against baseline manuscript separately'),f,indent=2)
    print(f'Wrote {len(rows)} descriptive summary rows; six input hashes matched.')

if __name__=='__main__':main()
