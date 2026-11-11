import os,sys,pathlib,json,hashlib,datetime
BASE=pathlib.Path(__file__).resolve().parent.parent

def execution_paths():
    """Initialize research I/O only when a data-dependent script is run."""
    global R,O,PRIVATE
    source=os.environ.get('NHANES_SOURCE_ROOT')
    output=os.environ.get('NHANES_OUTPUT_ROOT')
    if not source or not output:
        raise SystemExit('Set NHANES_SOURCE_ROOT to an authorized archive and NHANES_OUTPUT_ROOT to a separate writable output directory. See README.md.')
    R=pathlib.Path(source).resolve()
    out=pathlib.Path(output).resolve()
    if not (R/'data').is_dir():
        raise SystemExit('NHANES_SOURCE_ROOT has no data directory. No research files were opened.')
    if any(out.is_relative_to(p) or p.is_relative_to(out) for p in (R,BASE)):
        raise SystemExit('NHANES_OUTPUT_ROOT must be separate from both the source archive and the release candidate.')
    O=out/'aggregate';PRIVATE=out/'private'
    O.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
    return R,O,PRIVATE

import numpy as np,pandas as pd
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def primary(d):return d[d.representation.eq('R5')&d.algorithm.eq('boost_monotonic')].copy().reset_index(drop=True)
def preds(view,period='temporal'):
 return pd.read_parquet(R/f'data/modeling/{period}/{view}/{"temporal_predictions" if period=="temporal" else "nested_oof_predictions"}.parquet')
def csv(rows,name):pd.DataFrame(rows).to_csv(O/name,index=False,encoding='utf-8-sig')
def logmeta(name,extra):
 (O/name).write_text(json.dumps(dict(timestamp_utc=now(),**extra),indent=2,default=str),encoding='utf8')
