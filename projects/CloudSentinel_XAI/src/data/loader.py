from pathlib import Path
import pandas as pd
def read_flow_csvs(path,max_rows_per_file=None):
 p=Path(path); paths=[p] if p.is_file() else sorted(p.rglob('*.csv')) if p.exists() else []
 if not paths: raise FileNotFoundError(f'No CSV files found under {path}')
 frames=[]; provenance=[]
 for f in paths:
  d=pd.read_csv(f,low_memory=False,nrows=max_rows_per_file); d.columns=[str(c).strip() for c in d.columns]
  label=next((c for c in ('Label','label',' Attack') if c in d.columns),None)
  if label is None: raise ValueError(f'No Label column in {f}')
  if label!='Label': d=d.rename(columns={label:'Label'})
  d['__source_file__']=f.name; frames.append(d); provenance.append({'file':str(f),'rows':len(d),'columns':len(d.columns)})
 return pd.concat(frames,ignore_index=True,sort=False),provenance
