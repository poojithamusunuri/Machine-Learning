import argparse,json
from pathlib import Path
import numpy as np
from src.data.loader import read_flow_csvs
def audit(df):
 n=df.drop(columns=['Label','__source_file__'],errors='ignore').select_dtypes(include=np.number)
 return {'rows':len(df),'columns':len(df.columns),'labels':df['Label'].fillna('<MISSING>').astype(str).value_counts().to_dict(),'missing_cells':int(df.isna().sum().sum()),'duplicates':int(df.drop(columns=['__source_file__'],errors='ignore').duplicated().sum()),'infinite_numeric_cells':int(np.isinf(n.to_numpy(dtype=float,copy=False)).sum()) if n.shape[1] else 0,'per_column':[{'name':str(c),'dtype':str(df[c].dtype),'missing':int(df[c].isna().sum()),'unique':int(df[c].nunique(dropna=True))} for c in df.columns]}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--input',default='data/raw'); p.add_argument('--output',default='outputs/audit'); a=p.parse_args(); df,prov=read_flow_csvs(a.input); r=audit(df); r['provenance']=prov; o=Path(a.output); o.mkdir(parents=True,exist_ok=True); (o/'dataset_audit.json').write_text(json.dumps(r,indent=2)); print(json.dumps({k:v for k,v in r.items() if k!='per_column'},indent=2))
if __name__=='__main__': main()
