import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
DROP={'Label','__source_file__','Flow ID','Src IP','Dst IP','Timestamp','Source IP','Destination IP','Src Port','Dst Port','Source Port','Destination Port'}
def clean_frame(df,extra_drop=None):
 f=df.copy(); f.columns=[str(c).strip() for c in f.columns]
 if 'Label' not in f: raise ValueError('Expected Label column')
 y=f['Label'].astype('string').str.strip().astype(str); X=f.drop(columns=list(DROP|set(extra_drop or [])),errors='ignore')
 X=X.drop(columns=[c for c in X.columns if str(c).strip().lower().startswith('unnamed:')],errors='ignore')
 for c in X.columns:
  if X[c].dtype=='object':
   n=pd.to_numeric(X[c],errors='coerce')
   if X[c].notna().sum() and n.notna().sum()/X[c].notna().sum()>=.9: X[c]=n
   else: X[c]=X[c].astype('string').str.strip()
 X=X.replace([np.inf,-np.inf],np.nan).loc[:,lambda d:d.notna().any()]
 return X,y
def build_preprocessor(X):
 num=X.select_dtypes(include=['number','bool']).columns.tolist(); cat=[c for c in X.columns if c not in num]
 n=Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())])
 c=Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',min_frequency=5))])
 return ColumnTransformer([('num',n,num),('cat',c,cat)],verbose_feature_names_out=False)
