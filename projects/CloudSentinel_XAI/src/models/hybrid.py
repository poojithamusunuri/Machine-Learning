import numpy as np
from sklearn.ensemble import RandomForestClassifier,IsolationForest
from sklearn.metrics import precision_recall_curve
def f1_threshold(y,s):
 p,r,t=precision_recall_curve(y,s)
 if not len(t): return .5
 f=2*p[:-1]*r[:-1]/(p[:-1]+r[:-1]+1e-12); return float(t[int(np.nanargmax(f))])
class HybridDetector:
 def __init__(self,seed=42,contamination=.01): self.seed=seed; self.contamination=contamination
 def fit(self,X,y,benign_X):
  self.supervised=RandomForestClassifier(n_estimators=300,class_weight='balanced_subsample',min_samples_leaf=2,n_jobs=-1,random_state=self.seed).fit(X,y)
  self.anomaly=IsolationForest(n_estimators=250,contamination=self.contamination,n_jobs=-1,random_state=self.seed).fit(benign_X); return self
 def score(self,X):
  return self.supervised.predict_proba(X)[:,list(self.supervised.classes_).index(1)],-self.anomaly.score_samples(X)
 def calibrate(self,X,y,target_fpr=.01):
  s,a=self.score(X); self.supervised_threshold=f1_threshold(y,s); b=a[np.asarray(y)==0]
  if not len(b): raise ValueError('Validation must include benign rows')
  self.anomaly_threshold=float(np.quantile(b,1-target_fpr)); return self
 def predict_components(self,X):
  s,a=self.score(X); ss=s>=self.supervised_threshold; aa=a>=self.anomaly_threshold
  return {'supervised_score':s,'anomaly_score':a,'supervised_alert':ss,'anomaly_alert':aa,'hybrid_alert':ss|aa}
