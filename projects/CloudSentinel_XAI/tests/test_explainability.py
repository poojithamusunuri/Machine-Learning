import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.explainability import local_feature_sensitivity

def test_local_sensitivity_returns_ranked_features():
    rng=np.random.default_rng(8)
    X=rng.normal(size=(120,3)); y=(X[:,0] + .1*rng.normal(size=120)>0).astype(int)
    model=RandomForestClassifier(n_estimators=20,random_state=8).fit(X,y)
    result=local_feature_sensitivity(model,X[0],X[:30],['f0','f1','f2'],top_k=2)
    assert list(result.columns)[0]=='feature'
    assert len(result)==2
    assert result['absolute_delta'].is_monotonic_decreasing
