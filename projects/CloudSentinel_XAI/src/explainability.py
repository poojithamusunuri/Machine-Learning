"""Model-agnostic feature sensitivity diagnostics (not causal explanations)."""
import numpy as np
import pandas as pd

def local_feature_sensitivity(model, row, background, feature_names, top_k=12):
    """Measure score change when each feature is replaced by its background median.

    This is a perturbation diagnostic, not SHAP and not a causal counterfactual.
    Correlated features can make individual effects misleading.
    """
    x=np.asarray(row,dtype=float).reshape(1,-1)
    bg=np.asarray(background,dtype=float)
    names=list(feature_names)
    if x.shape[1] != len(names) or bg.shape[1] != len(names):
        raise ValueError("Feature count mismatch")
    cls=list(model.classes_).index(1)
    baseline=float(model.predict_proba(x)[0,cls])
    med=np.nanmedian(bg,axis=0)
    rows=[]
    for j,name in enumerate(names):
        changed=x.copy(); changed[0,j]=med[j]
        score=float(model.predict_proba(changed)[0,cls])
        rows.append({"feature":str(name),"baseline_attack_score":baseline,
                     "score_after_median_replacement":score,"score_delta":score-baseline,
                     "absolute_delta":abs(score-baseline)})
    return pd.DataFrame(rows).sort_values("absolute_delta",ascending=False).head(top_k).reset_index(drop=True)

def global_permutation_sensitivity(model, X, y, feature_names, n_repeats=3, seed=42, max_features=100):
    """Estimate held-out score sensitivity by permuting one encoded feature at a time.

    Supply development validation data only when using results to make modeling decisions.
    """
    from sklearn.metrics import average_precision_score
    X=np.asarray(X,dtype=float); y=np.asarray(y,dtype=int); names=list(feature_names)
    if X.shape[1] != len(names): raise ValueError("Feature count mismatch")
    if X.shape[1] > max_features:
        imp=np.asarray(getattr(model,"feature_importances_",np.ones(X.shape[1])))
        candidates=np.argsort(imp)[-max_features:]
    else: candidates=np.arange(X.shape[1])
    cls=list(model.classes_).index(1); baseline=average_precision_score(y,model.predict_proba(X)[:,cls])
    rng=np.random.default_rng(seed); out=[]
    for j in candidates:
        drops=[]
        for _ in range(n_repeats):
            xp=X.copy(); xp[:,j]=xp[rng.permutation(len(xp)),j]
            drops.append(baseline-average_precision_score(y,model.predict_proba(xp)[:,cls]))
        out.append({"feature":names[j],"baseline_pr_auc":float(baseline),"mean_pr_auc_drop":float(np.mean(drops)),"std_pr_auc_drop":float(np.std(drops))})
    return pd.DataFrame(out).sort_values("mean_pr_auc_drop",ascending=False).reset_index(drop=True)
