import argparse,json
from pathlib import Path
import yaml,numpy as np
from sklearn.model_selection import train_test_split
from src.data.loader import read_flow_csvs
from src.data.preprocessing import clean_frame,build_preprocessor
from src.models.hybrid import HybridDetector
from src.evaluation.metrics import binary_metrics
def run(cfg):
 seed=int(cfg.get('random_seed',42)); raw,prov=read_flow_csvs(cfg['data_dir'],cfg.get('max_rows_per_file')); X,labels=clean_frame(raw); fam=labels.map(cfg.get('family_by_label',{})).fillna(labels); benign=labels.str.upper().eq(str(cfg.get('benign_label','BENIGN')).upper()); held=str(cfg['heldout_family']); hm=fam.str.casefold().eq(held.casefold())
 if not hm.any(): raise ValueError(f'No rows mapped to {held!r}; available labels/families: {sorted(fam.unique())}')
 Xd=X.loc[~hm].reset_index(drop=True); yd=(~benign.loc[~hm]).astype(int).reset_index(drop=True); Xh=X.loc[hm].reset_index(drop=True)
 if benign.loc[hm].any(): raise ValueError('Held-out family mapping unexpectedly includes benign rows')
 Xtmp,Xtest,ytmp,ytest=train_test_split(Xd,yd,test_size=float(cfg.get('test_size',.2)),random_state=seed,stratify=yd); rel=float(cfg.get('validation_size',.2))/(1-float(cfg.get('test_size',.2))); Xtr,Xv,ytr,yv=train_test_split(Xtmp,ytmp,test_size=rel,random_state=seed,stratify=ytmp)
 prep=build_preprocessor(Xtr); A=prep.fit_transform(Xtr); V=prep.transform(Xv); T=prep.transform(Xtest)
 # Held-out evaluation contains held-out-family attacks plus benign rows from the untouched known-family test partition.
 benign_test_mask=(np.asarray(ytest)==0); Xhold_eval=__import__('pandas').concat([Xh, Xtest.loc[benign_test_mask]],ignore_index=True)
 yhold_eval=np.concatenate([np.ones(len(Xh),dtype=int),np.zeros(int(benign_test_mask.sum()),dtype=int)])
 H=prep.transform(Xhold_eval)
 det=HybridDetector(seed,float(cfg.get('contamination',.01))).fit(A,np.asarray(ytr),A[np.asarray(ytr)==0]).calibrate(V,np.asarray(yv),float(cfg.get('benign_fpr_target',.01))); results={}
 for split,y,Z in [('known_family_test',ytest,T),('heldout_family_test_plus_benign',yhold_eval,H)]:
  q=det.predict_components(Z); combo=np.maximum(q['supervised_score']/max(det.supervised_threshold,1e-8),q['anomaly_score']/max(det.anomaly_threshold,1e-8)); results[split]={'supervised':binary_metrics(y,q['supervised_alert'],q['supervised_score']),'anomaly_only':binary_metrics(y,q['anomaly_alert'],q['anomaly_score']),'hybrid_union':binary_metrics(y,q['hybrid_alert'],combo)}
 out=Path(cfg.get('output_dir','outputs')); art=Path(cfg.get('artifact_dir','artifacts')); (out/'metrics').mkdir(parents=True,exist_ok=True); art.mkdir(parents=True,exist_ok=True); (out/'metrics/metrics.json').write_text(json.dumps(results,indent=2,allow_nan=False))
 import joblib; joblib.dump({'preprocessor':prep,'detector':det},art/'cloudsentinel_pipeline.joblib')
 meta={'rows':len(raw),'heldout_family':held,'heldout_attack_rows':len(Xh),'heldout_eval_benign_rows':int((np.asarray(ytest)==0).sum()),'train_rows':len(Xtr),'validation_rows':len(Xv),'known_test_rows':len(Xtest),'thresholds':{'supervised':det.supervised_threshold,'anomaly':det.anomaly_threshold},'labels':labels.value_counts().to_dict(),'source_files':prov,'protocol':'held-out family excluded from fitting/calibration'}; (out/'metrics/run_metadata.json').write_text(json.dumps(meta,indent=2))
 report=['# CloudSentinel-XAI experiment report','',f'- Rows loaded: {len(raw):,}',f'- Held-out family: {held} ({len(Xh):,} rows)','']
 for split,models in results.items():
  report += [f'## {split}','','| Detector | Precision | Recall | F1 | FPR | PR-AUC |','|---|---:|---:|---:|---:|---:|']
  for name,m in models.items():
   fpr='NA' if m['false_positive_rate'] is None else f"{m['false_positive_rate']:.4f}"; pr='NA' if 'pr_auc' not in m else f"{m['pr_auc']:.4f}"
   report.append(f"| {name} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {fpr} | {pr} |")
 report += ['', '## Limitations','', 'Held-out family is a novelty proxy, not proof of detecting all real-world zero-days. Hybrid union can increase false positives. Verify mapping and dataset provenance.']; Path('reports').mkdir(exist_ok=True); Path('reports/experiment_report.md').write_text('\n'.join(report)); print(json.dumps(results,indent=2)); return results
def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/experiment.yaml'); a=p.parse_args(); cfg=yaml.safe_load(Path(a.config).read_text()); mp=Path('configs/attack_family_map.yaml')
 if mp.exists(): cfg['family_by_label']=yaml.safe_load(mp.read_text()).get('family_by_label',{})
 run(cfg)
if __name__=='__main__': main()
