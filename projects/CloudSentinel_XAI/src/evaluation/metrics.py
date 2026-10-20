import numpy as np
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score,precision_score,recall_score,roc_auc_score,average_precision_score
def binary_metrics(y,pred,scores=None):
 y=np.asarray(y,dtype=int); pred=np.asarray(pred,dtype=int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 m={'n':int(len(y)),'accuracy':float(accuracy_score(y,pred)),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'false_positive_rate':float(fp/(fp+tn)) if fp+tn else None}
 if scores is not None and len(np.unique(y))==2: m.update(roc_auc=float(roc_auc_score(y,scores)),pr_auc=float(average_precision_score(y,scores)))
 return m
