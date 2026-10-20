from src.evaluation.metrics import binary_metrics
def test_metrics():
 m=binary_metrics([0,0,1,1],[0,1,1,1],[.1,.7,.8,.9]); assert (m['tn'],m['fp'],m['fn'],m['tp'])==(1,1,0,2); assert m['recall']==1.0
