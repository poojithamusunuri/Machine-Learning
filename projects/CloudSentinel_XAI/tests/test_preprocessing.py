import numpy as np,pandas as pd
from src.data.preprocessing import clean_frame
def test_clean():
 X,y=clean_frame(pd.DataFrame({'Flow ID':['a','b'],'Timestamp':['x','y'],'v':[1.,np.inf],'Label':['BENIGN','Bot']})); assert 'Flow ID' not in X and X['v'].isna().sum()==1 and y.tolist()==['BENIGN','Bot']
