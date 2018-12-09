import numpy as np
import pandas as pd
from pandas import DataFrame

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin

class FillMeasuredColumns(TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, *args, **kwargs):
        return self
    
    def isMeasured(self, row, hormone):
        #print(row[hormone])
        if np.isnan(row[hormone]):
            return 0
        return 1
    
    def transform(self, df, **transform_params):
        measured_df = df.filter(regex='measured')
        for column in measured_df.columns:
            hormone_name = column.split(" ")[0]
            #print(hormone_name)
            df[column] = df.apply(lambda row: self.isMeasured(row, hormone_name), axis=1)
        
        return df