import numpy as np
import pandas as pd
from pandas import DataFrame

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin

class BinaryEncoder(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, *args, **kwargs):
        return self 
    
    def mapToBool(self, value):
        if not isinstance(value, str):
            return float('nan')
        
        return (int(1) if value[0].lower() == 't' else int(0))
    
    def transform(self, df, **transform_params):
        for column in self.columns:
            if column in df.columns:
                df[column] = df[column].map(lambda value: self.mapToBool(value))
            
        return df