import numpy as np
import pandas as pd
from pandas import DataFrame

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin

import ast

class DecodeJSONColumn(TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name
        
    def fit(self, *args, **kwargs):
        return self
    
    def handleJson(self, row):
        if pd.isna(row):
        #if row == "NaN":
            return pd.Series()
        return pd.Series(ast.literal_eval(row))
    
    def transform(self, df, **transform_params):
        new_columns = df.pop(self.column_name).apply(lambda x: self.handleJson(x))
        #print(new_columns)
        #if new_columns == False:
        #    return df
        return df.join(new_columns)