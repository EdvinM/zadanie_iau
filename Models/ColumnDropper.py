import numpy as np
import pandas as pd
from pandas import DataFrame

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin

class ColumnDropper(TransformerMixin):
    def __init__(self, column_names):
        self.column_names = column_names
        
    def fit(self, *args, **kwargs):
        return self
    
    def transform(self, df, **transform_params):
        return df.drop(self.column_names, axis=1)