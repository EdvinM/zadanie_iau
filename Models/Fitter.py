import numpy as np
import pandas as pd
from pandas import DataFrame

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin, BaseEstimator

class Fitter(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass
        
    def fit(self, *args, **kwargs):
        self.train = args[0]
        self.test = args[1]
        
        print("test- " + str(len(self.train)))
        print("test- " + str(len(self.test)))
        return self
    
    def transform(self, df, **transform_params):
        print("afoj edvin " + str(len(df)))
        return df