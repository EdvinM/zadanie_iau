import numpy as np
import pandas as pd
import scipy.stats as stats

from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin

from scipy.stats import boxcox

class OneHotEncoder(TransformerMixin):
    def __init__(self, column):
        self.column = column
    
    def fit(self, *args, **kwargs):
        self.train = args[0]
        return self
    
    def transform(self, df, **transform_params):
        output = pd.get_dummies(df[self.column], prefix=self.column)

        df = pd.concat([df, output], axis=1)
            
        return df
