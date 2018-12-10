import numpy as np
import pandas as pd
import scipy.stats as stats

from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin

from scipy.stats import boxcox

class QuantileReplacer(TransformerMixin):
    def __init__(self, column):
        self.column = column
    
    def fit(self, *args, **kwargs):
        self.train = args[0]
        return self
    
    def transform(self, df, **transform_params):
        _, p = stats.shapiro(df[self.column])
        
        if float(p) < 0.05:
#             if df.equals(self.train):
#                 print("dano je noob")
            
            transformed, att = boxcox(df[df[self.column] > 0][self.column])
            df.loc[df[self.column] > 0, self.column] = transformed
        
        Q_down = self.train[self.column].quantile(0.05)
        Q_up = self.train[self.column].quantile(0.95)
        
        df.loc[df[self.column] <= Q_down, self.column] = Q_down
        df.loc[df[self.column] >= Q_up, self.column] = Q_up

        return df
