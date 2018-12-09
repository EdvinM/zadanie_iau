import numpy as np
import pandas as pd
from pandas import DataFrame

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin

class Replacer(TransformerMixin):
    def __init__(self, columns, replaceWhat, replaceWith, typeFormatter):
        self.columns = columns
        self.replaceWhat = replaceWhat
        self.replaceWith = replaceWith
        self.typeFormatter = typeFormatter
        
    def fit(self, *args, **kwargs):
        return self
    
    def getFormatted(self, number):
        if number == self.replaceWhat: #or not type(self.typeFormatter) is number 
            return self.replaceWith
        else:
            return self.typeFormatter(number)
    
    def transform(self, df, **transform_params):
        for column in self.columns:
            if column in df.columns:
                df[column] = df[column].map(lambda x: self.getFormatted(x))
        
        return df