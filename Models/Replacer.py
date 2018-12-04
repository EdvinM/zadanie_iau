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
    
    def transform(self, df, **transform_params):
        for column in self.columns:
            df[column] = df[column].map(lambda x: self.replaceWith if x == self.replaceWhat else self.typeFormatter(x))
        
        return df