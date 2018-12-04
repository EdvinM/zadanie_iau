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
    
    def transform(self, df, **transform_params):
        new_columns = df.pop(self.column_name).apply(lambda x: pd.Series(ast.literal_eval(x)))
        
        return df.join(new_columns)