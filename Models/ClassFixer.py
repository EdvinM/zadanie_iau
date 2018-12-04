from pandas import DataFrame

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin

class ClassFixer(TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, *args, **kwargs):
        return self
    
    def transform(self, df, **transform_params):
        df['class'] = df['class'].map(lambda x: str(x).split('.')[0])
        return df