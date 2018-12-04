from pandas import DataFrame

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin

from datetime import datetime

class ComputeCurYear(TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, *args, **kwargs):
        return self
    
    def getCurYear(self, row):
        if not isinstance(row['date_of_birth'], str):
            return float('nan')
        
        datetime_object = datetime.strptime(row['date_of_birth'], '%Y-%m-%d')
        return datetime_object.year + int(row['age'])
    
    def transform(self, df, **transform_params):
        df['cur_year'] = df.apply(lambda row: self.getCurYear(row), axis=1)
        
        return df