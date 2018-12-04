from pandas import DataFrame

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin

import dateutil.parser as parser

class FixDates(TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, *args, **kwargs):
        return self
    
    def formatDate(self, dateString):
        if not isinstance(dateString, str):
            return float('nan')
        
        # zoberiem si len datum, cas orezem
        dateString = dateString[:min(len(dateString), 10)]
        dateObj = parser.parse(dateString, yearfirst = True)
        
        return dateObj.strftime('%Y-%m-%d')
    
    def transform(self, df, **transform_params):
        df['date_of_birth'] = df['date_of_birth'].map(lambda dateString: self.formatDate(dateString))
        
        return df