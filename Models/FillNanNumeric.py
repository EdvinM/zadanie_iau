from pandas import DataFrame

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin

class FillNanNumeric(TransformerMixin):
    def __init__(self, strategy, column_name):
        self.strategy = strategy
        self.column_name = column_name
        
    def fit(self, *args, **kwargs):
        return self
    
    def transform(self, df, **transform_params):
        imputer = SimpleImputer(missing_values=np.nan, strategy=self.strategy)
        imputer = imputer.fit(np.array(df[self.column_name]).reshape(-1, 1))
        
        output = imputer.transform(np.array(df[self.column_name]).reshape(-1, 1))
        df[self.column_name] = output
        
        return df