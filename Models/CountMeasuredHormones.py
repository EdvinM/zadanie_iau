from pandas import DataFrame

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin

class CountMeasuredHormones(TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name
        
    def fit(self, *args, **kwargs):
        return self
    
    def transform(self, df, **transform_params):
        measured_df = df.filter(regex='measured')
        
        # Urobime sucet po stlpcov pre dany riadok
        result = measured_df.sum(axis=1)

        # Vytvorime novy atribut 'measured_hormones' kde priradime tento sucet
        df[self.column_name] = result
        
        return df