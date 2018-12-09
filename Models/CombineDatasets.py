import numpy as np
import pandas as pd
from pandas import DataFrame

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin

class CombineDatasets(TransformerMixin):
    def __init__(self):
        #self.data_personal = data_personal
        #self.data_other = data_other
        return
        
    def handle_duplicate_rows(self, rows):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

        # Zoberieme si iba polozky numerickeho typu, nad ktorymi aplikujeme agregovanu funkciu: mean
        numeric_rows = rows.select_dtypes(include=numerics).agg(np.mean)

        # To iste urobime pre polozky ineho datoveho typu ako numericke, na ktore aplikujeme agregovanu funkciu najcastejsie
        # vyskytujucu hodnotu
        cat_rows = rows.select_dtypes(include=['object']).agg(self.most_common)

        # Skobinujeme tieto dva datasety
        combined = pd.concat([numeric_rows, cat_rows], axis=0, sort=True)
        #print(pd.DataFrame(combined))
        return combined

    # Funkcia ktora vrati najcastejsiu hodnotu
    def most_common(self, rows):
        counts = rows.value_counts()
        if len(counts) > 0:
            return counts.index[0]

        return float('nan')
        
    def fit(self, *args, **kwargs):
        self.data_personal = args[0]
        self.data_other = args[1]
        return self
    
    def transform(self, df, **transform_params):
        grouped = self.data_other[self.data_other.duplicated(subset=['name', 'address'], keep=False)].groupby(['name', 'address'])
        
        result = grouped.apply(self.handle_duplicate_rows).reset_index(drop=True)
        dropped_duplicates = self.data_other.drop_duplicates(subset=['name', 'address'], keep=False)
        self.data_other = pd.concat([dropped_duplicates, result], sort=True)
        
        df = pd.merge(self.data_personal, self.data_other, how='inner', left_on=['name', 'address'], right_on=['name', 'address'])
        
        return df