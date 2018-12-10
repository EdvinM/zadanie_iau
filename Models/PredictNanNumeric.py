import numpy as np
import pandas as pd
from pandas import DataFrame

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

class PredictNanNumeric(TransformerMixin):
    def __init__(self, predicting_column, columns):
        self.predicting_column = predicting_column
        self.columns = columns
        
    def get_data_columns(self, data):
        train_X = data[data[self.predicting_column].isnull() == False][self.columns]
        train_y = data[self.predicting_column].dropna().astype(int)

        return train_X, train_y    
    
    def get_nan_data_columns(self, data):
        return data[data[self.predicting_column].isnull()][self.columns]
        
    def fit(self, *args, **kwargs):
        self.train = args[0]
        return self
    
    def transform(self, df, **transform_params):
        train_X, train_y = self.get_data_columns(self.train)
        train_X_nan = self.get_nan_data_columns(df)
        
        if len(train_X_nan) <= 0:
            return df
        
        lr = LinearRegression()
        lr_model = lr.fit(train_X, train_y)
        lr_score = lr_model.score(train_X, train_y)
        

        knnr = KNeighborsRegressor(5)
        knnr_model = knnr.fit(train_X, train_y)
        knnr_score = knnr_model.score(train_X, train_y)
        
        if lr_score > knnr_score:
            predicted = lr_model.predict(train_X_nan)
        else:
            predicted = knnr_model.predict(train_X_nan)
            
        df.loc[df[self.predicting_column].isnull(), self.predicting_column] = predicted
        
        return df