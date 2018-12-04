from pandas import DataFrame

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin

from sklearn.neighbors import KNeighborsClassifier

class Classifier(TransformerMixin):
    def __init__(self, predicting_column, columns):
        self.predicting_column = predicting_column
        self.columns = columns
    
    def get_data_columns(self, data):
        train_X = data[data[self.predicting_column].isnull() == False][self.columns]
        train_y = data[self.predicting_column].dropna().astype(int)

        return train_X, train_y    
    
    def get_nan_data_columns(self, data):
        return data[data[self.predicting_column].isnull()][self.columns]
    
    def label_encoder(self, data, column_name):
        encoder = preprocessing.LabelEncoder()

        data.loc[data[self.predicting_column].isnull() == False, self.predicting_column] = encoder.fit_transform(data[data[self.predicting_column].isnull() == False][self.predicting_column])

        return encoder
    
    def label_decoder(self, data, encoder, column_name):
        return encoder.inverse_transform(data[column_name])
    
    def fit(self, *args, **kwargs):
        return self
    
    def transform(self, df, **transform_params):
        
        if df[self.predicting_column].dtype != np.dtype(np.float64):
            column_encoder = self.label_encoder(df, self.predicting_column)
        
        
        train_X, train_y = self.get_data_columns(df)
        train_X_nan = self.get_nan_data_columns(df)
        
        if len(train_X_nan) > 0:

            kNN = KNeighborsClassifier(n_neighbors=5)
            kNN.fit(train_X, train_y)

            predicted = kNN.predict(train_X_nan)

            df.loc[df[self.predicting_column].isnull(), self.predicting_column] = predicted


            if df[self.predicting_column].dtype != np.dtype(np.float64):
                df[self.predicting_column] = self.label_decoder(df, encoder, self.predicting_column).reshape(-1, 1)
        
        return df