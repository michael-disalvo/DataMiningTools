import pandas as pd
import numpy as np
import os

from sklearn.impute import SimpleImputer, typing
from typing import List


class Preprocessor:
    
    def __init__(self):
        self.rows_removed = 0
        self.features_removed = 0
        self.feature_names = []

    '''
        # num_std: used in outlier removal
        # variance: used in low variance feature removal
        # correlation: used in high correlation feature removal
    '''
    def fit(self, data: pd.DataFrame, thresholds: typing.Dict[str, float]):
        self.num_std_threshold = thresholds.get('num_std', 5)
        self.variance_threshold = thresholds.get('variance', 1e-4)
        self.correlation_threshold = thresholds.get('correlation', .9)
        self.feature_names = self.select_features(data).columns
        print(f"selected {len(self.feature_names)} features")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.rows_removed = 0
        self.features_removed = 0
        nrows, ncols = data.shape
        data = self.clean_rows(data)
        data = data[self.feature_names]
        print(_title('REPORT:'))
        print(f"{self.rows_removed}/{nrows} rows removed")
        print(f"{ncols - len(self.feature_names)}/{ncols} features removed")
        return data

    def summary(self, data: pd.DataFrame):
        _summary_size(data)
        _summary_missing_data(data)
    
    def clean_rows(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self.remove_outliers(self.remove_missing_values(self.remove_duplicates(data)), self.num_std_threshold)
        return data
    def select_features(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self.remove_correlated_features(self.remove_low_variance(data, self.variance_threshold), self.correlation_threshold)
        return data

    # ROW CLEANING METHODS
    
    def remove_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        nrows = data.shape[0]
        data = data.dropna()
        self.rows_removed += nrows - data.shape[0]
        return data
    def impute_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer = imputer.fit(data)
        return imputer.transform(data)
    def remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        nrows = data.shape[0]
        data = data.drop_duplicates()
        self.rows_removed += nrows - data.shape[0]
        return data
    def remove_outliers(self, data: pd.DataFrame, number_std: int = 5) -> pd.DataFrame:
        nrows = data.shape[0]
        numeric_columns = data.select_dtypes(include='number').columns
        for col in numeric_columns:
            mean = data[col].mean()
            std = data[col].std()
            data = data[(data[col] <= mean + (number_std * std))]
            data = data[(data[col] >= mean - (number_std * std))]
        self.rows_removed += nrows - data.shape[0]
        return data



    # FEATURE CLEANING METHODS 
    def remove_low_variance(self, data: pd.DataFrame, threshold: float=1e-4) -> pd.DataFrame:
        numeric_cols = data.select_dtypes(include='number').columns
        categorical_cols = data.select_dtypes(include='object').columns

        columns_to_drop = []
        for column in numeric_cols:
            if data[column].std()**2 < threshold:
                columns_to_drop.append(column)
        
        for column in categorical_cols:
            if data[column].value_counts().size == 1:
                columns_to_drop.append(column)
        
        self.features_removed += len(columns_to_drop)
        data = data.drop(columns=columns_to_drop)
        return data
    def remove_correlated_features(self, data: pd.DataFrame, threshold=0.9) -> pd.DataFrame:
        numeric_cols = data.select_dtypes('number').columns
        corr_matrix = data[numeric_cols].corr().abs()
        cols_to_drop = set()
        for i, col in enumerate(corr_matrix.columns):
            for j in range(i):
                if abs(corr_matrix.iloc[i,j]) > threshold:
                    cols_to_drop.add(col)
        self.features_removed += len(cols_to_drop)
        data = data.drop(columns=cols_to_drop)
        return data
    



    

        

    
    
def _summary_size(data: pd.DataFrame):
        print(_title('GENERAL'))
        nrows, ncols = data.shape
        print(f"number of instances: {nrows}")
        print(f"number of features: {ncols}")

def _summary_missing_data(data: pd.DataFrame, printMissing: bool = True):
        print(_title('MISSING DATA'))
        rows = set()
        columns = set()

        for col in data.columns:
            if data[col].isna().values.any():
                columns.add(col)
        for row in data.index:
            if data.loc[row].isna().values.any():
                rows.add(row)

        
        nrows, ncols = data.shape
        print(f"Rows with missing values: {len(rows)}/{nrows}")
        print(f"Cols with missing values: {len(columns)}/{ncols}")
        
        return 


def _title(title: str):
        return f"{'='*10}{title}{'='*10}"