# Author: Michael DiSalvo
# Acknowledgements: Dr. Natalia Khuri's my_util.py
# Date: 9/21/2022

import pandas as pd
import numpy as np
import os
import sklearn


def get_filepaths(dir_path: str):
    filepaths = []
    for item in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, item)):
            filepaths.add(item)
    return filepaths


# check dataframe for any missing or repeated values
def check_missing_data(data: pd.DataFrame, printMissing: bool = True) -> list(int):
    rows = set()
    duplicates = False
    if data.isnull().values.any():
        print("Missing values have been found.")
        missing_data = data.isnull()
        for col in data.columns:
            missing_vals = data.loc[:, col][missing_data.loc[:, col]]
            if data[col].isna().values.any():
                if printMissing:
                    print('Column:\t', col)
                    print('Rows:\t', missing_vals.index.values)
                    print("="*20)
                for row in missing_vals.index.values:
                    rows.add(row)
    
    percent_missing = round((len(rows) / data.shape[0])*100, 2)
    print("Percent of Rows with missing values:\t", percent_missing)
    
    return list(rows)

def check_duplicate_data(data: pd.DataFrame) -> bool:
    duplicates = False
    if data.duplicated().any():
        print('Duplicate data has been found.')
        duplicates = True
    
    return duplicates
        
def cross_validate(model: sklearn.base.ClassifierMixin, features: np.ndarray, target: np.ndarray, folds=5, random_state=0) -> float:
    skf = sklearn.model_selection.StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    scores = []
    for train, test in skf.split(features, target):
        model.fit(features.iloc[train], target.iloc[train])
        score = model.score(features.iloc[test], target.iloc[test])
        scores.append(score)

    return np.mean(score)