# Author: Michael DiSalvo
# Acknowledgements: Dr. Natalia Khuri's my_util.py
# Date: 9/21/2022

import pandas as pd
import numpy as np

# check dataframe for any missing or repeated values
def check_missing_repeated(data: pd.DataFrame, printMissing = True):

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
    
    if data.duplicated().any():
        print('Duplicate data has been found.')
        duplicates = True

    percent_missing = round((len(rows) / data.shape[0])*100, 2)
    print("Percent of Rows with missing values:\t", percent_missing)
    


    
    return list(rows), duplicates
        
