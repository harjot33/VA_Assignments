from pathlib import Path
from typing import List
import pandas as pd
import numpy as np

from assignments.assignment1.a_load_file import read_dataset
from assignments.assignment1.b_data_profile import *

##############################################
# Example(s). Read the comments in the following method(s)
##############################################
#Profiling the dataset
def pandas_profile(df: pd.DataFrame, result_html: str = 'report.html'):
    """
    This method will be responsible to extract a pandas profiling report from the dataset.
    Do not change this method, but run it and look through the html report it generated.
    Always be sure to investigate the profile of your dataset (max, min, missing values, number of 0, etc).
    """
    from pandas_profiling import ProfileReport

    profile = ProfileReport(df, title="Pandas Profiling Report")
    if result_html is not None:
        profile.to_file(result_html)
    return profile.to_json()


##############################################
# Implement all the below methods
# All methods should be dataset-independent, using only the methods done in the assignment
# so far and pandas/numpy/sklearn for the operations
##############################################

# Getting the max column
def get_column_max(df: pd.DataFrame, column_name: str) -> float:
    column = df[column_name]
    col_max = column.max()
    return col_max


# Getting the min column
def get_column_min(df: pd.DataFrame, column_name: str) -> float:
    column = df[column_name]
    col_min = column.min()
    return col_min

# Getting the column mean
def get_column_mean(df: pd.DataFrame, column_name: str) -> float:
    column = df[column_name]
    col_mean = column.mean()
    return col_mean


# Getting number of nans in a column
def get_column_count_of_nan(df: pd.DataFrame, column_name: str) -> float:
    nansum = df[column_name].isna().sum()
    return nansum


# Getting the number of duplicates
def get_column_number_of_duplicates(df: pd.DataFrame, column_name: str) -> float:
    dfvals = df[column_name].tolist()
    dfvals_unique = set(dfvals)

    dupsum = len(dfvals) - len(dfvals_unique)
    return dupsum


# Getting the numeric columns
def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    dfno = df.select_dtypes(include='number')
    dfnolist = list(dfno.columns)
    return dfnolist

# Getting the binary columns
def get_binary_columns(df: pd.DataFrame) -> List[str]:
    dfbool = df.select_dtypes(include='bool')
    dfboollist = list(dfbool.columns)
    return dfboollist


# Getting the text categorical columns
def get_text_categorical_columns(df: pd.DataFrame) -> List[str]:
    textcolumns = df.select_dtypes(include=['object']).columns.tolist()
    return textcolumns

# Getting the correlation between the columns
def get_correlation_between_columns(df: pd.DataFrame, col1: str, col2: str) -> float:
    cs = df[col1].corr(df[col2], method='pearson')
    return cs


if __name__ == "__main__":
    df = read_dataset(Path(r'H:\New folder (2)\Assignments (3)\Assignments\iris.csv', 'iris.csv'))
   ## a = pandas_profile(df)
    assert get_column_max(df, df.columns[0]) is not None
    assert get_column_min(df, df.columns[0]) is not None
    assert get_column_mean(df, df.columns[0]) is not None
    assert get_column_count_of_nan(df, df.columns[0]) is not None
    assert get_column_number_of_duplicates(df, df.columns[0]) is not None
    assert get_numeric_columns(df) is not None
    assert get_binary_columns(df) is not None
    assert get_text_categorical_columns(df) is not None
    assert get_correlation_between_columns(df, df.columns[0], df.columns[1]) is not None
