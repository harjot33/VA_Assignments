import collections
from pathlib import Path
from typing import Union, Optional
from enum import Enum
import random

import pandas as pd
import numpy as np

from assignments.assignment1.b_data_profile import *
from assignments.assignment1.a_load_file import read_dataset


##############################################
# Example(s). Read the comments in the following method(s)
##############################################
class WrongValueNumericRule(Enum):
    """
    You'll use these enumeration possibilities in your implemented methods below
    """
    MUST_BE_POSITIVE = 0
    MUST_BE_NEGATIVE = 1
    MUST_BE_GREATER_THAN = 2
    MUST_BE_LESS_THAN = 3


class DistanceMetric(Enum):
    """
    You'll use these enumeration possibilities in your implemented methods below
    """
    EUCLIDEAN = 0
    MANHATTAN = 1


##############################################
# Implement all the below methods
# All methods should be dataset-independent, using only the methods done in the assignment
# so far and pandas/numpy/sklearn for the operations
##############################################

# Finding the wrong numeric values and then fixing them
def fix_numeric_wrong_values(df: pd.DataFrame,
                             column: str,
                             must_be_rule: WrongValueNumericRule,
                             must_be_rule_optional_parameter: Optional[float] = None) -> pd.DataFrame:
    """
    This method should fix the wrong_values depending on the logic you think best to do and using the rule passed by parameter.
    Remember that wrong values are values that are in the dataset, but are wrongly inputted (for example, a negative age).
    Here you should only fix them (and not find them) by replacing them with np.nan ("not a number", or also called "missing value")
    :param df: Dataset
    :param column: the column to be investigated and fixed
    :param must_be_rule: one of WrongValueNumericRule identifying what rule should be followed to flag a value as a wrong value
    :param must_be_rule_optional_parameter: optional parameter for the "greater than" or "less than" cases
    :return: The dataset0 with fixed column
    """

    df_copy = df.copy()

    if must_be_rule == WrongValueNumericRule.MUST_BE_LESS_THAN:
        df_copy.loc[df_copy[column] >=
                    must_be_rule_optional_parameter, column] = np.nan
    elif must_be_rule == WrongValueNumericRule.MUST_BE_GREATER_THAN:
        df_copy.loc[df_copy[column] <=
                    must_be_rule_optional_parameter, column] = np.nan
    elif must_be_rule == WrongValueNumericRule.MUST_BE_NEGATIVE:
        df_copy.loc[df_copy[column] >= 0, column] = np.nan
    elif must_be_rule == WrongValueNumericRule.MUST_BE_POSITIVE:
        df_copy.loc[df_copy[column] < 0, column] = np.nan

    return df_copy

# Fixing the outliers in the column
def fix_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    This method should fix the column in respective to outliers depending on the logic you think best to do.
    Feel free to choose which logic you prefer, but if you are in doubt, use the simplest one to remove the row
    of the dataframe which is an outlier (note that the issue with this approach is when the dataset is small,
    dropping rows will make it even smaller).
    Remember that some datasets are large, and some are small, so think wisely on how to calculate outliers
    and when to remove/replace them. Also remember that some columns are numeric, some are categorical, some are binary
    and some are datetime. Use the methods in b_dataset_profile to your advantage!
    :param df: Dataset
    :param column: the column to be investigated and fixed
    :return: The dataset with fixed column
    """
    # Firstly, retrieved the numeric columns from the b_data_profile python script, such that after this,
    # I made use of the IQR method to remove the outliers, by splitting the data into the upper and lower limit.
    # All the values which are above the upper limit and below the lower limit are then discarded.
    ls = get_numeric_columns(df)
    if ls is None:
        return df
    else:
        if column in ls:
            LowPercent25 = df[column].quantile(0.25)
            HighPercent75 = df[column].quantile(0.75)
            IQR = HighPercent75 - LowPercent25

            LowPercent25_limit = LowPercent25 - (1.5 * IQR)
            HighPercent75_limit = HighPercent75 + (1.5 * IQR)
            df = df[(df[column] < HighPercent75_limit) & (df[column] > LowPercent25_limit)]

    return df


# Fixing the nans that were present in the code
def fix_nans(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    This method should fix all nans (missing data) depending on the logic you think best to do
    Remember that some datasets are large, and some are small, so think wisely on when to use each possible
    removal/fix/replace of nans. Also remember that some columns are numeric, some are categorical, some are binary
    and some are datetime. Use the methods in b_dataset_profile to your advantage!
    :param df: Dataset
    :param column: the column to be investigated and fixed
    :return: The fixed dataset
    """

    # dropping rows which have all nans
    df.dropna(how='all', inplace=True)

    # Getting the respective columns
    numeric_cols = get_numeric_columns(df)
    binary_cols = get_binary_columns(df)
    category_cols = get_text_categorical_columns(df)

    # Replacing nans in numeric columns with the mean
    if column in numeric_cols:
        mean = df[column].mean()
        df[column] = df[column].fillna(mean)
        return df

    # Replacing nans in binary columns with the random.choices method where we take the weights of
    # each choice as equal and then take the length of nans i.e how many times they have ocurred
    # and then we assign the values based on this, nans get replaced.
    if column in binary_cols:
        len = df[column].isna().sum()
        nanvalues = df[column].isna()
        replacenans = random.choices([0, 1], weights=[.5, .5], k=len)
        df.loc[nanvalues, column] = replacenans
        return df

    # Replacing the nans in categorical columns with the mode of the values in the column
    if column in category_cols:
        mode = df[column].mode()[0]
        df[column] = df[column].fillna(mode)
        return df

    return df


# Normalization of the column
def normalize_column(df_column: pd.Series) -> pd.Series:
    """
    This method should recalculate all values of a numeric column and normalise it between 0 and 1.
    :param df_column: Dataset's column
    :return: The column normalized
    """

    # Normalizing the data in the column using the following formula - zi = (xi – min(x)) / (max(x) – min(x))
    if df_column.dtype == np.number:
        df_column = (df_column - df_column.min()) / (df_column.max() - df_column.min())
        return df_column

    return df_column


# Standarization of the column
def standardize_column(df_column: pd.Series) -> pd.Series:
    """
    This method should recalculate all values of a numeric column and standardize it between -1 and 1 with its average at 0.
    :param df_column: Dataset's column
    :return: The column standardized
    """

    # Standardization of data, such that it has 0 mean and unit variance
    if df_column.dtype == np.number:
        df_column = ((df_column - df_column.min()) / (df_column.max() - df_column.min())) * (2) - 1

        return df_column

    return df_column


# Calculating the numeric distance between two columns
def calculate_numeric_distance(df_column_1: pd.Series, df_column_2: pd.Series,
                               distance_metric: DistanceMetric) -> pd.Series:
    """
    This method should calculate the distance between two numeric columns
    :param df_column_1: Dataset's column
    :param df_column_2: Dataset's column
    :param distance_metric: One of DistanceMetric, and for each one you should implement its logic
    :return: A new 'column' with the distance between the two inputted columns
    """
    if df_column_1.dtype == np.number and df_column_2.dtype == np.number:

        if distance_metric == DistanceMetric.MANHATTAN:
            npdistM = np.abs(df_column_1 - df_column_2)
            return npdistM
        elif distance_metric == DistanceMetric.EUCLIDEAN:
            npdistE = np.sqrt(np.square(df_column_1 - df_column_2))
            return npdistE

    return None


# Calculating the binary distance between two columns
def calculate_binary_distance(df_column_1: pd.Series, df_column_2: pd.Series) -> pd.Series:
    """
    This method should calculate the distance between two binary columns.
    Choose one of the possibilities shown in class for future experimentation.
    :param df_column_1: Dataset's column
    :param df_column_2: Dataset's column
    :return: A new 'column' with the distance between the two inputted columns
    """
    # Calculating the Binary distance between two columns.
    blist = []
    for col1value, col2value in zip(df_column_1, df_column_2):
        if (col1value == 0 and col2value == 1) or (col1value == 1 and col2value == 0):
            blist.append(1)
        else:
            blist.append(2)

    return pd.Series(blist)


if __name__ == "__main__":
    df = pd.DataFrame({'a': [1, 2, 3, None], 'b': [True, True, False, None], 'c': ['one', 'two', np.nan, None]})
    assert fix_numeric_wrong_values(df, 'a', WrongValueNumericRule.MUST_BE_LESS_THAN, 2) is not None
    assert fix_numeric_wrong_values(df, 'a', WrongValueNumericRule.MUST_BE_GREATER_THAN, 2) is not None
    assert fix_numeric_wrong_values(df, 'a', WrongValueNumericRule.MUST_BE_POSITIVE, 2) is not None
    assert fix_numeric_wrong_values(df, 'a', WrongValueNumericRule.MUST_BE_NEGATIVE, 2) is not None
    assert fix_outliers(df, 'c') is not None
    assert fix_nans(df, 'c') is not None
    assert normalize_column(df.loc[:, 'a']) is not None
    assert standardize_column(df.loc[:, 'a']) is not None
    assert calculate_numeric_distance(df.loc[:, 'a'], df.loc[:, 'a'], DistanceMetric.EUCLIDEAN) is not None
    assert calculate_numeric_distance(df.loc[:, 'a'], df.loc[:, 'a'], DistanceMetric.MANHATTAN) is not None
    assert calculate_binary_distance(df.loc[:, 'b'], df.loc[:, 'b']) is not None
    print("ok")
