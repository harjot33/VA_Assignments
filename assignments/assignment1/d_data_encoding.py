import collections
from pathlib import Path
from typing import Union, Optional
from enum import Enum

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from assignments.assignment1.b_data_profile import *
from assignments.assignment1.c_data_cleaning import *
from assignments.assignment1.a_load_file import read_dataset


##############################################
# Example(s). Read the comments in the following method(s)
##############################################


##############################################
# Implement all the below methods
# All methods should be dataset-independent, using only the methods done in the assignment
# so far and pandas/numpy/sklearn for the operations
##############################################
def generate_label_encoder(df_column: pd.Series) -> LabelEncoder:
    """
    This method should generate a (sklearn version of a) label encoder of a categorical column
    :param df_column: Dataset's column
    :return: A label encoder of the column
    """
    # Generating a Label Encoder using sklearn, then fitting the encoder onto the pandas Series

    return LabelEncoder().fit(df_column)


def generate_one_hot_encoder(df_column: pd.Series) -> OneHotEncoder:
    """
    This method should generate a (sklearn version of a) one hot encoder of a categorical column
    :param df_column: Dataset's column
    :return: A label encoder of the column
    """
    onehotencoder = OneHotEncoder(handle_unknown='ignore')
    # reshape the 1-D country array to 2-D as fit_transform expects 2-D and finally fit the object
    df_column_onehotencoder = onehotencoder.fit(df_column.values.reshape(-1, 1))
    return df_column_onehotencoder


def replace_with_label_encoder(df: pd.DataFrame, column: str, le: LabelEncoder) -> pd.DataFrame:
    """
    This method should replace the column of df with the label encoder's version of the column
    :param df: Dataset
    :param column: column to be replaced
    :param le: the label encoder to be used to replace the column
    :return: The df with the column replaced with the one from label encoder
    """
    # Replacing the column with the label encoder's encoded version
    df_new = df.copy()
    df_new[column] = le.transform(df_new[column])
    return df_new


def replace_with_one_hot_encoder(df: pd.DataFrame, column: str, ohe: OneHotEncoder,
                                 ohe_column_names: List[str]) -> pd.DataFrame:
    """
    This method should replace the column of df with all the columns generated from the one hot's version of the encoder
    Feel free to do it manually or through a sklearn ColumnTransformer
    :param df: Dataset
    :param column: column to be replaced
    :param ohe: the one hot encoder to be used to replace the column
    :param ohe_column_names: the names to be used as the one hot encoded's column names
    :return: The df with the column replaced with the one from label encoder
    """
    df_copy = df.copy(deep=True)
    encoded_column = ohe.transform(df_copy[column].values.reshape(-1, 1)).toarray()
    join_data = pd.DataFrame(encoded_column,
                             columns=ohe_column_names)  # creating new data frame which has encoded columns
    df_copy = df_copy.drop([column], axis=1)  # Dropping the original column
    df_copy = pd.concat([df_copy, join_data], axis=1)  # concatenating the encoded column with original data frame
    return df_copy


def replace_label_encoder_with_original_column(df: pd.DataFrame, column: str, le: LabelEncoder) -> pd.DataFrame:
    """
    This method should revert what is done in replace_with_label_encoder
    The column of df should be from the label encoder, and you should use the le to revert the column to the previous state
    :param df: Dataset
    :param column: column to be replaced
    :param le: the label encoder to be used to revert the column
    :return: The df with the column reverted from label encoder
    """

    # Reverted the column using the inverse transformation to its previous state.
    df[column] = le.inverse_transform(df[column])
    return df


def replace_one_hot_encoder_with_original_column(df: pd.DataFrame,
                                                 columns: List[str],
                                                 ohe: OneHotEncoder,
                                                 original_column_name: str) -> pd.DataFrame:
    """
    This method should revert what is done in replace_with_one_hot_encoder
    The columns (one of the method's arguments) are the columns of df that were placed there by the OneHotEncoder.
    You should use the ohe to revert these columns to the previous state (single column) which was present previously
    :param df: Dataset
    :param columns: the one hot encoded columns to be replaced
    :param ohe: the one hot encoder to be used to revert the columns
    :param original_column_name: the original column name which was used before being replaced with the one hot encoded version of it
    :return: The df with the columns reverted from the one hot encoder
    """
    df_copy = df.copy(deep=True)
    decoded_column = ohe.inverse_transform(df_copy[columns].values).squeeze()
    join_data = pd.DataFrame(decoded_column, columns=[original_column_name])        #Creating a data frame of the decoded column
    df_copy = df_copy.drop(columns, axis=1)                                         #Dropping the original encoded columns
    df_copy = pd.concat([df_copy, join_data], axis=1)                               #Concatinating the original column
    return df_copy



if __name__ == "__main__":
    df = pd.DataFrame({'a':[1,2,3,4], 'b': [True, True, False, False], 'c': ['one', 'two', 'three', 'four']})
    df2 = pd.DataFrame({'a':[1,2,3,4], 'b': [True, True, False, False], 'c': ['one', 'two', 'three', 'four']})
    le = generate_label_encoder(df.loc[:, 'c'])
    assert le is not None
    ohe = generate_one_hot_encoder(df.loc[:, 'c'])
    assert ohe is not None
    assert replace_with_label_encoder(df, 'c', le) is not None
    assert replace_with_one_hot_encoder(df, 'c', ohe, list(ohe.get_feature_names())) is not None
    assert replace_label_encoder_with_original_column(replace_with_label_encoder(df2, 'c', le), 'c', le) is not None
    assert replace_one_hot_encoder_with_original_column(replace_with_one_hot_encoder(df2, 'c', ohe, list(ohe.get_feature_names())),
                                                        list(ohe.get_feature_names()),
                                                        ohe,
                                                        'c') is not None
    print("ok")
