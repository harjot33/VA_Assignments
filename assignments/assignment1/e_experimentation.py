import collections
import itertools
from pathlib import Path
from typing import Union, Optional
from enum import Enum

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from assignments.assignment1.b_data_profile import *
from assignments.assignment1.c_data_cleaning import *
from assignments.assignment1.d_data_encoding import *
from assignments.assignment1.a_load_file import read_dataset


##############################################
# Example(s). Read the comments in the following method(s)
##############################################

##############################################
# Example(s). Read the comments in the following method(s)
##############################################


def process_iris_dataset() -> pd.DataFrame:
    """
    In this example, I call the methods you should have implemented in the other files
    to read and preprocess the iris dataset. This dataset is simple, and only has 5 columns:
    four numeric and one categorical. Depending on what I want to do in the future, I may want
    to transform these columns in other things (for example, I could transform a numeric column
    into a categorical one by splitting the number into bins, similar to how a histogram creates bins
    to be shown as a bar chart).

    In my case, what I want to do is to *remove missing numbers*, replacing them with valid ones,
    and *delete outliers* rows altogether (I could have decided to do something else, and this decision
    will be on you depending on what you'll do with the data afterwords, e.g. what machine learning
    algorithm you'll use). I will also standardize the numeric columns, create a new column with the average
    distance between the three numeric column and convert the categorical column to a onehot-encoding format.

    :return: A dataframe with no missing values, no outliers and onehotencoded categorical columns
    """
    df = read_dataset(Path('..', '..', 'iris.csv'))
    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_text_categorical_columns(df)

    for nc in numeric_columns:
        df = fix_outliers(df, nc)
        df = fix_nans(df, nc)
        df.loc[:, nc] = standardize_column(df.loc[:, nc])

    distances = pd.DataFrame()
    for nc_combination in list(itertools.combinations(numeric_columns, 2)):
        distances[str(nc_combination)] = calculate_numeric_distance(df.loc[:, nc_combination[0]],
                                                                    df.loc[:, nc_combination[1]],
                                                                    DistanceMetric.EUCLIDEAN).values
    df['numeric_mean'] = distances.mean(axis=1)

    for cc in categorical_columns:
        ohe = generate_one_hot_encoder(df.loc[:, cc])
        df = replace_with_one_hot_encoder(df, cc, ohe, list(ohe.get_feature_names()))

    return df


##############################################
# Implement all the below methods
# All methods should be dataset-independent, using only the methods done in the assignment
# so far and pandas/numpy/sklearn for the operations
##############################################
def process_iris_dataset_again() -> pd.DataFrame:
    """
    Consider the example above and once again perform a preprocessing and cleaning of the iris dataset.
    This time, use normalization for the numeric columns and use label_encoder for the categorical column.
    Also, for this example, consider that all petal_widths should be between 0.0 and 1.0, replace the wong_values
    of that column with the mean of that column. Also include a new (binary) column called "large_sepal_lenght"
    saying whether the row's sepal_length is larger (true) or not (false) than 5.0
    :return: A dataframe with the above conditions.
    """
    # Reading the iris dataset again
    df = read_dataset(Path(r'H:\New folder (2)\Assignments (3)\Assignments', 'iris.csv'))
    # Performing the data preprocessing steps -

    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_text_categorical_columns(df)

    # Firstly, nans and outliers are fixed, then we normalize all the numerical columns
    for numeric_column in numeric_columns:
        df = fix_nans(df, numeric_column)
        df = fix_outliers(df, numeric_column)
        df.loc[:, numeric_column] = normalize_column(df.loc[:, numeric_column])

    # Now the nans and the outliers of the categorical columns are fixed, then we encode the values of the categorical
    # columns using the label encoder, by firstly, generating it and then we encode the values and the encoded dataframe is returned.
    for categorical_column in categorical_columns:
        df = fix_nans(df, categorical_column)
        df = fix_outliers(df, categorical_column)
        labelencoder = generate_label_encoder(df.loc[:, categorical_column])
        df = replace_with_label_encoder(df, categorical_column, labelencoder)

        # Creating a new column which has binary values for the corresponding boolean condition
        df['large_sepal_length'] = df["sepal_length"] > 5.0

        # Replacing the wrong values with the nans (by default the fix_numeric_wrong values) function replaces it with the nan
        column = 'petal_width'
        must_be_rule_greater = WrongValueNumericRule.MUST_BE_GREATER_THAN
        optional_parameter = 0

        df = fix_numeric_wrong_values(df, column, must_be_rule_greater, optional_parameter)

        column = 'petal_width'
        must_be_rule_less = WrongValueNumericRule.MUST_BE_LESS_THAN
        optional_parameter = 1
        df = fix_numeric_wrong_values(df, column, must_be_rule_less, optional_parameter)

        # Replacing the nan values with the mean of the column
        df[column] = df[column].fillna(get_column_mean(df, column))

        print(df)
        return df


def process_amazon_video_game_dataset():
    """
    Now use the rating_Video_Games dataset following these rules:
    1. The rating has to be between 1.0 and 5.0
    2. Time should be converted from milliseconds to datetime.datetime format
    3. For the future use of this data, I don't care about who voted what, I only want the average rating per product,
        therefore replace the user column by counting how many ratings each product had (which should be a column called count),
        and the average rating (as the "review" column).
    :return: A dataframe with the above conditions. The columns at the end should be: asin,review,time,count
    """
    # Reading the amazon video game dataset again
    df = read_dataset(Path(r'H:\New folder (2)\Assignments (3)\Assignments', 'ratings_Video_Games.csv'))

    # 1. Keeping only those reviews whose rating is between 5 and 1
    df = df[(df['review'] <= 5) & (df['review'] >= 1)]

    # 2. Converting the milliseconds to date.datetime format
    df['time'] = pd.to_datetime(df['time'], unit='ms')

    # 3 Grouping by using the asin value, then the review of those asin values are showed as an average as a column
    # The number of users are then counted which creates the count column

    # Read about asin values from here
    # https://www.datafeedwatch.com/blog/amazon-asin-number-what-is-it-and-how-do-you-get-it#:~:text=Amazon%20Standard%20Identification%20Number%20(ASIN,new%20product%20in%20Amazon's%20catalog.

    df = df.groupby(by='asin', as_index=False).agg(
        {'review': np.mean, 'time': np.min, 'user': np.count_nonzero})
    df = df.rename(columns={'user': 'count'})

    return df


def process_amazon_video_game_dataset_again():
    """
    Now use the rating_Video_Games dataset following these rules (the third rule changed, and is more open-ended):
    1. The rating has to be between 1.0 and 5.0, drop any rows not following this rule
    2. Time should be converted from milliseconds to datetime.datetime format
    3. For the future use of this data, I just want to know more about the users, therefore show me how many reviews each user has,
        and a statistical analysis of each user (average, median, std, etc..., each as its own row)
    :return: A dataframe with the above conditions.
    """
    # Reading the amazon video game dataset
    df = read_dataset(Path(r'C:\Users\AVuser\Downloads\assignments\assignments', 'ratings_Video_Games.csv'))
    # 1. Keeping only those reviews whose rating is between 5 and 1
    df = df[(df['review'] <= 5) & (df['review'] >= 1)]

    # 2. Converting the milliseconds to date.datetime format
    df['time'] = pd.to_datetime(df['time'], unit='ms')

    # 3. Grouping by the user, and then finding the insights about the users, such that using numpy's statistical
    # analysis methods we figure out the mean, variance, average, standard deviation and median, also display the time as a column
    df = df.groupby(by='user', as_index=False).agg({'asin': np.count_nonzero,
                                                    'review': [np.count_nonzero, np.mean, np.var, np.average, np.std,
                                                               np.median], 'time': np.min})

    return df


def process_life_expectancy_dataset():
    """
    Now use the life_expectancy_years and geography datasets following these rules:
    1. The life expectancy dataset has missing values and outliers. Fix them.
    2. The geography dataset has problems with unicode letters. Make sure your code is handling it properly.
    3. Change the format of life expectancy, so that instead of one row with all 28 years, the data has 28 rows, one for each year,
        and with a column "year" with the year and a column "value" with the original value
    4. Merge (or more specifically, join) the two datasets with the common column being the country name (be careful with wrong values here)
    5. Drop all columns except country, continent, year, value and latitude (in this hypothetical example, we wish to analyse differences
        between southern and northern hemisphere)
    6. Change the latitude column from numerical to categorical (north vs south) and pass it though a label_encoder
    7. Change the continent column to a one_hot_encoder version of it
    :return: A dataframe with the above conditions.
    """
    # Reading the amazon video game dataset again
    df = read_dataset(Path(r'H:\New folder (2)\Assignments (3)\Assignments', 'life_expectancy_years.csv'))

    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_text_categorical_columns(df)

    # 1. Fixing the data loaded from the life expectancy dataset.
    # Firstly, nans and outliers are fixed, then we normalize all the numerical columns
    for numeric_column in numeric_columns:
        df = fix_nans(df, numeric_column)
        df = fix_outliers(df, numeric_column)

    # Now the nans and the outliers of the categorical columns are fixed, then we encode the values of the categorical
    # columns using the label encoder, by firstly, generating it and then we encode the values and the encoded dataframe is returned.
    for categorical_column in categorical_columns:
        df = fix_nans(df, categorical_column)
        df = fix_outliers(df, categorical_column)

    # 2. Reading the unicode data in the utf-8 format
    df2 = pd.read_csv(r'H:\New folder (2)\Assignments (3)\Assignments\geography.csv', encoding='utf-8')
    df2 = df2.rename(columns={'name': 'country'})

    # Handling the missing data and the outliers.
    numeric_columns = get_numeric_columns(df2)
    categorical_columns = get_text_categorical_columns(df2)

    # Firstly, nans and outliers are fixed, then we normalize all the numerical columns
    for numeric_column in numeric_columns:
        df2 = fix_nans(df2, numeric_column)
        df2 = fix_outliers(df2, numeric_column)

    # Now the nans and the outliers of the categorical columns are fixed, then we encode the values of the categorical
    # columns using the label encoder, by firstly, generating it and then we encode the values and the encoded dataframe is returned.
    for categorical_column in categorical_columns:
        df2 = fix_nans(df2, categorical_column)
        df2 = fix_outliers(df2, categorical_column)

    # Transposing the dataframe into the matrix
    df = df.transpose()
    df = df.reset_index(level=0, inplace=False)

    df = df.rename(columns=df.iloc[0]).drop(df.index[0])
    # Getting the country column value then using it as a column name and the corresponding column headers then become
    # the values of this column.
    # https://www.journaldev.com/33398/pandas-melt-unmelt-pivot-function

    df_melted = pd.melt(df, id_vars=["country"])

    # Renaming the new dataframe, as the column headers are incorrect, we need to change them to correctly identify them.

    df_melted.rename(columns={'country': 'year', 'variable': 'country'}, inplace=True)

    # print(df_melted)

    # Merging the two datasets based on the common column 'country'

    # https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html#database-style-dataframe-or-named-series-joining-merging
    combined_df = pd.merge(left=df_melted, right=df2, left_on='country', right_on='country')

    # 5. Dropping all the columns with
    combined_df = combined_df[['country', 'four_regions', 'year', 'value', 'Latitude']]
    combined_df.rename(columns={'four_regions': 'continent'}, inplace=True)

    column = 'Latitude'
    combined_df.loc[combined_df[column] >= 0, column] = 0
    combined_df.loc[combined_df[column] < 0, column] = 1

    # 6 Changing the numerical data into categorical and then I passed it through the label encoder.
    combined_df[column] = np.where(combined_df[column] == 0, 'north', 'south')

    print(combined_df)
    LabelEncoder = generate_label_encoder(combined_df['Latitude'])
    df_labelEncode = replace_with_label_encoder(combined_df, column, LabelEncoder)

    # 7 One Hot Encoding of the Continent Column
    OneHotEncoder = generate_one_hot_encoder(df_labelEncode['continent'])
    column_names = list(OneHotEncoder.get_feature_names())
    df_final = replace_with_one_hot_encoder(df_labelEncode, 'continent', OneHotEncoder, column_names)

    return df_final


if __name__ == "__main__":
    assert process_iris_dataset() is not None
    assert process_iris_dataset_again() is not None
    assert process_amazon_video_game_dataset() is not None
    assert process_amazon_video_game_dataset_again() is not None
    assert process_life_expectancy_dataset() is not None
