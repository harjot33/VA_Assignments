from imports import *
from sklearn.svm import LinearSVR
import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import re
import pandas as pd
import time
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from assignments.assignment2.c_clustering import *
from assignments.assignment2.a_classification import *
from assignments.assignment2.b_regression import *
from assignments.assignment1.a_load_file import *
from assignments.assignment1.b_data_profile import *
from assignments.assignment1.c_data_cleaning import *
from assignments.assignment1.d_data_encoding import *
from assignments.assignment1.e_experimentation import *
"""
Regression is a supervised form of machine learning, which means that it uses labeled data to train a model that 
can predict or forecast an output value for the given input/unseen features.

In this subtask, you'll be training and testing 2 different types of regression models, each with 2 different types of train-test splits and
compare the performance of same for a single different dataset which was downloaded from below given link. 

The dataset contains 36733 instances of 11 sensor measures aggregated over one hour (by means of average or sum) from a gas turbine 
located in Turkey's north western region for the purpose of studying flue gas emissions, namely CO and NOx (NO + NO2).
We will be predicting the "Carbon monoxide (CO)" emissions by this gas turbine using features that represent sensor measurements.
This dataset is split into 5 different files based on year. Read all 5 files and combine them in your code before
running regression algorithms.

Dataset1: "Gas Turbine CO and NOx Emission Data Set Data Set" 
https://archive.ics.uci.edu/ml/datasets/Gas+Turbine+CO+and+NOx+Emission+Data+Set
"""


#######################################################
# Read the comments carefully in the following method(s)
# Assignment questions are marked as "??subtask"
#######################################################


# Use case 1: 80/20 Data split - randomly
# MODEL_1:
def svm_regressor_1(x: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Method to train and test a Linear Support Vector Machine for given dataset.
    Refer the official documentation below to find api parameters and examples on how to train and test your model.
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR
    :param x: data features
    :param y: data outputs
    :return: trained_model and below given performance_scores for test dataset
    """

    # This is the svm_regressor_1 which takes the X as the input and the Y is the ouput labels, we split the dataset into
    # test and training sets using Sklearns inbuilt functionalitiy and then we get the model from calling the model's method and then we
    # fit the model onto the test data, after that we predict and then evaluate the findings the mean square error and mean absolute error.
    # ??subtask1: Split the data into 80%-20% train-test sets.
    #             Randomize the data selection.i.e. train and test data shoud be randomly selected in 80/20 ratio.
    #             Use a random_state=42, so that we can recreate same splitting when run multiple times.
    # ??subtask2: Train your model using train set.
    # ??subtask3: Predict test output value for test set.
    # ??subtask4: Measure the below given performance measures on test predictions.
    # Use methods provided by sklearn to perform train-test split and measure below asked model performance scores.

    # performance measure https://towardsdatascience.com/what-are-the-best-metrics-to-evaluate-your-regression-model-418ca481755b
    # mean_squared_error, mean_absolute_error
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    svm1_model = LinearSVR(verbose=0)
    svm1_model.fit(X_train, y_train)
    ypred = svm1_model.predict(X_test)
    svm1_mse = mean_squared_error(y_test, ypred)
    svm1_mae = mean_absolute_error(y_test, ypred)

    return dict(model=svm1_model, mse=svm1_mse, mae=svm1_mae)


# Use case 1: 80/20 Data split - randomly
# MODEL_2:
def random_forest_regressor_1(x: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Method to train and test a Random Forest regression model for given dataset.
    Refer the official documentation below to find api parameters and examples on how to train and test your model.
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    :param x: data features
    :param y: data outputs
    :return: trained_model and below given performance_scores for test dataset
    """
    # ??subtask1: Split the data into 80%-20% train-test sets.
    #             Randomize the data selection.i.e. train and test data shoud be randomly selected in 80/20 ratio.
    #             Use a random_state=42, so that we can recreate same splitting when run multiple times.
    # ??subtask2: Train your model using train set.
    # ??subtask3: Predict test output value for test set.
    # ??subtask4: Measure the below given performance measures on test predictions.
    # Use methods provided by sklearn to perform train-test split and measure below asked model performance scores.

    # performance measure https://towardsdatascience.com/what-are-the-best-metrics-to-evaluate-your-regression-model-418ca481755b
    # mean_squared_error, mean_absolute_error

    # This is the random_forest_regressor_1 which takes the X as the input and the Y is the ouput labels, we split the dataset into
    # test and training sets using Sklearns inbuilt functionalitiy and then we get the model from calling the model's method and then we
    # fit the model onto the test data, after that we predict and then evaluate the findings the mean square error and mean absolute error.
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    rf1_model = RandomForestRegressor()
    rf1_model.fit(X_train, y_train)
    ypred = rf1_model.predict(X_test)
    rf1_mse = mean_squared_error(y_test, ypred)
    rf1_mae = mean_absolute_error(y_test, ypred)

    return dict(model=rf1_model, mse=rf1_mse, mae=rf1_mae)


# Use Case 1: 80/20 Data split - randomly
def run_regression_models_1(x, y):
    """
    This method takes input features and labels and calls all the functions which trains and tests the above 2 regression models.
    :param x: data features
    :param y: data outputs
    :return: results after training and testing above 2 regression models
    """
    # ??subtask1: Drop the "Year" column from x that was added in "__main__" method during the process of file reading.
    #             It is important to remove this newly added "Year" column before training since this will add bias to data
    #             and is intended to be used only for train-test splitting in use case 2.

    # This is the run_regression_models_1 - Use Case which takes the X as the input and the Y is the ouput labels, we split the dataset into
    # test and training sets using Sklearns inbuilt functionalitiy and then we get the model from calling the model's method and then we
    # fit the model onto the test data, after that we predict and then evaluate the findings the mean square error and mean absolute error.
    # In this we dropped the Year column and then split the data and then performed the steps.
    x = x.drop(["Year"],axis=1)
    r1 = svm_regressor_1(x, y)
    assert len(r1.keys()) == 3
    r2 = random_forest_regressor_1(x, y)
    assert len(r2.keys()) == 3

    return r1, r2


# Use Case 2: Data split based on year
# MODEL_3:
def svm_regressor_2(x: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Method to train and test a Linear Support Vector Machine for given dataset.
    Refer the official documentation below to find api parameters and examples on how to train and test your model.
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR
    :param x: data features
    :param y: data outputs
    :return: trained_model and below given performance_scores for test dataset
    """
    # ??subtask1: Dataset has 5 different files in a folder "pp_gas_emission" ==> ["gt_2011.csv", "gt_2012.csv", "gt_2013.csv", "gt_2014.csv", "gt_2015.csv"]
    #             Read all the files and split the data such that data form ["gt_2011.csv", "gt_2012.csv", "gt_2013.csv", "gt_2014.csv"] is used as train set
    #             and data from ["gt_2015.csv"] is used as test set.
    #             Hint: Use "Year" column which was added in "__main__" method to make this train-test split.
    #                   Sklearn can't be used here. You need to do it using a logic. 
    #                   **Since y is pandas.Series and dosent have "Year" column, use Index of x to split y into y_train/y_test sets**
    # ??subtask2: Drop the "Year" column from x.
    #             It is important to remove this newly added "Year" column before training since this will add bias to data
    #             and is intended to be used only for train-test splitting in use case 2.
    # ??subtask3: Train your model using train set.
    # ??subtask3: Predict test output value for test set.
    # ??subtask5: Measure the below given performance measures on test predictions.
    # Use methods provided by sklearn to measure below asked model performance scores.

    # performance measure https://towardsdatascience.com/what-are-the-best-metrics-to-evaluate-your-regression-model-418ca481755b
    # mean_squared_error, mean_absolute_error
    # This is the svm_regressor_2 - Use Case which takes the X as the input and the Y is the ouput labels, we split the dataset into
    # test and training sets using Sklearns inbuilt functionalitiy and then we get the model from calling the model's method and then we
    # fit the model onto the test data, after that we predict and then evaluate the findings the mean square error and mean absolute error.
    # In this we dropped the Year column and then split the data and then performed the steps.
    # The datasets were combined and then one was used as a test set.
    # Then the training and testing splits were done using logic.
    # Finally all the predicition were done.
    X_train = x.loc[x['Year'] != "2015"]
    y_train = y.iloc[:X_train.shape[0]]
    X_test = x.loc[x['Year'] == "2015"]
    X_train  = X_train.drop(["Year"],axis=1)
    X_test = X_test.drop(["Year"], axis=1)
    y_test = y.iloc[X_train.shape[0]:]
    svm2_model = LinearSVR(verbose=0)
    svm2_model.fit(X_train, y_train)
    ypred = svm2_model.predict(X_test)
    svm2_mse = mean_squared_error(y_test, ypred)
    svm2_mae = mean_absolute_error(y_test, ypred)

    return dict(model=svm2_model, mse=svm2_mse, mae=svm2_mae)


# Use Case 2: Data split based on year
# MODEL_3:
def random_forest_regressor_2(x: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Method to train and test a Linear Support Vector Machine for given dataset.
    Refer the official documentation below to find api parameters and examples on how to train and test your model.
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR
    :param x: data features
    :param y: data outputs
    :return: trained_model and below given performance_scores for test dataset
    """
    # ??subtask1: Dataset has 5 different files in a folder "pp_gas_emission" ==> ["gt_2011.csv", "gt_2012.csv", "gt_2013.csv", "gt_2014.csv", "gt_2015.csv"]
    #             Read all the files and split the data such that data form ["gt_2011.csv", "gt_2012.csv", "gt_2013.csv", "gt_2014.csv"] is used as train set
    #             and data from ["gt_2015.csv"] is used as test set.
    #             Hint: Use "Year" column which was added in "__main__" method to make this train-test split.
    #                   Sklearn can't be used here. You need to do it using a logic.
    # ??subtask2: Drop the "Year" column from x.
    #             It is important to remove this newly added "Year" column before training since this will add bias to data
    #             and is intended to be used only for train-test splitting in use case 2.
    # ??subtask3: Train your model using train set.
    # ??subtask4: Predict test output value for test set.
    # ??subtask5: Measure the below given performance measures on test predictions.
    # Use methods provided by sklearn to measure below asked model performance scores.

    # performance measure https://towardsdatascience.com/what-are-the-best-metrics-to-evaluate-your-regression-model-418ca481755b
    # mean_squared_error, mean_absolute_error

    # This is the random_forest_regressor_2 - Use Case which takes the X as the input and the Y is the ouput labels, we split the dataset into
    # test and training sets using Sklearns inbuilt functionalitiy and then we get the model from calling the model's method and then we
    # fit the model onto the test data, after that we predict and then evaluate the findings the mean square error and mean absolute error.
    # In this we dropped the Year column and then split the data and then performed the steps.
    # The datasets were combined and then one was used as a test set.
    # Then the training and testing splits were done using logic.
    # Finally all the predicition were done.
    X_train = x.loc[x['Year'] != "2015"]
    y_train = y.iloc[:X_train.shape[0]]
    X_test = x.loc[x['Year'] == "2015"]
    X_train = X_train.drop(["Year"], axis=1)
    X_test = X_test.drop(["Year"], axis=1)
    y_test = y.iloc[X_train.shape[0]:]
    rf2_model = RandomForestRegressor()
    rf2_model.fit(X_train, y_train)
    ypred = rf2_model.predict(X_test)
    rf2_mse = mean_squared_error(y_test, ypred)
    rf2_mae = mean_absolute_error(y_test, ypred)

    return dict(model=rf2_model, mse=rf2_mse, mae=rf2_mae)


# Use Case 2: Data split based on year
def run_regression_models_2(x, y):
    """
    This method takes input features and labels and calls all the functions which trains and tests the above 2 regression models.
    No need to do anything here.
    :param x: data features
    :param y: data outputs
    :return: results after training and testing above 2 regression models
    """
    r1 = svm_regressor_2(x, y)
    assert len(r1.keys()) == 3
    r2 = random_forest_regressor_2(x, y)
    assert len(r2.keys()) == 3

    return r1, r2


def run_regression():
    start = time.time()
    print("Regression in progress...")
    # Dataset1:

    # ??subtask1 Read dataset from Folder "pp_gas_emission" which has 5 data files ["gt_2011.csv", "gt_2012.csv", "gt_2013.csv", "gt_2014.csv", "gt_2015.csv"]
    #            and store it in gt_emission_dataset in above line.
    # ??subtask2 Add a new column "Year" based on the file which you are reading.
    # ??subtask3 combine the above datasets from 5 different files into a single pandas dataframe.
    # ??subtask4 Drop "Nitrogen oxides (NOx)" from the dataframe since we will not be predicting the emission of Nitrogen oxides.
    # ??subtask5 Normalize the DataFrame such that all the columns have data in range[0,1]. Use methods from A1 for this.
    datasetfiles = glob.glob(r'H:\New folder (2)\Assignments (3)\Assignments\pp_gas_emission' + "/*.csv")
    li = []
    for filename in datasetfiles:
        match = re.search(r'.*([1-3][0-9]{3})', filename).group(1)
        df = pd.read_csv(filename, index_col=None, header=0)
        if "Year" not in df:
            df["Year"] = match
            li.append(df)
    gt_emission_dataset = pd.concat(li, axis=0, ignore_index=True)
    gt_emission_dataset = gt_emission_dataset.drop(["NOX"], axis=1)
    col_list = gt_emission_dataset.columns.tolist()

    for col in col_list:
        if not gt_emission_dataset[col].eq(gt_emission_dataset[col].iloc[0]).all():
            gt_emission_dataset.loc[:, col] = normalize_column(gt_emission_dataset.loc[:, col])
    output_col = "CO"
    feature_cols = gt_emission_dataset.columns.tolist()
    feature_cols.remove(output_col)
    x_gt_emission = gt_emission_dataset[feature_cols]
    y_gt_emission = gt_emission_dataset[output_col]
    print(y_gt_emission.shape)

    result1, result2 = run_regression_models_1(x_gt_emission, y_gt_emission)
    # Observe both the results and notice which model is preforming better in this use case.
    print(f"{10*'*'}Dataset1{gt_emission_dataset.shape}, usecase1:{10*'*'}\nSVM regressor 1: {result1}\nRandom Forest regressor 1: {result2}\n")

    result1, result2 = run_regression_models_2(x_gt_emission, y_gt_emission)
    # Observe both the results and notice which model is preforming better in this use case.
    print(f"{10*'*'}Dataset1{gt_emission_dataset.shape}, usecase2:{10*'*'}\nSVM regressor 2: {result1}\nRandom Forest regressor 2: {result2}\n")

    end = time.time()
    run_time = round(end - start, 4)
    print("Regression ended...")
    print(f"{30*'-'}\nRegression run_time:{run_time}s\n{30*'-'}\n")


if __name__ == "__main__":
    run_regression()

#Since we are using same parameter names (x: pd.DataFrame, y: pd.Series) in most methods, remember to copy the df passed as parameter 
# and work with the df_copy to avoid warnings and unexpected results. Its a standard practice!
