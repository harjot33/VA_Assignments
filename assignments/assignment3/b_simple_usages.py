from pathlib import Path
from typing import Tuple
from imports import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from textblob import TextBlob

##############################################
# In this file, we will use data and methods of previous assignments with visualization.
# But before you continue on, take some time to look on the internet about the many existing visualization types and their usages, for example:
# https://extremepresentation.typepad.com/blog/2006/09/choosing_a_good.html
# https://datavizcatalogue.com/
# https://plotly.com/python/
# https://www.tableau.com/learn/whitepapers/which-chart-or-graph-is-right-for-you
# Or just google "which visualization to use", and you'll find a near-infinite number of resources
#
# You may want to create a new visualization in the future, and for that I suggest using JavaScript and D3.js, but for the course, we will only
# use python and already available visualizations
##############################################
# Implement all the below methods
# Use the documentation of the libraries in case you need help to understand the required methods
# For ALL methods return the fig and ax of matplotlib or fig from plotly!
##############################################
from assignments.assignment1.a_load_file import read_dataset
from assignments.assignment1.b_data_profile import get_column_max, get_numeric_columns, get_text_categorical_columns, \
    get_binary_columns, get_correlation_between_columns
from assignments.assignment1.c_data_cleaning import fix_nans, fix_outliers
from assignments.assignment1.e_experimentation import process_life_expectancy_dataset
from assignments.assignment2.a_classification import decision_tree_classifier
from assignments.assignment2.c_clustering import cluster_iris_dataset_again
from assignments.assignment3 import a_libraries

def matplotlib_bar_chart() -> Tuple:
    """
    Create a bar chart with a1/b_data_profile's get column max.
    Show the max of each numeric column from iris dataset as the bars
    """

    #Read the dataset
    maxcolarray = []
    df = read_dataset(Path(r'H:\New folder (2)\Assignments (3)\Assignments', 'iris.csv'))
    colist = get_numeric_columns(df)
    for column in colist:
        maxcol = get_column_max(df,column)
        maxcolarray.append(maxcol)


    # Getting the x - converting the list into np array.
    x = np.array(maxcolarray)
    fig, ax = plt.subplots()
    limit = x.shape[0]
    ax.bar(np.arange(0, limit), x)
    ax.set_ylabel('Column Y Label')
    ax.set_xlabel('Column Max Value')
    ax.set_title('Max Columns')
    # plt.show()
    return fig,ax

def matplotlib_pie_chart() -> Tuple:
    """
    Create a pie chart where each piece of the chart has the number of columns which are numeric/categorical/binary
    from the output of a1/e_/process_life_expectancy_dataset
    """

    #Getting the processed dataframe from the previous assignment's function
    # Then getting the lists of all the numerical, categorical and binary columns.
    df = process_life_expectancy_dataset()
    nc = get_numeric_columns(df)
    cc = get_text_categorical_columns(df)
    bc = get_binary_columns(df)
    ncl = len(nc)
    ccl = len(cc)
    bcl = len(bc)
    x = np.array([ncl,ccl,bcl])
    fig, ax = plt.subplots()
    ax.pie(x,autopct='%1.1f%%',
           shadow=True, startangle=90, labels=["Numerical Columns","Categorical Columns","Binary Columns"])
    ax.set_title('Columns Distribution')
   # plt.show()

    return fig,ax


def matplotlib_histogram() -> Tuple:
    """
    Build 4 histograms as subplots in one figure with the numeric values of the iris dataset
    """
    #Reading the dataset
    df = read_dataset(Path(r'H:\New folder (2)\Assignments (3)\Assignments', 'iris.csv'))
    nc = get_numeric_columns(df)
    # Trimming the number of column name in the list to 4
    hiscols = list(nc)[:4]
    # Creating 2x2 subplots
    fig, ax = plt.subplots(2, 2)
    i = 0


    #Now implementing the plots via iteration
    for row in ax:
        for col in row:
            col.set_title('Numerical Values Histogram - '+str(i+1))
            col.set_ylabel('Column Y Label')
            col.set_xlabel('Column Max Value')
            col.hist(df[hiscols[i]].values)
            i = i + 1
    return fig,ax


def matplotlib_heatmap_chart() -> Tuple:
    """
    Remember a1/b_/pandas_profile? There is a heat map over there to analyse the correlation among columns.
    Use the pearson correlation (e.g. https://docs.scipy.org/doc/scipy-1.5.3/reference/generated/scipy.stats.pearsonr.html)
    to calculate the correlation between two numeric columns and show that as a heat map. Use the iris dataset.
    """
    #Reading the dataset
    df = read_dataset(Path(r'H:\New folder (2)\Assignments (3)\Assignments', 'iris.csv'))
    nc = get_numeric_columns(df)
    df = df[nc]
    # Getting the correlation matrix from the pandas corr method
    corrmatrix = df.corr()
    fig, ax = plt.subplots()
    ax.imshow(corrmatrix.values)
    ax.set_title('Correlation Heatmap')
    ax.set_ylabel('Column Y Label')
    ax.set_xlabel('Column Max Value')
   # plt.show()
    return fig, ax

# There are many other possibilities. Please, do check the documentation and examples so you
# may have a good breadth of tools for future work (in assignments, projects, and your own career)
###################################
# Once again, for ALL methods return the fig and ax of matplotlib or fig from plotly!


def plotly_scatter_plot_chart():
    """
    Use the result of a2/c_clustering/cluster_iris_dataset_again() as the color of a scatterplot made from the original (unprocessed)
    iris dataset. Choose among the numeric values to be the x and y coordinates.
    """

    df = read_dataset(Path(r'H:\New folder (2)\Assignments (3)\Assignments', 'iris.csv'))
    clustersiris = cluster_iris_dataset_again()
    #Inputting the clusters column from the processed dataset so that the color would be used to map the data points.
    df['clusters'] = clustersiris['clusters']
    fig = px.scatter(df, x="petal_length", y="petal_width", color="clusters",title="Scatter Plot of Petal Length and Width")
    return fig


def plotly_bar_plot_chart():
    """
    Use the result of a2/c_clustering/cluster_iris_dataset_again() and use x as 3 groups of bars (one for each iris species)
    and each group has multiple bars, one for each cluster, with y as the count of instances in the specific cluster/species combination.
    The grouped bar chart is like https://plotly.com/python/bar-charts/#grouped-bar-chart (search for the grouped bar chart visualization)
    """
    #Reading dataset
    df = read_dataset(Path(r'H:\New folder (2)\Assignments (3)\Assignments', 'iris.csv'))

    clustersiris = cluster_iris_dataset_again()
    df['clusters'] = clustersiris['clusters']

    # Firstly we perform grouping, then the size function is used to count the instances of the group such that following
    # that we use the unstack and stacking operation which pivot the table firstly with inner most column labels and then with the
    # row labels.
    #Code sampled from https://stackoverflow.com/questions/40015666/grouping-a-pandas-dataframe-in-a-suitable-format-for-creating-a-chart
    newdf = df.groupby(["species", "clusters"]).size().unstack(fill_value=0).stack().reset_index()
    newdf['clusters'] = newdf['clusters'].astype(str)
    newdf = newdf.rename({0: 'count'}, axis=1)

    fig = px.bar(newdf, x="species", color="clusters",y="count",barmode='group',title="Number of instances in a specific cluster")

    return fig



def plotly_polar_scatterplot_chart():
    """
    Do something similar to a1/e_/process_life_expectancy_dataset, but don't drop the latitude and longitude.
    Use these two values to figure out the theta to plot values as a compass (example: https://plotly.com/python/polar-chart/).
    Each point should be one country and the radius should be thd value from the dataset (add up all years and feel free to ignore everything else)
    """
    #Getting DF from life expectancy function
    df = process_life_expectancy_dataset()
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

    # Converting Latitude and Longitude into firstly cartesian coordinates.
    # https://stackoverflow.com/questions/1185408/converting-from-longitude-latitude-to-cartesian-coordinates


    combined_df['Latitude'], combined_df['Longitude'] = np.deg2rad(combined_df["Latitude"]), np.deg2rad(combined_df["Longitude"])
    radiusofearth = 6371
    combined_df['x'] = radiusofearth * np.cos(combined_df['Latitude']) * np.cos(combined_df['Longitude'])
    combined_df['y'] = radiusofearth * np.cos(combined_df['Latitude']) * np.sin(combined_df['Longitude'])


    #https://www.w3resource.com/python-exercises/numpy/python-numpy-random-exercise-14.php
    combined_df["theta"] = np.arctan2(combined_df["y"], combined_df["x"])
    fig = px.scatter_polar(combined_df,theta="theta",r="value", color="country", title='Scatter Plot for Life Expectancy Across the Globe')

    return fig


def plotly_table():
    """
    Show the data from a2/a_classification/decision_tree_classifier() as a table
    See https://plotly.com/python/table/ for documentation
    """
    iris = read_dataset(Path(r'H:\New folder (2)\Assignments (3)\Assignments\iris.csv'))
    label_col = "species"
    feature_cols = iris.columns.tolist()
    feature_cols.remove(label_col)
    x_iris = iris[feature_cols]
    y_iris = iris[label_col]
    r1 = decision_tree_classifier(x_iris, y_iris)
    print(r1["confusion_matrix"].shape)
    print(r1["confusion_matrix"])
    df = pd.DataFrame(data=r1["confusion_matrix"], index=["Setosa", "Versicolor", "Virginica"], columns=["Setosa", "Versicolor", "Virginica"])
    table = [df[x] for x in list(df.columns)]
    indexes = df.index.to_list()
    cols = list(df.columns)
    table.insert(0, indexes)
    fig = go.Figure(data=[go.Table(header=dict(values=["index"]+cols),
                                   cells=dict(values=table))])
    fig.update_layout(title_text='Performance Evaluation of Decision Tree Table')
    fig.update_layout({'margin':{'t':50}})
    return fig


def plotly_composite_line_bar():
    """
    Use the data from a1/e_/process_life_expectancy_dataset and show in a single graph on year on x and value on y where
    there are 5 line charts of 5 countries (you choose which) and one bar chart on the background with the total value of all 5
    countries added up.
    """
    df = process_life_expectancy_dataset()

    countries = ["Afghanistan", "Zimbabwe", "Bhutan", "Angola", "Canada"]
    print(df.head(1000))
    df2 = df.query('country in @countries')
    df3 = df2[["country","year","value"]]
    dfbar = df3[["year","value"]]
    dfbar = dfbar.groupby(["year"]).sum().reset_index()
    fig = px.bar(dfbar, x="year", y="value", title="Life Expectancy by Countries")

    for c in countries:
        dfcont = df2[df2["country"]==c]
        fig.add_trace(go.Scatter(x = dfcont['year'], y = dfcont['value'], name=c))

    return fig


def plotly_map():
    """
    Use the data from a1/e_/process_life_expectancy_dataset on a plotly map (anyone will do)
    Examples: https://plotly.com/python/maps/, https://plotly.com/python/choropleth-maps/#using-builtin-country-and-state-geometries
    Use the value from the dataset of a specific year (e.g. 1900) to show as the color in the map
    """
    df = process_life_expectancy_dataset()
    df = df[df["year"] == "1900"]

    fig = px.choropleth(df, locations="country", locationmode="country names", color="value",
                    hover_name="country", color_continuous_scale = px.colors.sequential.Viridis, title="Life Expectancy in Year 1900")

    return fig


def plotly_tree_map():
    """
    Use plotly's treemap to plot any data returned from any of a1/e_experimentation or a2 tasks
    Documentation: https://plotly.com/python/treemaps/
    """
    df = process_life_expectancy_dataset()
    treepath = df.groupby(["country","year","value"]).sum()
    treepath = treepath.reset_index()
    fig = px.treemap(treepath, path=['country', 'year'], values='value', title="Countries and their respective life expectancy in years")
    return fig


if __name__ == "__main__":
    # Here are examples of how to run each method
    # Although you may remove or comment them out for testing, make sure they are present
    # and work when submitting, since when we run your code, all methods will be run like this.
    # If these lines below returns errors when running, your file will be considered to not
    # run, and graded accordingly.



    fig_m_bc, _ = matplotlib_bar_chart()
    fig_m_pc, _ = matplotlib_pie_chart()
    fig_m_h, _ = matplotlib_histogram()
    fig_m_hc, _ = matplotlib_heatmap_chart()

    fig_p_s = plotly_scatter_plot_chart()
    fig_p_bpc = plotly_bar_plot_chart()
    fig_p_psc = plotly_polar_scatterplot_chart()
    fig_p_t = plotly_table()
    fig_p_clb = plotly_composite_line_bar()
    fig_p_map = plotly_map()
    fig_p_treemap = plotly_tree_map()


    # Uncomment the below lines to test your code
    # When submitting, leave the code below commented!!!
    # fig_m_bc.show()
    # fig_m_pc.show()
    # fig_m_h.show()
    # fig_m_hc.show()
    #
    # fig_p_s.show()
    # fig_p_bpc.show()
    # fig_p_psc.show()
    # fig_p_t.show()
    # fig_p_clb.show()
    # fig_p_map.show()
    # fig_p_treemap.show()
