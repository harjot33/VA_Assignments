#Import all the necessary dependables
import calendar
from pathlib import Path
import dash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Output, Input, State
from matplotlib.widgets import Button, Slider
import dash_core_components as dcc
from dash import html as html
import dash_bootstrap_components as dbc

from assignments.assignment1.a_load_file import read_dataset
from assignments.assignment1.b_data_profile import get_numeric_columns
from assignments.assignment1.c_data_cleaning import fix_nans, fix_outliers
from assignments.assignment2.c_clustering import simple_k_means


def task():
    """
    For the last assignment, there is only one task, which will use your knowledge from all previous assignments.
    If your methods of a1, a2 and a3 are well implemented, a4 will be fairly simple, so reuse the methods if possible for your own
    benefit! If you prefer, you can reimplement any logic you with in the assignment4 folder.

    For this task, feel free to create new files, modules or methods in this file as you see fit. Our test will be done by calling this
    task() method, and we expect to receive the dash app back (similar to a3) and we will run it. No other method will be called by us, so
    make sure your code is running as expected. We will basically run this code: `task().run_server(debug=True)`

    For this task, you will build a dash app where we can perform a simple form of interactivity on it. We will use the accidents.csv
    dataset. This accidents.csv dataset has information about traffic accidents in the UK, and is available to you now.
    You will show the most accident prone areas in the UK on a map, so the regular value shown in the map should let us know the number of accidents
    that occurred in a specific area of the map (which should include the accident's severity as well as a weight factor). That said, the purpose of
    this map is to be used by the police to identify how they can do things better.

    **This is raw data, so preprocess the data as per requirement. Drop columns that you feel are unnecessary for the analysis or clustering. 
    Don't forget to add comments about why you did what you did**

    
    ##############################################################
    # Your task will be to Implement all the below functionalities
    ##############################################################

    1. (40pts) Implement a map with the GPS points for the accidents per month. Have a slider(#slider1) that can be used to filter accident data for the month I need.
        You are free to choose a map style, but I suggest using a scatter plot map.

    2. (10pts) Implement a dropdown to select few of the numerical columns in the dataset that can be used meaningfully to represent the size of the GPS points. 
        By default the size of the GPS point on map should be based on the value of "accident_severity" column.

    3. (30pts) You have to Cluster the points now. Be sure to have a text somewhere on the webpage that says what clustering algorithm you are using (e.g. KMeans, dbscan, etc).
        For clustering, you should run a clustering method over the dataset (it should run fairly fast for quick iteration, so make sure to use a simple clustering procedure)
        **COLOR** the GPS points based on the clusetring label returned from the algorithm.

    4. (10pts) Have a button(#run_clustering) to run or rerun the clustering algorithm over the filtered dataset (filtered using #slider1 to select months).

    5. (10pts) At least one parameter of the clustering algorithm should be made available for us to tinker with as a button/slider/dropdown. 
        When I change it and click #run_clustering button, your code should rerun the clustering algorithm. 
        example: change the number of clusters in kmeans or eps value in dbscan.

        Please note: distance values should be in meters, for example dbscan uses eps as a parameter. This value should be read in mts from users and converted appropriately to be used in clustering, 
        so input_eps=100 should mean algorithm uses 100mts circle for finding core and non-core points. 
  
    The total points is 100pts
    """
    # Firstly read the dataset using the previous assignment's function
    df = read_dataset(Path(r'H:\New folder (2)\Assignments (3)\Assignments\assignments\assignment4', 'accidents.csv'))
    # Dropped the following columns
    # Reasons for dropping these columns
    # accident_index - The index of the accident doesn't have any impact, it is just there for identification purposes
    # accident_year - This information was being given by the date column, so that is why it was dropped
    # location_easting_osgr & location_northing_osgr - These location columns were of no need, since I was using longitude and latitude values
    # time - The time of the accident has no use in the operations that were to be performed
    # local_authority_ons_district - This column has stastical information used by the office of national statistics in UK so no usage of this.
    # local_authority_highway - Dropped this column since it refers to the local authority on that highway, I haven't inferred anything from this so it was wise to drop it.
    # lsoa_of_accident_location - This refers to the location code that the UK uses, I already am using the latitude and longitude values for this, so it was redundant, hence, dropped.

    df.drop(['accident_index', 'accident_year','accident_reference','location_easting_osgr','location_northing_osgr','day_of_week','time','local_authority_ons_district','local_authority_highway','lsoa_of_accident_location'], axis=1, inplace=True)

    months = {}

    # Getting the names of the months in a dictionary for later use.
    year = 2008
    for i in range(0,12):
        months[i]=calendar.month_name[i+1]

    numeric_columns = get_numeric_columns(df)

    # Numerical Columns being cleaned and preprocessed.
    for numeric_column in numeric_columns:
        df = fix_nans(df, numeric_column)

        datasetoptions = []
        for col in df.columns:
            datasetoptions.append({'label': col, 'value': col})

    #Defining and Setting the dash application parameters.
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    # Main Layout of the Application
    app.layout = dbc.Container([
        html.Title("My Dash Web App"),
        html.H1(children="Assignment - 4 | Harjot Singh"),
        html.Div(children='Various Parts of the Assignment are as follows'),
        html.Hr(),
        # Creating the Slider for the Task - 1 - The slider has limits 0 - 11 Signifying all the months of the year.
        dbc.FormGroup([
            dbc.Label(id='slider-value'),
            dcc.Slider(id="slider", min=0, max=11, step=1, value=0,    marks=months),
        ]),

        # This contains the graph that will be displayed on the dash application. - Task 1
        dbc.Row([
            dbc.Col(dcc.Graph(id='firstvisualization')),
        ]),
        #This dropdown refers to the tasks asked in the Task 2 - This allows the user to choose the column which would represent the size
        # of the GPS Point
        dbc.FormGroup([
            dbc.Label('Choose the GPS Points representer'),
            dcc.Dropdown(id='pointdrop',
                         value='accident_severity',
                         options=[{'label': 'Accident Severity', 'value': 'accident_severity'},
                                  {'label': 'Number of Casualties', 'value': 'number_of_casualties'},
                                  {'label': 'Number of Vehicles', 'value': 'number_of_vehicles'}
                                  ],clearable=False,
                         searchable=False),


        ]),
        # This card gives information about the clustering algorithm that is being used - in my case it was K Means Clustering
        html.Div([
            dbc.Card(
                dbc.CardBody([
                    html.H4("Clustering Algorithm Being Used - K-Means Clustering"),
                ]),
                style={"width": "18rem"},
            ),
        ]),
        # This dropdown allows the user to choose the number of clusters that will be fed to the K-Means Cluster Algorithm - Task 5
        dbc.FormGroup([
            dbc.Label('Choose the K-Means Clusters'),
            dcc.Dropdown(id='clusters',
                         value=2,
                         options=[
                                  {'label': '2', 'value': 2},
                                  {'label': '3', 'value': 3},
                                  {'label': '4', 'value': 4},
                                  {'label': '5', 'value': 5},
                                  {'label': '6', 'value': 6}
                                  ],clearable=False,
                         searchable=False),


        ]),
        # This button is used to run the clustering algorithm - Task 4
        dbc.Button('Run K-Clustering', id='clusteringbutton', color='primary', style={'margin-bottom': '1em'}, block=True,n_clicks=0),

    ])

    # The application callbacks - The input of this application are the slider input, GPS Point drop down, number of clusters and the clustering button.
    # The output of the callback is the graph firstvisualization whose type is of figure.
    @app.callback(
        Output(component_id='firstvisualization', component_property='figure'),
        [State(component_id='slider', component_property='value'),
         State(component_id='pointdrop', component_property='value'),
         State(component_id='clusters', component_property='value')],
        [Input(component_id='clusteringbutton', component_property='n_clicks')]
    )
    #This function is automatically called when there is a callback - it takes the sliderval, pointval, clustersno, buttonclicks as the parameters
    def first(sliderval, pointval,clustersno,buttonclicks):
        flag = False

        # This if statement filters the data based on the slider value chosen by the user - Month wise.
        if sliderval == 0:
            df["date"]= pd.to_datetime(df["date"])
            filterdf = df.copy()
            filterdf = df[(df['date'] >= '2020-01-01') & (df['date'] <= '2020-01-31')]

            if buttonclicks != 0: # This statement checks if the user has clicked on the clustering button, if they have then it performs clustering
                # after the clustering is performed the color of the gps points is assigned as the cluster labels.
                filterdfc = filterdf.drop(["date"], axis=1)
                model = simple_k_means(filterdfc,clustersno,'euclidean')
                ok = model["model"].labels_
                filterdf.is_copy = False
                filterdf['Cluster Labels'] = ok
                flag = True

            if flag:
                color = filterdf['Cluster Labels']
            else:
                color = pointval

            # Setting the access token.
            px.set_mapbox_access_token('pk.eyJ1IjoiaGFyam90MzMiLCJhIjoiY2t3bWF0eGoyMmFqZDJwbzR1Z3J4aTkwNiJ9.g77qYjUkTeDa-is1jDh0Lg')
            # The scatter mapbox having the latitude and latitude of the accident location, along with the color and size.
            fig = px.scatter_mapbox(filterdf, lat="latitude", lon="longitude", color=color, size=pointval,
                                    color_continuous_scale=px.colors.cyclical.Edge, size_max=10, zoom=7)

            return fig
        elif sliderval == 1:
            df["date"]= pd.to_datetime(df["date"])
            filterdf = df.copy()
            filterdf = df[(df['date'] >= '2020-02-01') & (df['date'] <= '2020-02-29')]
            if buttonclicks != 0:
                # This statement checks if the user has clicked on the clustering button, if they have then it performs clustering
                # after the clustering is performed the color of the gps points is assigned as the cluster labels.
                filterdfc = filterdf.drop(["date"], axis=1)
                model = simple_k_means(filterdfc,clustersno,'euclidean')
                ok = model["model"].labels_
                filterdf.is_copy = False
                filterdf['Cluster Labels'] = ok
                flag = True

            if flag:
                color = filterdf['Cluster Labels']
            else:
                color = pointval
            px.set_mapbox_access_token('pk.eyJ1IjoiaGFyam90MzMiLCJhIjoiY2t3bWF0eGoyMmFqZDJwbzR1Z3J4aTkwNiJ9.g77qYjUkTeDa-is1jDh0Lg')
            # The scatter mapbox having the latitude and latitude of the accident location, along with the color and size.

            fig = px.scatter_mapbox(filterdf, lat="latitude", lon="longitude", color=color, size=pointval,
                                    color_continuous_scale=px.colors.cyclical.IceFire, size_max=10, zoom=7)

            return fig
        elif sliderval == 2:
            df["date"]= pd.to_datetime(df["date"])
            filterdf = df.copy()
            filterdf = df[(df['date'] >= '2020-03-01') & (df['date'] <= '2020-03-31')]
            if buttonclicks != 0:
                # This statement checks if the user has clicked on the clustering button, if they have then it performs clustering
                # after the clustering is performed the color of the gps points is assigned as the cluster labels.
                filterdfc = filterdf.drop(["date"], axis=1)
                model = simple_k_means(filterdfc,clustersno,'euclidean')
                ok = model["model"].labels_
                filterdf.is_copy = False
                filterdf['Cluster Labels'] = ok
                flag = True

            if flag:
                color = filterdf['Cluster Labels']
            else:
                color = pointval
            px.set_mapbox_access_token('pk.eyJ1IjoiaGFyam90MzMiLCJhIjoiY2t3bWF0eGoyMmFqZDJwbzR1Z3J4aTkwNiJ9.g77qYjUkTeDa-is1jDh0Lg')
            # The scatter mapbox having the latitude and latitude of the accident location, along with the color and size.
            fig = px.scatter_mapbox(filterdf, lat="latitude", lon="longitude", color=color, size=pointval,
                                    color_continuous_scale=px.colors.cyclical.IceFire, size_max=10, zoom=7)
            return fig
        elif sliderval == 3:
            df["date"]= pd.to_datetime(df["date"])
            filterdf = df.copy()
            filterdf = df[(df['date'] >= '2020-04-01') & (df['date'] <= '2020-04-30')]
            if buttonclicks != 0:
                # This statement checks if the user has clicked on the clustering button, if they have then it performs clustering
                # after the clustering is performed the color of the gps points is assigned as the cluster labels.
                filterdfc = filterdf.drop(["date"], axis=1)
                model = simple_k_means(filterdfc,clustersno,'euclidean')
                ok = model["model"].labels_
                filterdf.is_copy = False
                filterdf['Cluster Labels'] = ok
                flag = True

            if flag:
                color = filterdf['Cluster Labels']
            else:
                color = pointval
            px.set_mapbox_access_token('pk.eyJ1IjoiaGFyam90MzMiLCJhIjoiY2t3bWF0eGoyMmFqZDJwbzR1Z3J4aTkwNiJ9.g77qYjUkTeDa-is1jDh0Lg')
            # The scatter mapbox having the latitude and latitude of the accident location, along with the color and size.
            fig = px.scatter_mapbox(filterdf, lat="latitude", lon="longitude", color=color, size=pointval,
                                    color_continuous_scale=px.colors.cyclical.IceFire, size_max=10, zoom=7)
            return fig
        elif sliderval == 4:
            df["date"]= pd.to_datetime(df["date"])
            filterdf = df.copy()
            filterdf = df[(df['date'] >= '2020-05-01') & (df['date'] <= '2020-05-31')]
            if buttonclicks != 0:
                # This statement checks if the user has clicked on the clustering button, if they have then it performs clustering
                # after the clustering is performed the color of the gps points is assigned as the cluster labels.
                filterdfc = filterdf.drop(["date"], axis=1)
                model = simple_k_means(filterdfc,clustersno,'euclidean')
                ok = model["model"].labels_
                filterdf.is_copy = False
                filterdf['Cluster Labels'] = ok
                flag = True

            if flag:
                color = filterdf['Cluster Labels']
            else:
                color = pointval
            px.set_mapbox_access_token('pk.eyJ1IjoiaGFyam90MzMiLCJhIjoiY2t3bWF0eGoyMmFqZDJwbzR1Z3J4aTkwNiJ9.g77qYjUkTeDa-is1jDh0Lg')
            # The scatter mapbox having the latitude and latitude of the accident location, along with the color and size.
            fig = px.scatter_mapbox(filterdf, lat="latitude", lon="longitude", color=color, size=pointval,
                                    color_continuous_scale=px.colors.cyclical.IceFire, size_max=10, zoom=7)
            return fig
        elif sliderval == 5:
            df["date"]= pd.to_datetime(df["date"])
            filterdf = df.copy()
            filterdf = df[(df['date'] >= '2020-06-01') & (df['date'] <= '2020-06-30')]
            if buttonclicks != 0:
                # This statement checks if the user has clicked on the clustering button, if they have then it performs clustering
                # after the clustering is performed the color of the gps points is assigned as the cluster labels.
                filterdfc = filterdf.drop(["date"], axis=1)
                model = simple_k_means(filterdfc,clustersno,'euclidean')
                ok = model["model"].labels_
                filterdf.is_copy = False
                filterdf['Cluster Labels'] = ok
                flag = True

            if flag:
                color = filterdf['Cluster Labels']
            else:
                color = pointval
            px.set_mapbox_access_token('pk.eyJ1IjoiaGFyam90MzMiLCJhIjoiY2t3bWF0eGoyMmFqZDJwbzR1Z3J4aTkwNiJ9.g77qYjUkTeDa-is1jDh0Lg')
            # The scatter mapbox having the latitude and latitude of the accident location, along with the color and size.
            fig = px.scatter_mapbox(filterdf, lat="latitude", lon="longitude", color=color, size=pointval,
                                    color_continuous_scale=px.colors.cyclical.IceFire, size_max=10, zoom=7)
            return fig
        elif sliderval == 6:
            df["date"]= pd.to_datetime(df["date"])
            filterdf = df.copy()
            filterdf = df[(df['date'] >= '2020-07-01') & (df['date'] <= '2020-07-31')]
            if buttonclicks != 0:
                # This statement checks if the user has clicked on the clustering button, if they have then it performs clustering
                # after the clustering is performed the color of the gps points is assigned as the cluster labels.
                filterdfc = filterdf.drop(["date"], axis=1)
                model = simple_k_means(filterdfc,clustersno,'euclidean')
                ok = model["model"].labels_
                filterdf.is_copy = False
                filterdf['Cluster Labels'] = ok
                flag = True

            if flag:
                color = filterdf['Cluster Labels']
            else:
                color = pointval
            px.set_mapbox_access_token('pk.eyJ1IjoiaGFyam90MzMiLCJhIjoiY2t3bWF0eGoyMmFqZDJwbzR1Z3J4aTkwNiJ9.g77qYjUkTeDa-is1jDh0Lg')
            # The scatter mapbox having the latitude and latitude of the accident location, along with the color and size.
            fig = px.scatter_mapbox(filterdf, lat="latitude", lon="longitude", color=color, size=pointval,
                                    color_continuous_scale=px.colors.cyclical.IceFire, size_max=10, zoom=7)
            return fig
        elif sliderval == 7:
            df["date"]= pd.to_datetime(df["date"])
            filterdf = df.copy()
            filterdf = df[(df['date'] >= '2020-08-01') & (df['date'] <= '2020-08-31')]
            if buttonclicks != 0:
                # This statement checks if the user has clicked on the clustering button, if they have then it performs clustering
                # after the clustering is performed the color of the gps points is assigned as the cluster labels.
                filterdfc = filterdf.drop(["date"], axis=1)
                model = simple_k_means(filterdfc,clustersno,'euclidean')
                ok = model["model"].labels_
                filterdf.is_copy = False
                filterdf['Cluster Labels'] = ok
                flag = True

            if flag:
                color = filterdf['Cluster Labels']
            else:
                color = pointval
            px.set_mapbox_access_token('pk.eyJ1IjoiaGFyam90MzMiLCJhIjoiY2t3bWF0eGoyMmFqZDJwbzR1Z3J4aTkwNiJ9.g77qYjUkTeDa-is1jDh0Lg')
            # The scatter mapbox having the latitude and latitude of the accident location, along with the color and size.
            fig = px.scatter_mapbox(filterdf, lat="latitude", lon="longitude", color=color, size=pointval,
                                    color_continuous_scale=px.colors.cyclical.IceFire, size_max=10, zoom=7)
            return fig
        elif sliderval == 8:
            df["date"]= pd.to_datetime(df["date"])
            filterdf = df.copy()
            filterdf = df[(df['date'] >= '2020-09-01') & (df['date'] <= '2020-09-30')]
            if buttonclicks != 0:
                # This statement checks if the user has clicked on the clustering button, if they have then it performs clustering
                # after the clustering is performed the color of the gps points is assigned as the cluster labels.
                filterdfc = filterdf.drop(["date"], axis=1)
                model = simple_k_means(filterdfc,clustersno,'euclidean')
                ok = model["model"].labels_
                filterdf.is_copy = False
                filterdf['Cluster Labels'] = ok
                flag = True

            if flag:
                color = filterdf['Cluster Labels']
            else:
                color = pointval
            px.set_mapbox_access_token('pk.eyJ1IjoiaGFyam90MzMiLCJhIjoiY2t3bWF0eGoyMmFqZDJwbzR1Z3J4aTkwNiJ9.g77qYjUkTeDa-is1jDh0Lg')
            # The scatter mapbox having the latitude and latitude of the accident location, along with the color and size.
            fig = px.scatter_mapbox(filterdf, lat="latitude", lon="longitude", color=color, size=pointval,
                                    color_continuous_scale=px.colors.cyclical.IceFire, size_max=10, zoom=7)
            return fig
        elif sliderval == 9:
            df["date"]= pd.to_datetime(df["date"])
            filterdf = df.copy()
            filterdf = df[(df['date'] >= '2020-10-01') & (df['date'] <= '2020-10-31')]
            if buttonclicks != 0:
                # This statement checks if the user has clicked on the clustering button, if they have then it performs clustering
                # after the clustering is performed the color of the gps points is assigned as the cluster labels.
                filterdfc = filterdf.drop(["date"], axis=1)
                model = simple_k_means(filterdfc,clustersno,'euclidean')
                ok = model["model"].labels_
                filterdf.is_copy = False
                filterdf['Cluster Labels'] = ok
                flag = True

            if flag:
                color = filterdf['Cluster Labels']
            else:
                color = pointval
            px.set_mapbox_access_token('pk.eyJ1IjoiaGFyam90MzMiLCJhIjoiY2t3bWF0eGoyMmFqZDJwbzR1Z3J4aTkwNiJ9.g77qYjUkTeDa-is1jDh0Lg')
            # The scatter mapbox having the latitude and latitude of the accident location, along with the color and size.
            fig = px.scatter_mapbox(filterdf, lat="latitude", lon="longitude", color=color, size=pointval,
                                    color_continuous_scale=px.colors.cyclical.IceFire, size_max=10, zoom=7)
            return fig
        elif sliderval == 10:
            df["date"]= pd.to_datetime(df["date"])
            filterdf = df.copy()
            filterdf = df[(df['date'] >= '2020-11-01') & (df['date'] <= '2020-11-30')]
            if buttonclicks != 0:
                # This statement checks if the user has clicked on the clustering button, if they have then it performs clustering
                # after the clustering is performed the color of the gps points is assigned as the cluster labels.
                filterdfc = filterdf.drop(["date"], axis=1)
                model = simple_k_means(filterdfc,clustersno,'euclidean')
                ok = model["model"].labels_
                filterdf.is_copy = False
                filterdf['Cluster Labels'] = ok
                flag = True

            if flag:
                color = filterdf['Cluster Labels']
            else:
                color = pointval
            px.set_mapbox_access_token('pk.eyJ1IjoiaGFyam90MzMiLCJhIjoiY2t3bWF0eGoyMmFqZDJwbzR1Z3J4aTkwNiJ9.g77qYjUkTeDa-is1jDh0Lg')
            # The scatter mapbox having the latitude and latitude of the accident location, along with the color and size.
            fig = px.scatter_mapbox(filterdf, lat="latitude", lon="longitude", color=color, size=pointval,
                                    color_continuous_scale=px.colors.cyclical.IceFire, size_max=10, zoom=7)
            return fig
        elif sliderval == 11:
            df["date"]= pd.to_datetime(df["date"])
            filterdf = df.copy()
            filterdf = df[(df['date'] >= '2020-12-01') & (df['date'] <= '2020-12-31')]
            if buttonclicks != 0:
                # This statement checks if the user has clicked on the clustering button, if they have then it performs clustering
                # after the clustering is performed the color of the gps points is assigned as the cluster labels.
                filterdfc = filterdf.drop(["date"], axis=1)
                model = simple_k_means(filterdfc,clustersno,'euclidean')
                ok = model["model"].labels_
                filterdf.is_copy = False
                filterdf['Cluster Labels'] = ok
                flag = True

            if flag:
                color = filterdf['Cluster Labels']
            else:
                color = pointval
            px.set_mapbox_access_token('pk.eyJ1IjoiaGFyam90MzMiLCJhIjoiY2t3bWF0eGoyMmFqZDJwbzR1Z3J4aTkwNiJ9.g77qYjUkTeDa-is1jDh0Lg')
            # The scatter mapbox having the latitude and latitude of the accident location, along with the color and size.
            fig = px.scatter_mapbox(filterdf, lat="latitude", lon="longitude", color=color, size=pointval,
                                    color_continuous_scale=px.colors.cyclical.IceFire, size_max=10, zoom=7)
            return fig

    # Returning the Application
    return app
if __name__ == "__main__":
    app_t = task()
    app_t.run_server(debug=True)
