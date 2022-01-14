from pathlib import Path
from typing import Tuple, List

import dash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Output, Input, State
from matplotlib.widgets import Button, Slider
import dash_core_components as dcc
from dash import html as html
import dash_bootstrap_components as dbc

##############################################
# Now let's use dash, a library built on top of flask (a backend framework for python) and plotly
# Check the documentation at https://dash.plotly.com/
# For a complete example, check https://dash-bootstrap-components.opensource.faculty.ai/examples/iris/
# Example(s). Read the comments in the following method(s)
##############################################
from assignments.assignment1.a_load_file import read_dataset
from assignments.assignment1.b_data_profile import get_numeric_columns, get_text_categorical_columns
from assignments.assignment1.c_data_cleaning import fix_nans, fix_outliers
from assignments.assignment1.e_experimentation import process_life_expectancy_dataset, process_amazon_video_game_dataset
from assignments.assignment2.c_clustering import cluster_iris_dataset_again
from assignments.assignment3.b_simple_usages import plotly_bar_plot_chart, plotly_scatter_plot_chart, \
    plotly_polar_scatterplot_chart


def dash_simple_example():
    """
    Here is a simple example from https://dash.plotly.com/layout
    The focus is to create a fig from plotly and add it to dash, but differently from using just plotly, now we can use html elements,
    such as H1 for headers, Div for dividers, and all interations (buttons, sliders, etc).
    Check dash documentation for html and core components.
    """
    app = dash.Dash(__name__)

    df = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })

    # You create a fig just as you did in a_
    fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

    app.layout = html.Div(children=[
        html.H1(children='Hello Dash'),
        html.Div(children='Dash: A web application framework for Python.'),
        dcc.Graph(
            id='example-graph',
            figure=fig  # and include the fig here as a dcc.Graph
        )
    ])

    # To run the app, use app.run_server(debug=True)
    # However, do return the app instead so that the file can be run as in the end of this file
    return app


def dash_with_bootstrap_example():
    """
    Here is a simple example from https://dash.plotly.com/layout
    See examples of components from the bootstrap library at https://dash-bootstrap-components.opensource.faculty.ai/docs/components/alert/
    """
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    df = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })

    fig1 = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
    fig2 = px.line(df, x="Fruit", y="Amount", color="City")

    app.layout = dbc.Container([
        html.H1(children='Hello Dash'),
        html.Hr(),
        dbc.Row([
            dbc.Col(html.Div(children='Dash: A web application framework for Python.'), md=4),
            dbc.Col(dbc.Button('Example Button', color='primary', style={'margin-bottom': '1em'}, block=True), md=8)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='example-graph1', figure=fig1)),
            dbc.Col(dcc.Graph(id='example-graph2', figure=fig2))
        ])
    ])

    # To run the app, use app.run_server(debug=True)
    # However, do return the app instead so that the file can be run as in the end of this file
    return app


def dash_callback_example():
    """
    Here is a more complex example that uses callbacks. With this example, I believe you will suddenly perceive why dash (and webapps)
    are so much better for visual analysis.
    """
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    df = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })

    app.layout = dbc.Container([
        html.H1(children='Hello Dash'),
        html.Div(children='Dash: A web application framework for Python.'),
        html.Hr(),
        dbc.FormGroup([
            dbc.Label("Choose Dataset"),
            dcc.Dropdown(id="dropdown", value=1,
                         options=[{"label": "First Data", "value": 1}, {"label": "Second Data", "value": 2}]),
        ]),
        dbc.FormGroup([
            dbc.Label(id='slider-value'),
            dcc.Slider(id="slider", min=1, max=10, step=0.5, value=1),
        ]),
        dbc.Button('Run Callback', id='example-button', color='primary', style={'margin-bottom': '1em'}, block=True),
        dbc.Row([
            dbc.Col(dcc.Graph(id='example-graph')),
            # Not including fig here because it will be generated with the callback
        ])
    ])
    @app.callback(  # See documentation or tutorial to see how to use this
        Output('example-graph', 'figure'),
        # Outputs is what you wish to update with the callback, which in this case is the figure
        [Input('example-button', 'n_clicks')],
        # Use inputs to define when this callback is called, and read from the values in the inputs as parameters in the method
        [State('dropdown', 'value'),
         # Use states to read values from the interface, but values only in states will not trigger the callback when changed
         State('slider',
               'value')])  # For example, here if you change the slider, this method will not be called, it will only be called when you click the button
    def update_figure(n_clicks, dropdown_value, slider_value):
        df2 = df[:]
        df2.Amount = df2.Amount * slider_value
        if dropdown_value == 1:
            return px.bar(df2, x="Fruit", y="Amount", color="City", barmode="group")
        else:
            return px.line(df2, x="City", y="Amount", color="Fruit")

    @app.callback(Output('slider-value', 'children'), [Input('slider', 'value')])
    def update_slider_value(slider):
        return f'Multiplier: {slider}'

    #  You can also use app.callback to get selection from any of the plotly graphs, including tables and maps, and update anything you wish.
    #  See some examples at https://dash-gallery.plotly.host/Portal/

    return app


##############################################
# Implement all the below methods
# Use the documentation of the libraries in case you need help to understand the required methods
##############################################
#This method gets the dataframe and then creates a list of dictionary with the column name as key and value
def datasetchoices(df: pd.DataFrame) -> List[str]:
    datasetoptions = []
    for col in df.columns:
        datasetoptions.append({'label': col, 'value': col})
    return datasetoptions

#This method takes the dataset choice and based on that assigns the values
def chosendataset(datasetchoice, dfiris, dfgame, dflife):
    if datasetchoice == 'iris':
        x = dfiris
        y = dfiris
        xvalues = dfiris[0]['value']
        yvalues = dfiris[1]['value']
        return x, y, xvalues, yvalues
    elif datasetchoice == 'life':
        x = dflife
        y = dflife
        xvalues = dflife[0]['value']
        yvalues = dflife[1]['value']
        return x, y, xvalues, yvalues
    elif datasetchoice == 'game':
        x = dfgame
        y = dfgame
        xvalues = dfgame[0]['value']
        yvalues = dfgame[1]['value']
        return x, y, xvalues, yvalues


#This method takes the input and then returns a fig having line graph
def linegraph(data, x_data, y_data):
    no_of_rows = str(data.shape[0])
    return px.line(data, x=x_data, y=y_data), no_of_rows

#This method takes the input and then returns a fig having bar graph
def bargraph(data, x_data, y_data):
    no_of_rows = str(data.shape[0])
    return px.bar(data, x=x_data, y=y_data), no_of_rows

#This method takes the input and then returns a fig having composite graph
def compositegraph(df, x_data, y_data):
    no_of_rows = str(df.shape[0])
    fig = px.bar(df, x=x_data, y=y_data)
    fig.add_trace(go.Scatter(x = df[x_data], y = df[y_data]))
    return fig, no_of_rows


def dataupdation(chosendataset, xdata, ydata, chosengraph, dfiris, dfgame, dflife):
    if chosendataset == 'iris':
        if chosengraph == 'Bar Graph':
            return bargraph(dfiris,xdata,ydata)
        elif chosengraph == 'Line Graph':
            return linegraph(dfiris, xdata, ydata)
        elif chosengraph == 'Composite Bar Line Graph':
            return compositegraph(dfiris,xdata,ydata)
    elif chosendataset == 'game':
        if chosengraph == 'Bar Graph':
            return bargraph(dfgame,xdata,ydata)
        elif chosengraph == 'Line Graph':
            return linegraph(dfgame, xdata, ydata)
        elif chosengraph == 'Composite Bar Line Graph':
            return compositegraph(dfgame,xdata,ydata)
    elif chosendataset == 'life':
        if chosengraph == 'Bar Graph':
            return bargraph(dflife,xdata,ydata)
        elif chosengraph == 'Line Graph':
            return linegraph(dflife, xdata, ydata)
        elif chosengraph == 'Composite Bar Line Graph':
            return compositegraph(dflife,xdata,ydata)
    else:
        return bargraph(dfiris,xdata,ydata)
def dash_task():
    """
    There is only only one task to do, a web app with:
    1. Some nice title
    2. One visualization placeholder for dataset visualization
        a. A dropdown to allow me to select which dataset I want to see (iris, video_game and life_expectancy)
        b. Two other dropdowns for me to choose what column to put in x and what column to put in y of the visualization
        c. Another dropdown for me to choose what type of graph I want (see examples in file a_) (at least 3 choices of graphs)
        d. Feel free to change the structure of the dataset if you prefer (e.g. change life_expectancy so that
            there is one column of "year", one for "country" and one for "value")
    4. A https://dash-bootstrap-components.opensource.faculty.ai/docs/components/card/ with the number of rows being showed on the above graph
    5. Another visualization with:
        a. It will containing the figure created in the tasks in a_, b_ or c_ related to plotly's figures
        b. Add a dropdown for me to choose among 3 (or more if you wish) different graphs from a_, b_ or c_ (choose the ones you like)
        c. In this visualization, if I select data in the visualization, update some text in the page (can be a new bootstrap card with text inside)
            with the number of values selected. (see https://dash.plotly.com/interactive-graphing for examples)
    """

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    # Data Handling
    dfiris = read_dataset(Path(r'H:\New folder (2)\Assignments (3)\Assignments', 'iris.csv'))
    numeric_columns = get_numeric_columns(dfiris)
    categorical_columns = get_text_categorical_columns(dfiris)

    #Getting the numerical columns - Preprocessing the data
    for numeric_column in numeric_columns:
        dfiris = fix_nans(dfiris, numeric_column)
        dfiris = fix_outliers(dfiris, numeric_column)

    dfiris = dfiris[numeric_columns]

    dflife = process_life_expectancy_dataset()
    numeric_columns = get_numeric_columns(dflife)
    dflife = dflife[["country", "year", "value"]]

    dfgame = process_amazon_video_game_dataset()
    # Removing the duplicate users
    dfgame = dfgame.drop_duplicates('asin', keep='last')

    #Getting the column lists of the datasets
    dfiriscollist = datasetchoices(dfiris)
    dflifecollist = datasetchoices(dflife)
    dfgamecollist = datasetchoices(dfgame)
    app.layout = dbc.Container([
        # Part One - The first layout of the application  - Contains the definition for the user's first interaction
        # User can choose dataset
        # Choose X AND Y
        # Choose what graph they want to see
        html.Title("My Dash Web App"),
        html.H1(children="Assignment - 3 | Harjot Singh"),
        html.Div(children='Various Parts of the Assignment are as follows'),
        html.Hr(),
        dbc.FormGroup([
            dbc.Label('Choose a dataset of your liking from the following'),
            dcc.Dropdown(id='dataset_dropdown',
                         value='iris',
                         options=[{'label': 'Iris Dataset', 'value': 'iris'},
                                  {'label': 'Prime Video Game Dataset', 'value': 'game'},
                                  {'label': 'Life Expectancy Dataset', 'value': 'life'}
                                  ],clearable=False,
                                             searchable=False),


        ]),

        dbc.FormGroup([
            dbc.Label('Choose the X Data'),
            dcc.Dropdown(id='xdata', value='petal_length',
                         options=[{'label': 'Select', 'value': 1}],clearable=False,
                         searchable=False),
        ]),

        dbc.FormGroup([
            dbc.Label('Choose the Y Data '),
            dcc.Dropdown(id='ydata',
                         value='petal_width',
                         options=[{'label': 'Select', 'value': 1}],clearable=False,
                         searchable=False),
        ]),

        dbc.Row([
            dbc.Col(dcc.Graph(id='firstvisualization')),
        ]),


        dbc.FormGroup([
            dbc.Label('Choose the graph of your choice'),
            dcc.Dropdown(id='graphdropdown',
                         value='compositelinebar',
                         options=[{'label': 'Line Graph', 'value': 'linegraph'},
                                  {'label': 'Bar Graph', 'value': 'bargraph'},
                                  {'label': 'Composite Line Bar Graph', 'value': 'compositelinebar'}
                                  ],clearable=False,
                         searchable=False),
        ]),
        # Part Two - The second layout of the application  - Contains the definition for the user's second interaction
        # User can see the number of rows being displayed
        html.Div([
            dbc.Card(
                dbc.CardBody([
                    html.H4("Number of Rows Being Displayed"),
                    html.P(
                        id='no_of_rows',
                        children={}
                    ),
                ]),
                style={"width": "18rem"},
            ),
        ]),


        # Part Three - The third layout of the application  - Contains the definition for the user's third interaction
        # User can choose the plotly visualizations
        # Choose what graph they want to see
        dbc.FormGroup([
            dbc.Label('Select Plotly Based Visualizations'),
            dcc.Dropdown(id='plotlydropdown',
                         value='pbarplot',
                         options=[{'label': 'Plotly Bar Plot', 'value': 'pbarplot'},
                                  {'label': 'Plotly Scatter Plot', 'value': 'pscatterplot'},
                                  {'label': 'Plotly Polar Plot', 'value': 'spolarplot'}
                                  ],clearable=False,
                         searchable=False),
        ]),

        dbc.Row([
            dbc.Col(dcc.Graph(id='secondvisualization')),
        ]),

        html.Div([
            dbc.Card(
                dbc.CardBody([
                    html.H4("Number of Values Selected By the Visualization"),
                    html.P(
                        id='no_of_values',
                        children={}
                    ),
                ]),
                style={"width": "18rem"},
            ),
        ]),
    ])


    #This is the first layer of the callbacks - The user firstly selects the dataset
    # Then in response the columns get changed and the user can select the values from the drop downs

    @app.callback(
        [Output(component_id='xdata', component_property='options'),
         Output(component_id='ydata', component_property='options'),
         Output(component_id='xdata', component_property='value'),
         Output(component_id='ydata', component_property='value')
         ],
        [Input(component_id='dataset_dropdown', component_property='value')]
    )
    def uponselection(dataset):
        if dataset == 'iris':
            return dfiriscollist, dfiriscollist, dfiriscollist[0]['value'], dfiriscollist[1]['value']
        elif dataset == 'life':
            return dflifecollist, dflifecollist, dflifecollist[0]['value'], dflifecollist[1]['value']
        elif dataset == 'game':
            return dfgamecollist, dfgamecollist, dfgamecollist[0]['value'], dfgamecollist[1]['value']

    #This is the second layer of the callbacks - The user selects the dataset, then the x value, y value and the graph
    # Then in response the graph and the no of rows are displayed
    @app.callback(
        [Output(component_id='firstvisualization', component_property='figure'),
         Output(component_id='no_of_rows', component_property='children')],
        [Input(component_id='dataset_dropdown', component_property='value'),
         Input(component_id='xdata', component_property='value'),
         Input(component_id='ydata', component_property='value'),
         Input(component_id='graphdropdown', component_property='value')]
    )
    # The selection of the graphs based on the dataset and the graph choice
    def graphselect(dataset, xdata, ydata, graph):
        if dataset == 'iris':
            if graph == 'bargraph':
                return bargraph(dfiris,xdata,ydata)
            elif graph == 'linegraph':
                return linegraph(dfiris,xdata,ydata)
            elif graph == 'compositelinebar':
                return compositegraph(dfiris,xdata,ydata)
        elif dataset == 'game':
            if graph == 'bargraph':
                return bargraph(dfgame,xdata,ydata)
            elif graph == 'linegraph':
                return linegraph(dfgame,xdata,ydata)
            elif graph == 'compositelinebar':
                return compositegraph(dfgame,xdata,ydata)
        elif dataset == 'life':
            if graph == 'bargraph':
                return bargraph(dflife,xdata,ydata)
            elif graph == 'linegraph':
                return linegraph(dflife,xdata,ydata)
            elif graph == 'compositelinebar':
                return compositegraph(dflife,xdata,ydata)
        else:
            return compositegraph(dflife,xdata,ydata)

    #This is the third layer of the callbacks - The user selects the plotly based visualization options
    # Then in response the plotly based visualizations are displayed along with the number of values being selected
    @app.callback(
            [Output(component_id='secondvisualization', component_property='figure'),
             Output(component_id='no_of_values', component_property='children')],
            [Input(component_id='plotlydropdown', component_property='value')]
    )
    def plotlyvisualisation(plotlygraphchoice):
        if plotlygraphchoice == 'pbarplot':
            df = read_dataset(Path(r'H:\New folder (2)\Assignments (3)\Assignments', 'iris.csv'))
            clustersiris = cluster_iris_dataset_again()
            df['clusters'] = clustersiris['clusters']
            newdf = df.groupby(["species", "clusters"]).size().unstack(fill_value=0).stack().reset_index()
            newdf['clusters'] = newdf['clusters'].astype(str)
            newdf = newdf.rename({0: 'count'}, axis=1)
            points = str(len(newdf)*3)
            pointz = "Data Points - " + points
            fig = plotly_bar_plot_chart()
            return fig, pointz
        elif plotlygraphchoice == 'pscatterplot':
            fig = plotly_scatter_plot_chart()
            df = read_dataset(Path(r'H:\New folder (2)\Assignments (3)\Assignments', 'iris.csv'))
            clustersiris = cluster_iris_dataset_again()
            #Inputting the clusters column from the processed dataset so that the color would be used to map the data points.
            df['clusters'] = clustersiris['clusters']
            points = str(len(df)*2)
            pointz = "Data Points - " + points
            return fig, pointz
        elif plotlygraphchoice == 'spolarplot':
            df = read_dataset(Path(r'H:\New folder (2)\Assignments (3)\Assignments', 'life_expectancy_years.csv'))
            points = str(len(df)*2)
            pointz = "Data Points - " + points
            fig = plotly_polar_scatterplot_chart()
            return fig, pointz

    return app

if __name__ == "__main__":
    # Here are examples of how to run each method
    # Although you may remove or comment them out for testing, make sure they are present
    # and work when submitting, since when we run your code, all methods will be run like this.
    # If these lines below returns errors when running, your file will be considered to not
    # run, and graded accordingly.
    #app_ce = dash_callback_example()
    #app_b = dash_with_bootstrap_example()
    app_c = dash_callback_example()
    #app_t = dash_task()

    # Uncomment the below lines to test your code
    # When submitting, leave the code below commented!!!
    # app_ce.run_server(debug=True)
    # app_b.run_server(debug=True)
    # app_c.run_server(debug=True)
    app_c.run_server(debug=True)
