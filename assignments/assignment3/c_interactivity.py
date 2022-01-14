from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.widgets import Button, Slider


###############
# Interactivity in visualizations is challenging due to limitations and chunkiness of libraries.
# For example, some libraries works well in Jupyter Notebooks, but then the code makes barely any sense and becomes hard to change/update,
# defeating the purpose of using Jupyter notebooks in the first place, and other libraries provide a window of their own, but
# they are very tied to the running code, and far from the experience you'd expect from a proper visual analytics webapp
#
# We will try out some approaches to exercise in this file, but the next file will give you the proper tooling to make a
# well-rounded and efficient code for visual interactions.
##############################################
# Example(s). Read the comments in the following method(s)
##############################################
from assignments.assignment1.a_load_file import read_dataset
from assignments.assignment1.b_data_profile import get_numeric_columns
from assignments.assignment1.c_data_cleaning import fix_nans, fix_outliers
from assignments.assignment2.c_clustering import simple_k_means
from assignments.assignment3 import b_simple_usages


def matplotlib_simple_example():
    """
    Using the same logic from before, we can add sliders or buttons to select options for interactivity.
    Matplotlib is limited to only a few options, but do take a look, since they are useful for fast prototyping and analysis

    In case you are using PyCharm, I suggest you to uncheck the 'Show plots in tool window'
    to be able to see and interact with the buttons defined below.
    This example comes from https://matplotlib.org/3.1.1/gallery/widgets/buttons.html
    """
    freqs = np.arange(2, 20, 3)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    t = np.arange(0.0, 1.0, 0.001)
    s = np.sin(2 * np.pi * freqs[0] * t)
    l, = plt.plot(t, s, lw=2)

    class Index(object):
        ind = 0

        def next(self, event):
            self.ind += 1
            i = self.ind % len(freqs)
            ydata = np.sin(2 * np.pi * freqs[i] * t)
            l.set_ydata(ydata)
            plt.draw()

        def prev(self, event):
            self.ind -= 1
            i = self.ind % len(freqs)
            ydata = np.sin(2 * np.pi * freqs[i] * t)
            l.set_ydata(ydata)
            plt.draw()

    callback = Index()
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)

    return fig, ax


def matplotlib_simple_example2():
    """
    Here is another example, which also has a slider and simplifies a bit the callbacks
    """
    data = np.random.rand(10, 5)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    ax.bar(np.arange(10).astype(str).tolist(), data[:, 0])

    class Index(object):
        ind = 0
        multiplier = 1

        def change_data(self, event, i):
            self.ind = np.clip(self.ind + i, 0, data.shape[1] - 1)
            ax.clear()
            ax.bar(np.arange(10).astype(str).tolist(), data[:, self.ind] * self.multiplier)
            plt.draw()

        def change_multiplier(self, value):
            self.multiplier = value
            ax.clear()
            ax.bar(np.arange(10).astype(str).tolist(), data[:, self.ind] * self.multiplier)
            plt.draw()

    callback = Index()
    axprev = plt.axes([0.1, 0.05, 0.12, 0.075])
    axnext = plt.axes([0.23, 0.05, 0.12, 0.075])
    axslider = plt.axes([0.55, 0.1, 0.35, 0.03])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(lambda event: callback.change_data(event, 1))
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(lambda event: callback.change_data(event, -1))
    slider = Slider(axslider, 'multiplier', 1, 10, 1)
    slider.on_changed(callback.change_multiplier)

    return fig, ax


def plotly_slider_example():
    """
    Here is a simple example from https://plotly.com/python/sliders/ of how to include a slider in plotly
    Notice how similar it already is to GapMinder!
    """
    df = px.data.gapminder()
    fig = px.scatter(df, x="gdpPercap", y="lifeExp",
                     animation_frame="year",  # set which column makes the animation though a slider
                     size="pop",
                     color="continent",
                     hover_name="country",
                     log_x=True,
                     size_max=55,
                     range_x=[100, 100000],
                     range_y=[25, 90])

    fig["layout"].pop("updatemenus")  # optional, drop animation buttons

    return fig


def plotly_button_example():
    """
    To have buttons, plotly requires us to use go (and not px) to generate the graph.
    The button options are very restrictive, since all they can do is change a parameter from the go graph.
    In the example below, it changes the "mode" value of the graph (between lines and scatter)
    The code is a modified code example taken from https://plotly.com/python/custom-buttons/
    """
    x = np.random.rand(50) * np.random.randint(-10, 10)
    y = np.random.rand(50) * np.random.randint(-10, 10)

    fig = go.Figure()

    # Add surface trace
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers'))

    # Add dropdown
    fig.update_layout(
        updatemenus=[
            dict(type="buttons",
                 direction="left",
                 buttons=[
                     dict(
                         label="line",  # just the name of the button
                         method="update",  # This is the method of update (check https://plotly.com/python/custom-buttons/)
                         args=[{"mode": "markers"}],  # This is the value being updated in the visualization
                     ), dict(
                         label="scatter",  # just the name of the button
                         method="update",  # This is the method of update (check https://plotly.com/python/custom-buttons/)
                         args=[{"mode": "line"}],  # This is the value being updated in the visualization
                     )
                 ],
                 pad={"r": 10, "t": 10}, showactive=True, x=0.11, xanchor="left", y=1.1, yanchor="top"  # Layout-related values
                 ),
        ]
    )

    fig.show()
    return fig


##############################################
# Implement all the below methods
# Use the documentation of the libraries in case you need help to understand the required methods
##############################################
def matplotlib_interactivity():
    """
    Do an interactive matplotlib plot where I can select which visualization I want.
    Make either a slider, a dropdown or several buttons and make so each option gives me a different visualization from
    the matplotlib figures of b_simple_usages. Return just the resulting fig as is done in plotly_slider_example.
    """
    # Initializing x and y with random values.
    # Creating a 2D Matrix with Random Values
    x = np.random.rand(50) * np.random.randint(-10, 10)
    y = np.random.rand(50) * np.random.randint(-10, 10)
    matrix_2D = np.random.rand(10, 10) * np.random.randint(-10, 10)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    ax.bar(x, y)
    ax.set_xlabel('X - LABEL ')
    ax.set_ylabel('Y - LABEL ')
    ax.set_title('Interactive Matplotlib Charts')
    class Index(object):
        def bargraph(self, event):
            ax.clear()
            ax.bar(x, y)
            plt.draw()

        def piechart(self, event):
            ax.clear()
            labels = range(len(x))
            ax.pie(x, labels=labels)
            plt.draw()

        def heatmap(self, event):
            ax.clear()
            ax.imshow(matrix_2D, cmap='YlGnBu')
            plt.draw()

        def histogram(self, event):
            ax.clear()
            ax.hist(x, 10)
            plt.draw()

    callback = Index()

    bargraph_axis = plt.axes([0.13, 0.05, 0.12, 0.075])
    bargraph_button = Button(bargraph_axis, 'Bar Graph')
    bargraph_button.on_clicked(callback.bargraph)

    piechart_axis = plt.axes([0.26, 0.05, 0.12, 0.075])
    piechart_button = Button(piechart_axis, 'Pie Chart')
    piechart_button.on_clicked(callback.piechart)

    heatmap_axis = plt.axes([0.39, 0.05, 0.12, 0.075])
    heatmap_button = Button(heatmap_axis, 'Heatmap')
    heatmap_button.on_clicked(callback.heatmap)

    histogram_axis = plt.axes([0.51, 0.05, 0.12, 0.075])
    histogram_button = Button(histogram_axis, 'Histogram')
    histogram_button.on_clicked(callback.histogram)

    return fig

def matplotlib_cluster_interactivity():
    """
    Do an interactive matplotlib plot where I can select how many clusters I want to train from.
    Use iris dataset (just numeric columns) and k-means (feel free to reuse as/c_clustering if you wish).
    The slider (or dropdown) should range from 2 to 10. Return just the resulting fig.
    """

    df = read_dataset(Path(r'H:\New folder (2)\Assignments (3)\Assignments\iris.csv'))
    numeric_columns = get_numeric_columns(df)
    for numeric_column in numeric_columns:
        df = fix_nans(df, numeric_column)
        df = fix_outliers(df, numeric_column)

    df = df[df.columns[df.columns.isin(numeric_columns)]]

    kmeansmodel = simple_k_means(df, 2)
    fig, ax = plt.subplots()
    ax.plot(kmeansmodel['clusters'])
    ax.set_xlabel('X - LABEL ')
    ax.set_ylabel('Y - LABEL ')
    ax.set_title('Interactive Matplotlib Charts')

    # The callback function is responsible for making sure that specific cluster data is displayed when a new slider
    # is selected by the user.
    class Index(object):

        def clustercall(self, clusterno):
            model = simple_k_means(df, clusterno)
            ax.clear()
            ax.plot(model['clusters'])


    # The slide is moving from the lower limit : 2 to upper limit : 10.
    callback = Index()
    axslider = plt.axes([0.55, 0.1, 0.35, 0.03])
    slider = Slider(axslider, 'Cluster Number', 2, 10, valstep=1)
    slider.on_changed(callback.clustercall)

    return fig

def plotly_interactivity():
    """
    Do a plotly graph with all plotly 6 figs from b_simple_usages, and make 6 buttons (one for each fig).
    Change the displayed graph depending on which button I click. Return just the resulting fig.
    """
    # Getting the figure from the bar plot method in b
    fig = b_simple_usages.plotly_bar_plot_chart()
    bardict = fig.to_dict()
    bardata = bardict["data"]
    bardata2 = {}
    # Sorting the keys and values and assigning them to the particular key
    for key, value in bardata[0].items():
        bardata2[key] = [value]
    # appending values into the existing key when length is greater than 1 - more than 1 entries
    if(len(bardata) > 1):
        for data in bardata[1:]:
            for key, value in data.items():
                bardata2[key].append(value)
    barbutton = dict(label="Bar Graph",method="update",args=[bardata2],)
    # Getting the figure from the scatter plot method in b
    scatterdict = b_simple_usages.plotly_scatter_plot_chart().to_dict()
    scatterdata = scatterdict["data"]
    scatterdata2 = {}
    for key, value in scatterdata[0].items():
        scatterdata2[key] = [value]
    if(len(scatterdata) > 1):
        for data in scatterdata[1:]:
            for key, value in data.items():
                scatterdata2[key].append(value)
    #creating a dictionary of the button along with the data as the arguement key
    scatterbutton = dict(label="Scatter Plot",method="update",args=[scatterdata2],)

    # Getting the figure from the map plot method in b
    mapdict = b_simple_usages.plotly_map().to_dict()
    mapdata = mapdict["data"]
    mapdata2 = {}
    for key, value in mapdata[0].items():
        mapdata2[key] = [value]
    if(len(mapdata) > 1):
        for data in mapdata[1:]:
            for key, value in data.items():
                mapdata2[key].append(value)
    mapbutton = dict(label="Map Graph",method="update",args=[mapdata2],)
    buttons = [ barbutton, scatterbutton, mapbutton
    ]


    #Giving the update layout method the dictionary of the buttton as a value passed to the buttons dictionary key,
    # adding padding and the indentation to the memnu"
    fig.update_layout(
        updatemenus=[
            dict(type="buttons",
                 direction="right",
                 buttons=buttons,
                 pad={"r": 15, "t": 15}, showactive=True, x=0.12, xanchor="left", y=1.1, yanchor="top"
                 ),
        ]
    )



    return fig



if __name__ == "__main__":
    # Here are examples of how to run each method
    # Although you may remove or comment them out for testing, make sure they are present
    # and work when submitting, since when we run your code, all methods will be run like this.
    # If these lines below returns errors when running, your file will be considered to not
    # run, and graded accordingly.
    fig_m_i = matplotlib_interactivity()
    fig_m_ci = matplotlib_cluster_interactivity()
    fig_p =  plotly_interactivity()

    # Uncomment the below lines to test your code
    # When submitting, leave the code below commented!!!
    # matplotlib_simple_example()[0].show()
    # matplotlib_simple_example2()[0].show()
    # plotly_slider_example().show()
    # plotly_button_example().show()
    # fig_m_i.show()
    # fig_m_ci.show()
    fig_p.show()
