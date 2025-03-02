#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 19:12:21 2024

@author: quentin
"""

"""#=============================================================================
   #=============================================================================
   #=============================================================================

    Main.

#=============================================================================
   #=============================================================================
   #============================================================================="""


import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State, ALL 
import dash_bootstrap_components as dbc
import pandas as pd
import dask.dataframe as dd
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as pxf
from termcolor import colored
import numpy as np
import seaborn as sns

import os
import time
import webbrowser
import socket
import psutil
import threading
import requests
import shutil
import json

import web_interface_style as wis
import Dash_callback_ids as dci

import table_dropdown_style as tds
import figure_creation as fc
import data_plot_preparation as dpp
import figure_dropdown_style as fds
import correlation_feature as cf

import open_dataframe as od
import Function_dataframe as fd
import Function_errors as fe

"""#=============================================================================
   #=============================================================================
   #============================================================================="""

# Save the project on github with: !bash ./save_project_on_git.sh
GitHub_adress= 'https://github.com/KantMG/Exploratory_Data_Analysis'

# Save the project on the laptop:
Project_path='/home/quentin/Documents/Work/Data_analytics/Studies/Kaggle_challenges/Titanic/kaggle/input/titanic/'
file_name = "train.csv"

# Project_path='/home/quentin/Documents/Work/Data_analytics/Datasets/Kaggle/home-data-for-ml-course/'
# file_name = "train.csv"

# Project_path='/home/quentin/Documents/Work/Data_analytics/Studies/Kaggle_challenges/Store_Sales/kaggle/input/store-sales-time-series-forecasting/'
# file_name = "train.csv"

# Get the current working directory or script path as needed
current_file_path = os.getcwd()+'/Main.py'



start_time = time.time()


Large_file_memory = False
df1 = od.read_and_rename(
    Project_path+file_name,
    large_file=Large_file_memory
)


tar = "Survived"

List_dim = ["1D", "2D", "3D"]
List_graph_type = ["Histogram", "Curve", "Scatter", "Boxes", "Colormesh", "Pie", "Histogram Movie", "Curve Movie", "Scatter Movie"]

# Global variable to hold previous clicks for subplot buttons
previous_clicks = {}
previous_reset_clicks = 0
last_clicked_index = 0

List_col_tab2 = df1.columns.tolist()
List_col_tab3 = df1.columns.tolist()
List_col_fig_tab3 = df1.columns.tolist()


List_col_exclude_tab2 = []


print(colored("***************** Start dash ****************", "yellow"))


# Initialize the Dash app with suppress_callback_exceptions set to True
app, dark_dropdown_style, uniform_style = wis.web_interface_style()

# Callback to manage tab content
@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H1("Dash interface dedicated to the Exploratory Data Analysis.", style={"color": "#FFD700", 'height': '20px'}, className="text-light"),
            tab1_content()
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H1("Graphic interface dedicated to Exploratory Data Analysis.", style={"color": "#FFD700"}, className="text-light"),
            tab2_content()
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.Div([
                html.H1("Tabular interface dedicated to Exploratory Data Analysis.", style={"color": "#FFD700"}, className="text-light"),
            ]),
            dcc.Input(id='input-value', type='text', placeholder='Enter a value...', style={**dark_dropdown_style, **uniform_style}),
            html.Div(id='dynamic-content')
        ])


"""
# =============================================================================
# =============================================================================
# =============================================================================
# Tab-1
# =============================================================================
# =============================================================================
# =============================================================================
"""

def tab1_content():
    print()
    print("Time computation=", time.time()-start_time)
    print(colored("=====================  Tab1_content  =========================", "yellow"))
    
    # Content Descriptions
    Text1 = "This project highlights the evolution over the years of movie and series production."
    Text2 = "It analyzes the adaptation of production methods and consumption patterns."
    Text3 = "The focus lies on how much countries invest in film production and their influence on others."
    
    dataset_link = "The IMDb Non-Commercial Datasets are open source and can be found [here](https://developer.imdb.com/non-commercial-datasets/)."
    
    Text5 = "These datasets consist of a variety of tab-separated-values (TSV) formatted files in the UTF-8 character set."
    
    interface_description = "The interface comprises three tabs:"
    
    # Tab Descriptions
    tabs_description = [
        "🏠 Home: This is the homepage.",
        "📈 Analytics: Provides an analysis of the overall dataframe using Plotly for visualization.",
        " It offers various functionalities including:",
        "   - Data filtration",
        "   - Machine learning regressions",
        "   - Smoothing with the Savitzky-Golay filter",
        "   - Variable creation",
        "   - Dynamic subplot modification",
        "🎥 Movies & Artists: Allows analysis based on requests for specific artist names."
    ]
    
    # Print all ids
    component_ids = dci.get_component_ids(app.layout)
    print("Component IDs:", component_ids)
        
    print(colored("==================== End Tab1_content ========================", "yellow"))  


    dtype_counts = df1.dtypes.value_counts()
    
    dtype_df = dtype_counts.reset_index()
    dtype_df.columns = ['dtype', 'count']  # Rename columns for clarity
    dtype_df['dtype'] = dtype_df['dtype'].astype(str)
    
    fig_dtype_df1 = px.pie(
        dtype_df, 
        values='count', 
        names='dtype',
        title='Data Types Distribution in '+file_name
    )
    fig_dtype_df1.update_traces(
        textinfo='label+percent',  # Show label and percentage on pie chart
        hoverinfo='label+value+percent',  # Show label, count (value), and percentage on hover
        customdata=dtype_df['count']  # Include count for hover
    )
    fig_dtype_df1.update_layout(
        plot_bgcolor='#1e1e1e',  # Darker background for the plot area
        paper_bgcolor='#101820',  # Dark gray for the paper
        font=dict(color='white'),  # White text color
        # title = figname,
        # title_font=dict(size=20, color='white')
        )


    missing_data = df1.isna().sum()
    
    fig_missing_bar = px.bar(
        x=missing_data.index,  # Column names (categories)
        y=missing_data.values,  # Count of missing values
        labels={'x': 'Columns', 'y': 'Count of Missing Values'},
        title='Count of Missing Values in Each Column'
    )
    fig_missing_bar.update_layout(
        plot_bgcolor='#1e1e1e',  # Darker background for the plot area
        paper_bgcolor='#101820',  # Dark gray for the paper
        font=dict(color='white'),  # White text color
        # title = figname,
        # title_font=dict(size=20, color='white')
        )


    unique_counts = df1.nunique()
    unique_counts_df = unique_counts.reset_index()  # Reset index to create a DataFrame
    unique_counts_df.columns = ['Columns', 'Unique Count']


    fig_nuinque_bar = px.bar(
        unique_counts_df,
        x="Columns",  # Column names (categories)
        y='Unique Count',  # Count of missing values
        # labels={'x': 'Columns', 'y': 'Count of Missing Values'},
        title='Count of Unique Values in Each Column'
    )
    fig_nuinque_bar.update_layout(
        plot_bgcolor='#1e1e1e',  # Darker background for the plot area
        paper_bgcolor='#101820',  # Dark gray for the paper
        font=dict(color='white'),  # White text color
        # title = figname,
        # title_font=dict(size=20, color='white')
        )


    # fig_missing_heatmap = sns.heatmap(missing_data, cbar=False)

    
    df1_description = df1.describe().reset_index()
    data_table_df1 = tds.dropdown_table(df1_description, 'table-df1', tab,
                                     dark_dropdown_style, uniform_style, False)[1]    
    
    
    
    correlation_matrix = df1.corr(numeric_only=True)
    
    # Create the heatmap using Plotly Express
    fig_correlation_heatmap = px.imshow(
        correlation_matrix,
        color_continuous_scale='RdBu',  # Color scale similar to Seaborn
        labels=dict(x='Columns', y='Columns', color='Correlation'),
        title='Correlation Matrix Heatmap',
        zmin=-1,  # Set minimum value for the colorbar
        zmax=1    # Set maximum value for the colorbar
    )    
    fig_correlation_heatmap.update_traces(text=correlation_matrix.round(2).values, texttemplate="%{text}", textfont={"size": 12})
    fig_correlation_heatmap.update_layout(
        plot_bgcolor='#1e1e1e',  # Darker background for the plot area
        paper_bgcolor='#101820',  # Dark gray for the paper
        font=dict(color='white'),  # White text color
        # title = figname,
        # title_font=dict(size=20, color='white')
        )
    


    cf.correlation_target(df1, tar)
    
    
    return html.Div([
        html.Div([
            html.H2("Dataframe name:", style={"color": "#FFD700"}, className="text-light"),
            html.P(file_name),
        ]),
        html.Div([
            html.H2("Feature characteristics:", style={"color": "#FFD700"}, className="text-light"),
            # html.P(Text5),
        ]),
        
        
        
        html.Div(
            style={'display': 'flex'}, 
            children=[
                # Graph on the left for data types distribution
                html.Div(
                    [dcc.Graph(id='dtype-df1', style={'width': '80%', 'height': '500px'},
                               figure=fig_dtype_df1)], 
                    style={'margin-left': '20px', 'width': '45%'}  # Adjust width as needed
                ),
                # Graph on the right for missing values heatmap
                html.Div(
                    [dcc.Graph(id='heatmap-df1', style={'width': '90%', 'height': '500px'},
                               figure=fig_missing_bar)], 
                    style={'margin-left': '20px', 'width': '45%'}  # Adjust width as needed
                ),
            ]
        ),

        html.Div([
            html.Div(style={'display': 'flex', 'margin-top': '10px', 'overflowX': 'auto'}, children=[
                html.Div(data_table_df1, style={'width': '100%'})  # Table display
            ])
        ]),

        html.Div([
            html.H2("Feature connection:", style={"color": "#FFD700"}, className="text-light"),
        ]),

        html.Div(
            style={'display': 'flex'},
            children=[
                html.Div(
                    [dcc.Graph(id='correlation-heatmap-df2', style={'width': '70%', 'height': '700px'},
                               figure=fig_correlation_heatmap)],
                    style={'margin-left': '10px', 'width': '70%'}
                ),
                html.Div(
                    [dcc.Graph(id='nunique-df1', style={'width': '90%', 'height': '500px'},
                               figure=fig_nuinque_bar)], 
                    style={'margin-left': '20px', 'width': '45%'}  # Adjust width as needed
                ),
            ]
        ),


        html.Div([
            html.H2("Feature/Target connection:", style={"color": "#FFD700"}, className="text-light"),
            # html.P("Select the target:"),
        ]),

        html.Div(
            style={'display': 'flex'},
            children=[
                html.Div(
                    [dcc.Graph(id='correlation-heatmap-df2', style={'width': '70%', 'height': '700px'},
                               figure=fig_correlation_heatmap)],
                    style={'margin-left': '10px', 'width': '70%'}
                ),
            ]
        )



    ], style={'padding': '20px'})


# df1.dtypes.value_counts().plot.pie()

# sns.heatmap(df1.isna(), cbar=False)

# df1.describe()


        # html.Div([
        #     html.H2("Nan value:", style={"color": "#FFD700"}, className="text-light"),
        #     html.P(interface_description),
        #     *[html.P(text) for text in tabs_description],  # Dynamically add tab descriptions
        # ])
        



"""
# =============================================================================
# =============================================================================
# =============================================================================
# Tab-2
# =============================================================================
# =============================================================================
# =============================================================================
"""


def tab2_content():
    print()
    print("Time computation=", time.time()-start_time)
    print(colored("=====================  Tab2_content  =========================", "yellow"))
    tab = 'tab-2'
    # Display dropdowns without loading data initially
    
    exclude_cols = List_col_exclude_tab2

    df_selected = df1[[col for col in df1.columns if col not in exclude_cols]]
        
    dropdowns_with_labels_for_fig_tab2 = fds.dropdown_figure(df_selected, 'graph-df1', tab, dark_dropdown_style, uniform_style, Large_file_memory)

    dropdowns_with_labels_for_fig_filter_tab2 = fds.button_modal_dropdowns_inputs("filter-"+tab,  "Filter on data",
                                                                  df_selected, 'graph-df1', tab,
                                                                  "Select filters on the dataframe.", dark_dropdown_style, uniform_style)
    
    button_dropdown_function_tab2 = fds.button_modal_double_input("function-"+tab,  "Function creation",
                                                                  "Enter function name", "Enter operation (e.g., A + B)",
                                                                  "Create Function", dark_dropdown_style, uniform_style)

    button_dropdown_regression_tab2 = fds.button_modal_dropdown_and_double_input("regression-"+tab, "Regression model", 
                                                                                 ["Polynomial Regression", "Decision Tree", "k-NN"], "Enter an order if needed", 
                                                                                 "Enter a test size ratio (0-1)", "Create regression", dark_dropdown_style, uniform_style)

    button_dropdown_smoothing_tab2 = fds.button_modal_dropdown_input("smoothing-"+tab,  "Smoothing", 
                                                                     ["Savitzky-Golay Filter"], "Enter an order if needed",
                                                                     "Select a smoothing function", dark_dropdown_style, uniform_style)

    button_subplot_tab2 = fds.button_modal_subplot_creation("subplot-"+tab,  "Subplot creation", 
                                                                     "Number of subplot", "Number of rows", "Number of columns",
                                                                     "Configuration of the subplot figure", dark_dropdown_style, uniform_style)

    component_ids = dci.get_component_ids(app.layout)
    print("Component IDs:", component_ids)
    print(colored("==================== End Tab2_content ========================", "yellow"))
    return html.Div([
        html.Div([
            fds.figure_position_dash(tab,
                                     'graph-output-'+tab,
                                     dropdowns_with_labels_for_fig_tab2,
                                     dropdowns_with_labels_for_fig_filter_tab2,
                                     button_dropdown_function_tab2,
                                     button_dropdown_regression_tab2,
                                     button_dropdown_smoothing_tab2,
                                     button_subplot_tab2
                                     )
        ], style={'padding': '20px'}),
    ], style={'padding': '20px'})
    

# =============================================================================
# Callback for graph in tab-2
# =============================================================================

#  -----------------------------------------------------------------

@app.callback(
    Output("modal-function-tab-2", "is_open"),
    [Input("open-modal-function-tab-2", "n_clicks"), Input("submit-button-function-tab-2", "n_clicks")],
    [State("modal-function-tab-2", "is_open")]
)
def toggle_modal(open_clicks, submit_clicks, is_open):
    if open_clicks or submit_clicks:
        return not is_open
    return is_open

@app.callback(
    Output('output-div-function-tab-2', 'children'),
    [Input('submit-button-function-tab-2', 'n_clicks')],
    [State('input_1-function-tab-2', 'value'), State('input_2-function-tab-2', 'value')]
)
def update_output(n_clicks, func_name, input_value):
    print("Submit button clicks:", n_clicks)  # Check for clicks
    print("Function Name:", func_name)  # Current function name
    print("Input Value:", input_value)  # Value of input expression
    
    if n_clicks > 0:
        try:
            # Validate that func_name and input_value are provided
            if not func_name or not input_value:
                return "Error: Function name and input expression are required."

            # Transform input expression to reference DataFrame columns correctly
            expression = input_value
            for column in df1.columns:
                expression = expression.replace(column, f"df['{column}']")
                
            # Create a new function that evaluates the transformed expression
            exec(f"def {func_name}(df): return {expression}", {}, locals())
            # Calculate the result for all rows
            df1[func_name] = locals()[func_name](df1)  # Add a new column with the results
            
            return f"New column '{func_name}' added to the dataframe."
        except Exception as e:
            return f"Error: {str(e)}"
    return ""

@app.callback(
    Output('x-dropdown-tab-2', 'options'),
    [Input('submit-button-function-tab-2', 'n_clicks')],
    [State('input_1-function-tab-2', 'value'), State('input_2-function-tab-2', 'value')]
)
def update_dropdown_options(n_clicks, func_name, input_value):
    print()
    print(colored("------------ callback update_x_dropdown_tab2 ------------", "red"))
    return [{'label': col, 'value': col} for col in df1.columns if col not in List_col_exclude_tab2]

#  -----------------------------------------------------------------

@app.callback(
    Output("modal-regression-tab-2", "is_open"),
    [Input("open-modal-regression-tab-2", "n_clicks"), Input("submit-button-regression-tab-2", "n_clicks")],
    [State("modal-regression-tab-2", "is_open")]
)
def toggle_modal(open_clicks, submit_clicks, is_open):
    if open_clicks or submit_clicks:
        return not is_open
    return is_open

#  -----------------------------------------------------------------

@app.callback(
    Output("modal-smoothing-tab-2", "is_open"),
    [Input("open-modal-smoothing-tab-2", "n_clicks"), Input("submit-button-smoothing-tab-2", "n_clicks")],
    [State("modal-smoothing-tab-2", "is_open")]
)
def toggle_modal(open_clicks, submit_clicks, is_open):
    if open_clicks or submit_clicks:
        return not is_open
    return is_open

#  -----------------------------------------------------------------

@app.callback(
    Output("modal-filter-tab-2", "is_open"),
    [Input("open-modal-filter-tab-2", "n_clicks"), Input("submit-button-filter-tab-2", "n_clicks")],
    [State("modal-filter-tab-2", "is_open")]
)
def toggle_modal(open_clicks, submit_clicks, is_open):
    if open_clicks or submit_clicks:
        return not is_open
    return is_open

#  -----------------------------------------------------------------

@app.callback(
    Output("modal-subplot-tab-2", "is_open"),
    [Input("open-modal-subplot-tab-2", "n_clicks"), Input("submit-button-subplot-tab-2", "n_clicks"), Input("submit-reset-button-subplot-tab-2", "n_clicks")],
    [State("modal-subplot-tab-2", "is_open")]
)
def toggle_modal(open_clicks, submit_clicks, reset_click, is_open):
    if open_clicks or submit_clicks or reset_click:
        return not is_open
    return is_open

@app.callback(
    Output('output-div-subplot-tab-2', 'children'),
    [Input('submit-reset-button-subplot-tab-2', 'n_clicks'), 
     Input('submit-button-subplot-tab-2', 'n_clicks')],
    [State('input_1-subplot-tab-2', 'value'), State('input_2-subplot-tab-2', 'value'), State('input_3-subplot-tab-2', 'value')],
    prevent_initial_call=True
)
def update_output(reset_click, n_clicks, input_1_value, input_2_value, input_3_value):

    print(colored("-------------- callback update_output --------------", "red"))
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    print("Triggered component:", triggered_id)
    print()

    print("Submit button clicks:", n_clicks)  # Check for clicks
    print("Inputs Name:", [input_1_value, input_2_value, input_3_value])  # Current function name
    
    global previous_clicks, previous_reset_clicks
    # Reset button clicked
    if reset_click > previous_reset_clicks:
        previous_reset_clicks = reset_click
        previous_clicks = [[]]
        return [[]]
    
    if n_clicks > 0:
        try:
            # Validate that func_name and input_value are provided
            if not input_1_value or not input_2_value or not input_3_value:
                return dash.no_update
                # return "Error: Function name and input expression are required."
            
            # Create the buttons which will correspond to each subplot.
            buttons_subplot_tab2 = fds.buttons_subplots("Figure-"+tab+"-subplot-", "Subplot ",
                                                        input_1_value, input_2_value, input_3_value, dark_dropdown_style, uniform_style)
            
            previous_clicks = [0] * input_1_value
            return buttons_subplot_tab2
        except Exception as e:
            return f"Error: {str(e)}"
    return ""


#  -----------------------------------------------------------------

@app.callback(
    Output('y-dropdown-tab-2', 'options'),
    [Input('x-dropdown-tab-2', 'value')]+
    [Input('tabs', 'value')]
)
def update_y_dropdown_tab2(selected_x, selected_tab):
    print()
    print(colored("------------ callback update_y_dropdown_tab2 ------------", "red"))
    print("Active Tab=", selected_tab)
    print("Time computation=", time.time()-start_time)
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    print("Triggered component:", triggered_id)
    print()
    if selected_tab == 'tab-2':
        if selected_x is None:
            print("X Dropdown Value is None, returning an empty list [].")
            return []
        print(f"Selected X: {selected_x}")  # Additional debugging
        exclude_cols=List_col_exclude_tab2
        return update_y_dropdown_utility(selected_x, df1.columns, exclude_cols)
    return dash.no_update

@app.callback(
    Output('z-dropdown-tab-2', 'options'),
    [Input('x-dropdown-tab-2', 'value'),
     Input('y-dropdown-tab-2', 'value'),
    Input('tabs', 'value')]
)
def update_z_dropdown_tab2(selected_x, selected_y, selected_tab):
    print()
    print(colored("------------ callback update_z_dropdown_tab2 ------------", "red"))
    print("Active Tab=", selected_tab)
    print("Time computation=", time.time()-start_time)
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    print("Triggered component:", triggered_id)
    print()
    if selected_tab == 'tab-2':
        if selected_y is None:
            print("Y Dropdown Value is None, returning an empty list [].")
            return []
        print(f"Selected y: {selected_y}")  # Additional debugging
        exclude_cols=List_col_exclude_tab2
        return update_z_dropdown_utility(selected_x, selected_y, df1.columns, exclude_cols)
    return dash.no_update

@app.callback(
    [Output('Func on y-dropdown-tab-2', 'options'),
    Output('Func on y-dropdown-tab-2', 'value')],
    Input('y-dropdown-tab-2', 'value'),
    Input('tabs', 'value')  # Include tab value to conditionally trigger callback
)
def update_yfunc_dropdown_tab2(selected_y, selected_tab):
    print()
    print(colored("------------ callback update_yfunc_dropdown_tab2 ------------", "red"))
    print("Active Tab=", selected_tab)
    print("Time computation=", time.time()-start_time)
    if selected_tab == 'tab-2':
        if selected_y is None:
            print("Y Dropdown Value is None, returning an empty list [].")
            return [], []
        print(f"Selected Y: {selected_y}")  # Additional debugging
        
        function_on_y = ["Avg", "Avg on the ordinate", "Value in x_y interval"]
        
        return update_func_dropdown_utility(selected_y, function_on_y, None)
    return dash.no_update, dash.no_update

@app.callback(
    [Output('Func on z-dropdown-tab-2', 'options'),
    Output('Func on z-dropdown-tab-2', 'value')],
    Input('z-dropdown-tab-2', 'value'),
    Input('tabs', 'value')  # Include tab value to conditionally trigger callback
)
def update_zfunc_dropdown_tab2(selected_z, selected_tab):
    print()
    print(colored("------------ callback update_zfunc_dropdown_tab2 ------------", "red"))
    print("Active Tab=", selected_tab)
    print("Time computation=", time.time()-start_time)
    if selected_tab == 'tab-2':
        if selected_z is None:
            print("Z Dropdown Value is None, returning an empty list [].")
            return [], []  # Return an empty options list if the DF is not ready
        # Proceed to get options based on selected_x and stored_df1...
        print(f"Selected Z: {selected_z}")  # Additional debugging
        
        function_on_z = ["Avg", "Avg on the ordinate", "Weight on y"]
        
        return update_func_dropdown_utility(selected_z, function_on_z, 'Avg')
    return dash.no_update, dash.no_update

@app.callback(
    Output('Dim-dropdown-tab-2', 'options'),
    Input('y-dropdown-tab-2', 'value'),
    Input('tabs', 'value')  # Include tab value to conditionally trigger callback
)
def update_dim_dropdown_tab2(selected_y, selected_tab):
    print()
    print(colored("-------- callback update_dim_dropdown_tab2 --------", "red"))
    print("Active Tab=", selected_tab)
    print("Time computation=", time.time()-start_time)
    if selected_tab == 'tab-2':
        if selected_y is None:
            return [{'label': "1D", 'value': "1D"}, {'label': "2D", 'value': "2D"}]  # Return an empty options list if the DF is not ready
        return [{'label': col, 'value': col} for col in List_dim]
    return dash.no_update

@app.callback(
    [Output('Graph-dropdown-tab-2', 'options'),
    Output('Graph-dropdown-tab-2', 'value')],
    Input('Dim-dropdown-tab-2', 'value'),
    Input('tabs', 'value')  # Include tab value to conditionally trigger callback
)
def update_graph_dropdown_tab2(selected_dim, selected_tab):
    print()
    print(colored("------------ callback update_graph_dropdown_tab2 ------------", "red"))
    print("Active Tab=", selected_tab)
    print("Time computation=", time.time()-start_time)
    if selected_tab == 'tab-2':
        if selected_dim == "1D":
            return [{'label': col, 'value': col} for col in List_graph_type if col not in ("Colormesh", "Pie")], 'Histogram'
        if selected_dim == "2D":
            return [{'label': col, 'value': col} for col in List_graph_type if col not in ("Histogram", "Curve", "Scatter", "Histogram Movie", "Curve Movie", "Scatter Movie", "Boxes")], None
        if selected_dim == "3D":
            return [{'label': col, 'value': col} for col in List_graph_type], None
    return dash.no_update, dash.no_update

@app.callback(
    Output('graph-output-tab-2', 'figure'), Output('figure-store-tab-2', 'data'),
    [Input('tabs', 'value'),
     Input('x-dropdown-tab-2', 'value'),
     Input('y-dropdown-tab-2', 'value'),
     Input('z-dropdown-tab-2', 'value'),
     Input('Func on y-dropdown-tab-2', 'value'),
     Input('Func on z-dropdown-tab-2', 'value'),
     Input('Graph-dropdown-tab-2', 'value'),
     Input('Dim-dropdown-tab-2', 'value'),
     Input("dropdown-regression-tab-2", "value"),
     Input("input_1-regression-tab-2", "value"),
     Input("input_2-regression-tab-2", "value"),
     Input("submit-button-regression-tab-2", "n_clicks"),
     Input("dropdown-smoothing-tab-2", "value"),
     Input("input-smoothing-tab-2", "value"),
     Input("submit-button-smoothing-tab-2", "n_clicks"),
     Input("input_1-subplot-tab-2", "value"),
     Input("input_2-subplot-tab-2", "value"),
     Input("input_3-subplot-tab-2", "value"),
     Input("hide-dropdowns-tab-2", "n_clicks"),
     Input("submit-button-filter-tab-2", "n_clicks")] +
    [Input(f'fig-dropdown-{col}-tab-2', 'value') for col in List_col_tab2] +
    [Input({'type': 'subplot-button', 'index': ALL}, 'n_clicks')],
    State('graph-output-tab-2', 'figure'),
    State('figure-store-tab-2', 'data')
    )
def update_graph_tab2(selected_tab, x_dropdown_value, y_dropdown_value, z_dropdown_value,
                      yfunc_dropdown_value, zfunc_dropdown_value, graph_dropdown_value, dim_dropdown_value,
                      reg_dropdown_value, reg_order_value, test_size_value, sub_bot_reg_value,
                      smt_dropdown_value, smt_order_value, sub_bot_smt_value,
                      nb_subplots, nb_subplots_row, nb_subplots_col,
                      hide_drop_fig, sub_bot_filter_value, *args):

    global previous_clicks, last_clicked_index
    
    print()
    print(colored("------------ callback update_graph_tab2 ------------", "red"))
    current_fig = args[-2]
    data_for_plot = args[-1]
    filter_values = list(args[0:len(List_col_tab2)])
    filter_values = {List_col_tab2[i]: (filter_values[i] if filter_values[i] != '' else None) for i in range(min(len(List_col_tab2), len(filter_values)))}
    subplot_button_clicks = list(args[len(List_col_tab2):-2])
    
    # Now to get the flat list
    if subplot_button_clicks and isinstance(subplot_button_clicks, list):
        subplot_button_clicks = subplot_button_clicks[0]  # Access the first element
        print("Subplot Button Clicks:", subplot_button_clicks)
    else:
        subplot_button_clicks = []  # Handle cases where subplot_button_clicks might be empty or wrongly structured
        print("No subplot button clicks found.")

    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    print("Triggered component:", triggered_id)
    print()
    
    if triggered_id in ["dropdown-regression-tab-2", "input-regression-tab-2", "dropdown-smoothing-tab-2", "input-smoothing-tab-2", "input_1-subplot-tab-2", "input_2-subplot-tab-2", "input_3-subplot-tab-2"]:
        return dash.no_update

    df_col_numeric = df1.select_dtypes(include=['float64', 'int64']).columns.tolist()
    df_col_all = df1.columns.tolist()
    df_col_string = [col for col in df_col_all if col not in df_col_numeric]   
    
    if z_dropdown_value is not None and z_dropdown_value not in df_col_numeric:
        print("z-dropdown-tab-2 is "+z_dropdown_value+" which is a string column.")
        print("Please select a numeric column for z-dropdown-tab-2")
        return dash.no_update

    if graph_dropdown_value is None:
        print("Please select a graphic type.")
        return dash.no_update
    
    if triggered_id in list([f'fig-dropdown-{col}-tab-2' for col in List_col_tab2]):
        return dash.no_update

    if triggered_id == "submit-button-regression-tab-2":
        return update_graph_minor_change_utility(x_dropdown_value, y_dropdown_value, z_dropdown_value,
                                                 yfunc_dropdown_value, zfunc_dropdown_value, 
                                                 graph_dropdown_value, dim_dropdown_value,
                                                 reg_dropdown_value, reg_order_value, test_size_value,
                                                 current_fig, data_for_plot, df_col_string)

    if  triggered_id == "hide-dropdowns-tab-2":
        fig_json_serializable = go.Figure(current_fig)
        if hide_drop_fig % 2 == 1:  # Check if the button has been clicked an odd number of times
            # Remove the dropdowns
            fig_json_serializable["layout"]["updatemenus"] = []
        else:
            # Restore dropdowns
            fig_json_serializable.update_layout(updatemenus=fig_json_serializable["layout"]["updatemenus"])
        return fig_json_serializable, data_for_plot
    
    print("Active Tab=", selected_tab)
    print("Time computation=", time.time()-start_time)

    df1_filtered = od.apply_filter(df1, filter_values)
    
    
    ###################### Subplot part ######################
    
    
    # Parse the triggered_id if it is a JSON string
    try:
        parsed_id = json.loads(triggered_id)
    except json.JSONDecodeError:
        print("Could not decode JSON.")
        parsed_id = {}

    # Check if the parsed_id corresponds to a subplot button
    if parsed_id.get('type') == 'subplot-button':
        print('subplot-button type, return no update.')
        if all(click == 0 for click in subplot_button_clicks) :

            return update_graph_subplot_creation(x_dropdown_value, y_dropdown_value, z_dropdown_value,
                                        yfunc_dropdown_value, zfunc_dropdown_value,
                                        graph_dropdown_value, dim_dropdown_value,
                                        nb_subplots, nb_subplots_row, nb_subplots_col,
                                        current_fig, data_for_plot)        
        
        else:
            print(previous_clicks)
            for index, (prev, curr) in enumerate(zip(previous_clicks, subplot_button_clicks)):
                print(curr , prev)
                if curr > prev:  # If current clicks > previous clicks, this button was clicked
                    last_clicked_index = index
                    print(f"Subplot button at index {index} was clicked.")
    
                    # Update the previous clicks
                    previous_clicks = subplot_button_clicks.copy()
            return dash.no_update
    
    # Check whether subplot_button_clicks is valid and not empty
    if not subplot_button_clicks:
        print("No subplot buttons have been clicked. The figure is unique.")  
    elif all(x == 0 for x in subplot_button_clicks):
        print("Subplot buttons are all 0.")
        return update_graph_subplot(x_dropdown_value, y_dropdown_value, z_dropdown_value,
                                    yfunc_dropdown_value, zfunc_dropdown_value,
                                    graph_dropdown_value, dim_dropdown_value,
                                    smt_dropdown_value, smt_order_value, sub_bot_smt_value,
                                    0, nb_subplots, nb_subplots_row, nb_subplots_col,
                                    df1_filtered, df_col_string, current_fig, data_for_plot, Large_file_memory)
    else:           
        # Use last_clicked_index for any needed logic
        if last_clicked_index is not None:
            print(f"Last clicked subplot button index: {last_clicked_index}")
            # Additional logic based on the last clicked button can go here

        return update_graph_subplot(x_dropdown_value, y_dropdown_value, z_dropdown_value,
                                    yfunc_dropdown_value, zfunc_dropdown_value,
                                    graph_dropdown_value, dim_dropdown_value,
                                    smt_dropdown_value, smt_order_value, sub_bot_smt_value,
                                    last_clicked_index, nb_subplots, nb_subplots_row, nb_subplots_col,
                                    df1_filtered, df_col_string, current_fig, data_for_plot, Large_file_memory)
            
            
    
    return update_graph_utility(x_dropdown_value, y_dropdown_value, z_dropdown_value, yfunc_dropdown_value, zfunc_dropdown_value, graph_dropdown_value, dim_dropdown_value, smt_dropdown_value, smt_order_value, sub_bot_smt_value, df1_filtered, df_col_string, Large_file_memory)


"""
# =============================================================================
# =============================================================================
# =============================================================================
# Tab-3
# =============================================================================
# =============================================================================
# =============================================================================
"""

# Callback to update UI based on input value in Tab 3
@app.callback(
    [Output('dynamic-content', 'children'), Output('stored-df2', 'data')],
    Input('input-value', 'value')
)
def update_ui(input_value):
    if not input_value:  # Return nothing if input is empty or None
        return '', None
    print()
    print(colored("------------ callback update_ui ------------", "red"))

    # Check if the input value exists in the 'nconst' column of df_name
    if input_value in df_name['primaryName'].values:
        
        tab = "tab-3"
        
        print(input_value)
        nconst_value = df_name[df_name['primaryName'] == input_value]['nconst'].iloc[0]
        birthYear_value = int(df_name[df_name['primaryName'] == input_value]['birthYear'].iloc[0])
        deathYear_value = int(df_name[df_name['primaryName'] == input_value]['deathYear'].iloc[0])
        
        # Display the found nconst value (for debugging purposes)
        print(f"Matched nconst: {nconst_value}")
                        
        List_col = ["startYear", "runtimeMinutes", "genres", "isAdult", "directors", "writers", "averageRating", "numVotes", "nconst", "category", "characters", "title", "isOriginalTitle"]
        
        List_filter = [None, None, None, None, None, None, None, None, nconst_value, None, None, None, True]
        
        df2 = od.open_dataframe(List_col, List_filter, Project_path, Large_file_memory, Get_file_sys_mem)
        filters = {List_col[i]: (List_filter[i] if List_filter[i] != '' else None) for i in range(min(len(List_col), len(List_filter)))}
        df2 = od.apply_filter(df2, filters)
        exclude_col = ["tconst", "isAdult", "nconst", "isOriginalTitle", "characters"]
        df2 = df2.drop(columns=exclude_col)
        
        # Create a mapping from nconst to primaryName
        mapping = dict(zip(df_name['nconst'], df_name['primaryName']))
        
        # Replace the strings in df2 using the mapping
        df2['directors'] = df2['directors'].replace(mapping)
        df2['writers'] = df2['writers'].replace(mapping)
        
        
        if Large_file_memory:
            df2 = df2.compute()
        
        if len(df2.index) == 0: 
            return html.Div([
                html.Div([
                    html.P(f'The artist '+input_value+' doesnt have referenced movies.'),
                ])
                ], style={'padding': '20px'}), df2.to_dict('records')        
        else:
                        
            # Split the strings into individual elements and flatten the list
            all_elements = df2['category'].str.split(',').explode().str.strip()
            primaryProfession = all_elements.value_counts()
            primaryProfession = primaryProfession[primaryProfession > 1].index.tolist()

            # Create the table with the appropriate dropdowns for each column
            dropdowns_with_labels, data_table_df2 = tds.dropdown_table(df2, 'table-df2', tab, dark_dropdown_style, uniform_style, True)
            
            exclude_col = ["title"]
            df2_filter = df2.drop(columns=exclude_col)            

            dropdowns_with_labels_for_fig_tab3 = fds.dropdown_figure(df2_filter, 'graph-df2', tab, dark_dropdown_style, uniform_style, Large_file_memory)


            dropdowns_with_labels_for_fig_filter_tab3 = fds.button_modal_dropdowns_inputs("filter-"+tab,  "Filter on data",
                                                                          df2_filter, 'graph-df2', tab,
                                                                          "Select filters on the dataframe.", dark_dropdown_style, uniform_style)
    


            button_dropdown_function_tab3 = fds.button_modal_double_input("function-"+tab,  "Function creation",
                                                                          "Enter function name", "Enter operation (e.g., A + B)",
                                                                          "Create Function", dark_dropdown_style, uniform_style)
        
            button_dropdown_regression_tab3 = fds.button_modal_dropdown_input("regression-"+tab, "Regression model", 
                                                                              ["Polynomial Regression", "Decision Tree", "k-NN"], "Enter an order if needed",
                                                                             "Create regression", dark_dropdown_style, uniform_style)
        
            button_dropdown_smoothing_tab3 = fds.button_modal_dropdown_input("smoothing-"+tab,  "Smoothing", 
                                                                             ["Savitzky-Golay Filter"], "Enter an order if needed",
                                                                             "Select a smoothing function", dark_dropdown_style, uniform_style)

            button_subplot_tab3 = fds.button_modal_subplot_creation("subplot-"+tab,  "Subplot creation", 
                                                                             "Number of subplot", "Number of rows", "Number of columns",
                                                                             "Configuration of the subplot figure", dark_dropdown_style, uniform_style)



            return html.Div([
                html.P(f'The artist '+input_value+' is born in '+str(birthYear_value)+' and died in '+str(deathYear_value)+' during its career as '+', '.join(primaryProfession)+' he participated to the creation of the following productions.'),
                html.Div(style={'display': 'flex', 'margin-top': '10px', 'flex-wrap': 'wrap'}, children=[
                    html.Div(dropdowns_with_labels, style={'display': 'flex', 'justify-content': 'flex-start', 'gap': '5px'})
                ]),
                html.Div(style={'display': 'flex', 'margin-top': '10px'}, children=[
                    html.Div(data_table_df2, style={'width': '100%'})  # Adjusted to take full width
                ]),

                html.Div([
                    html.H1("Graphic interface dedicated to the dataframe related to the artist "+input_value+".", style={"color": "#FFD700"}, className="text-light"),
                    
                    fds.figure_position_dash(tab,
                                             'graph-output-'+tab, 
                                             dropdowns_with_labels_for_fig_tab3, 
                                             dropdowns_with_labels_for_fig_filter_tab3,
                                             button_dropdown_function_tab3,
                                             button_dropdown_regression_tab3,
                                             button_dropdown_smoothing_tab3,
                                             button_subplot_tab3
                                             )
                    
                ], style={'padding': '20px'})
                                
            ], style={'padding': '20px'}), df2.to_dict('records')
        
    
    # If the input does not correspond to any primaryName, filter df_name
    filtered_df = df_name[df_name['primaryName'].str.contains(input_value, case=False, na=False)]
    dropdowns_with_labels_df_name, data_table_df_name = tds.dropdown_table(filtered_df, 'table-df_name', 'tab-3' , dark_dropdown_style, uniform_style, False)
    return data_table_df_name, None



# =============================================================================
# Callback for table-df2 in tab-3
# =============================================================================
tab = 'tab-3'

@app.callback(
    Output('table-df2', 'data'),
    [Input(f'{col}-dropdown-table-'+tab, 'value') for col in List_col_tab3],
    Input('tabs', 'value'),  # Include tab value to conditionally trigger callback
    State('stored-df2', 'data')  # Ensure this is included as State
)
def update_stored_df2(*args):
    print()
    print(colored("------------ callback update_stored_df2 ------------", "red")) 
    selected_tab = args[-2]
    stored_df2 = args[-1]         # The last argument is stored_df2
    selected_values = {col: args[i] for i, col in enumerate(List_col_tab3)}
    
    if selected_tab == 'tab-3':  # Only execute if in the Data Visualization tab
        if stored_df2 is None:  # Check if stored_df2 is None or empty
            return []
        # Convert the stored data back to a DataFrame
        df2 = pd.DataFrame(stored_df2)
        # Create a copy of the DataFrame to avoid modifying the original stored data
        filtered_data_table = df2.copy()
        print("Update table")
        print(filtered_data_table)
        filtered_data_table = od.apply_filter(filtered_data_table, selected_values)
        
        return filtered_data_table.to_dict('records')
    return []  # Return empty if not in the right tab


# =============================================================================
# Callback for graph-df2 in tab-3
# =============================================================================

@app.callback(
    Output('y-dropdown-tab-3', 'options'),
    Input('x-dropdown-tab-3', 'value'),
    Input('tabs', 'value')  # Include tab value to conditionally trigger callback
)
def update_y_dropdown_tab3(selected_x, selected_tab):
    print()
    print(colored("------------ callback update_y_dropdown_tab3 ------------", "red")) 
    if selected_tab == 'tab-3':  # Only execute if in the correct tab
        exclude_cols = ["title", "characters"]
        return update_y_dropdown_utility(selected_x, List_col_fig_tab3, exclude_cols)
    return []  # Return empty if not in the right tab



@app.callback(
    [Output('Func on y-dropdown-tab-3', 'options'),
    Output('Func on y-dropdown-tab-3', 'value')],
    Input('y-dropdown-tab-3', 'value'),
    Input('tabs', 'value')  # Include tab value to conditionally trigger callback
)
def update_yfunc_dropdown_tab3(selected_y, selected_tab):
    print()
    print(colored("------------ callback update_yfunc_dropdown_tab3 ------------", "red"))
    print("Active Tab=", selected_tab)
    print("Time computation=", time.time()-start_time)
    if selected_tab == 'tab-3':
        if selected_y is None:
            print("Y Dropdown Value is None, returning an empty list [].")
            return [], []  # Return an empty options list if the DF is not ready
        # Proceed to get options based on selected_x and stored_df1...
        print(f"Selected Y: {selected_y}")  # Additional debugging
        
        function_on_y = ["Avg", "Avg on the ordinate", "Value in x_y interval"]
        
        return update_func_dropdown_utility(selected_y, function_on_y, None)
    return dash.no_update, dash.no_update


@app.callback(
    [Output('Func on z-dropdown-tab-3', 'options'),
    Output('Func on z-dropdown-tab-3', 'value')],
    Input('z-dropdown-tab-3', 'value'),
    Input('tabs', 'value')  # Include tab value to conditionally trigger callback
)
def update_zfunc_dropdown_tab3(selected_z, selected_tab):
    print()
    print(colored("------------ callback update_zfunc_dropdown_tab2 ------------", "red"))
    print("Active Tab=", selected_tab)
    print("Time computation=", time.time()-start_time)
    if selected_tab == 'tab-3':
        if selected_z is None:
            print("Z Dropdown Value is None, returning an empty list [].")
            return [], []  # Return an empty options list if the DF is not ready
        # Proceed to get options based on selected_x and stored_df1...
        print(f"Selected Z: {selected_z}")  # Additional debugging
        
        function_on_z = ["Avg", "Avg on the ordinate", "Weight on y"]
        
        return update_func_dropdown_utility(selected_z, function_on_z, 'Avg')
    return dash.no_update, dash.no_update


@app.callback(
    Output('graph-output-tab-3', 'figure'), Output('figure-store-tab-3', 'data'),
    [Input('tabs', 'value'),
     Input('x-dropdown-tab-3', 'value'),
     Input('y-dropdown-tab-3', 'value'),
     Input('z-dropdown-tab-3', 'value'),
     Input('Func on y-dropdown-tab-3', 'value'),
     Input('Func on z-dropdown-tab-3', 'value'),
     Input('Graph-dropdown-tab-3', 'value'),
     Input('Dim-dropdown-tab-3', 'value'),
     Input("dropdown-smoothing-tab-3", "value"),
     Input("input-smoothing-tab-3", "value"),
     Input("submit-button-smoothing-tab-3", "n_clicks")] +
    [Input(f'fig-dropdown-{col}-tab-3', 'value') for col in List_col_fig_tab3],
    State('stored-df2', 'data')
)
def update_graph_tab3(selected_tab, x_dropdown_value, y_dropdown_value, z_dropdown_value, yfunc_dropdown_value, zfunc_dropdown_value, graph_dropdown_value, dim_dropdown_value, smt_dropdown_value, smt_order_value, sub_bot_smt_value, *args):
    print()
    print(colored("------------ callback update_graph_tab3 ------------", "red")) 
    stored_df2 = args[-1]
    
    # Convert the stored data back to a DataFrame
    df2 = pd.DataFrame(stored_df2)
    # Create a copy of the DataFrame to avoid modifying the original stored data
    filtered_data_table = df2.copy()   

    df_col_numeric = df2.select_dtypes(include=['float64', 'int64']).columns.tolist()
    df_col_all = df2.columns.tolist()
    df_col_string = [col for col in df_col_all if col not in df_col_numeric]  
        
    print("Active Tab:", selected_tab)
    print(filtered_data_table)
    if selected_tab == 'tab-3' and stored_df2 is not None:  # Only execute if in the correct tab
            print(x_dropdown_value, y_dropdown_value, z_dropdown_value, yfunc_dropdown_value, zfunc_dropdown_value, graph_dropdown_value, dim_dropdown_value)
            return update_graph_utility(x_dropdown_value, y_dropdown_value, z_dropdown_value, yfunc_dropdown_value, zfunc_dropdown_value, graph_dropdown_value, dim_dropdown_value, smt_dropdown_value, smt_order_value, sub_bot_smt_value, filtered_data_table, df_col_string, False)




# =============================================================================
# Utility Function for Graphs
# =============================================================================

def update_y_dropdown_utility(selected_x, List_cols, exclude_cols):
    """
    Utility function to generate dropdown options for the y-axis based on the selected x-axis column and dataframe.
    """
    return [{'label': col, 'value': col} for col in List_cols 
                    if col != selected_x and col not in exclude_cols]

def update_z_dropdown_utility(selected_x, selected_y, List_cols, exclude_cols):
    """
    Utility function to generate dropdown options for the z-axis based on the selected x-axis and y-axis column and dataframe.
    """
    return [{'label': col, 'value': col} for col in List_cols 
                    if col not in (selected_x, selected_y) and col not in exclude_cols]

def update_func_dropdown_utility(selected_y, function_on_axi, initial_value=None):
    """
    Utility function to generate dropdown options for the function based on the selected y-axis column.
    """
    
    df_col_numeric = df1.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if selected_y not in df_col_numeric:  # Check if y column is not numeric
        return [], None
    else:
        return [{'label': col, 'value': col} for col in function_on_axi], initial_value

def update_graph_utility(x_column, y_column, z_column, yfunc_column, zfunc_column, graph_type, dim_type, smt_dropdown_value, smt_order_value, sub_bot_smt_value, df, df_col_string, large_file_memory):

    """
    Utility function to generate a graph based on the provided parameters.
    """  
    
    if df is None:  # Check if stored_df is None or empty
        filtered_data_graph = None
    else:
        # Create a copy of the DataFrame to avoid modifying the original stored data
        filtered_data_graph = df.copy()
    # Create the figure based on filtered data

    fig, data_for_plot = fc.create_figure(filtered_data_graph, df_col_string, x_column, y_column, z_column, yfunc_column, zfunc_column, graph_type, dim_type, smt_dropdown_value, smt_order_value, sub_bot_smt_value, large_file_memory)
    return fig, data_for_plot

def update_graph_minor_change_utility(x_column, y_column, z_column, yfunc_column, zfunc_column, graph_type, dim_type, reg_type, reg_order, test_size_val, fig_json_serializable, data_for_plot, df_col_string):
    """
    Utility function to update a graph based on the provided parameters.
    """
    fig, data_for_plot = fc.figure_add_trace(fig_json_serializable, data_for_plot, df_col_string, x_column, y_column, z_column, yfunc_column, zfunc_column, graph_type, dim_type, reg_type, reg_order, test_size_val)
    return fig, data_for_plot

def update_graph_subplot_creation(x_column, y_column, z_column, yfunc_column, zfunc_column, graph_type, dim_type,
                            nb_subplots, nb_subplots_row, nb_subplots_col, current_fig, data_for_plot):
    """
    Utility function to update a graph based on the provided parameters.
    """
    fig, data_for_plot = fc.figure_add_subplot(current_fig, data_for_plot, x_column, y_column, z_column, yfunc_column, zfunc_column, graph_type, dim_type, nb_subplots, nb_subplots_row, nb_subplots_col)
    
    return fig, data_for_plot

def update_graph_subplot(x_column, y_column, z_column, yfunc_column, zfunc_column, graph_type, dim_type,
                         smt_dropdown_value, smt_order_value, sub_bot_smt_value,
                         index_subplot, nb_subplots, nb_subplots_row, nb_subplots_col, df, df_col_string, current_fig, data_for_plot, large_file_memory):
    """
    Utility function to update a graph based on the provided parameters.
    """
    fig, data_for_plot = fc.figure_update_subplot(df, df_col_string, current_fig, data_for_plot, x_column, y_column, z_column,
                                                  yfunc_column, zfunc_column, graph_type, dim_type, 
                                                  smt_dropdown_value, smt_order_value, sub_bot_smt_value,
                                                  index_subplot, nb_subplots, nb_subplots_row, nb_subplots_col, large_file_memory)
    return fig, data_for_plot

# =============================================================================
# End Utility Function for Graphs
# =============================================================================


"""#=============================================================================
   #=============================================================================
   #============================================================================="""

if __name__ == '__main__':
    app.run_server(debug=True, port=8054)

    # Specify the URL you want to open
    url = "http://127.0.0.1:8054/"
    
    # Open the URL in the default web browser
    webbrowser.open(url)

"""#=============================================================================
   #=============================================================================
   #============================================================================="""