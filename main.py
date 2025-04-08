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


import os
import time
import webbrowser
import socket
import psutil
import threading
import requests
import shutil
import json
from termcolor import colored

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State, ALL 
import dash_bootstrap_components as dbc

import pandas as pd
import dask.dataframe as dd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import imageio
import cv2

import Exploratory_Data_Analysis.web_interface_style as wis
import Exploratory_Data_Analysis.dash_callback_ids as dci

import Exploratory_Data_Analysis.table_dropdown_style as tds
import Exploratory_Data_Analysis.figure_creation as fc
import Exploratory_Data_Analysis.data_plot_preparation as dpp
import Exploratory_Data_Analysis.figure_dropdown_style as fds

import Exploratory_Data_Analysis.hypothesis_testing_methods as htm

import Exploratory_Data_Analysis.open_dataframe as od
import Exploratory_Data_Analysis.function_dataframe as fd
import Exploratory_Data_Analysis.debug_dash_infos as ddi

import Exploratory_Data_Analysis.app_state as aps

"""#=============================================================================
   #=============================================================================
   #============================================================================="""

# Save the project on github with: !bash ./save_project_on_git.sh
GitHub_adress= 'https://github.com/KantMG/Exploratory_Data_Analysis'

def Run_Exploratory_Data_Analysis(df1: pd.DataFrame, file_name: str, Large_file_memory: False, debug_info: False):   

    global Debug, previous_clicks, previous_reset_clicks, last_clicked_index
    aps.Debug = debug_info
    Debug = debug_info

    start_time = time.time()      
    
    shape_df1 = df1.shape
    nb_col_df1 = shape_df1[1]
    nb_row_df1 = shape_df1[0]
    
    dtype_counts = df1.dtypes.value_counts()
    
    dtype_df = dtype_counts.reset_index()
    dtype_df.columns = ['dtype', 'count']  # Rename columns for clarity
    dtype_df['dtype'] = dtype_df['dtype'].astype(str)
    
    # Getting numeric description and handling cases with no numeric features
    if df1.select_dtypes(include=[np.number]).empty:
        df1_num_description = pd.DataFrame()  # creates an empty DataFrame
    else:
        df1_num_description = df1.describe(include=[np.number])
        # Calculate the count of NaN values
        df1_num_nan_count = df1.select_dtypes(include=[np.number]).isna().sum()
        df1_num_description.loc['NaN'] = df1_num_nan_count
    
    # Getting object description and handling cases with no object features
    if df1.select_dtypes(include=['object']).empty:
        df1_obj_description = pd.DataFrame()  # creates an empty DataFrame
    else:
        df1_obj_description = df1.describe(include=['object'])
        df1_obj_nan_count = df1.select_dtypes(include=['object']).isna().sum()
        df1_obj_description.loc['NaN'] = df1_obj_nan_count
    
    df1_num_variance =  df1.select_dtypes(include=[np.number]).var()
    df1_num_description.loc['var'] = df1_num_variance.values
        
    if 'var' in df1_num_description.index and 'std' in df1_num_description.index:
        # Create new index order
        new_order = []
    
        # Append rows until 'std' is reached
        for idx in df1_num_description.index:
            new_order.append(idx)
            if idx == 'std':
                # Add the 'var' row immediately after 'std'
                new_order.append('var')
    
        # Remove duplicates maintaining order
        new_order = list(dict.fromkeys(new_order))
    
        # Reindex the DataFrame
        df1_num_description = df1_num_description.reindex(new_order)
        
    df1_num_description = df1_num_description.reset_index()
    df1_obj_description = df1_obj_description.reset_index()
    
    correlation_matrix = df1.corr(numeric_only=True)
    
    object_cols = df1.select_dtypes(include=['object', 'int64']).columns
        
    if len(object_cols) > 1:
        List_crosstab, crosstab_titles, column_pairs = [], [], []
        
        # Create crosstabs for all pairs of object columns and store column pairs
        for i in range(len(object_cols)):
            for j in range(i + 1, len(object_cols)):
                col1 = object_cols[i]
                col2 = object_cols[j]
                column_pairs.append((col1, col2))  # Store the column pairs
                crosstab_titles.append(f"{col1} cross {col2}")
    
        crosstab_options = [{'label': title, 'value': index} for index, title in enumerate(crosstab_titles)]
    
    # info = df1.info()
    memory_info = df1.memory_usage(deep=True)  # Get memory usage of each column
    total_memory_kb = memory_info.sum() / 1024  # Convert bytes to KB
            
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
    
    ddi.debug_print(colored("***************** Start dash ****************", "yellow"), debug=Debug)    
    
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
                html.H1("Dash interface dedicated to the Exploratory Data Analysis of:", style={"color": "#FFD700", 'height': '20px'}, className="text-light"),
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
                tab3_content()
            ])

    app.run_server(debug=True, port=8054)

    # Specify the URL you want to open
    url = "http://127.0.0.1:8054/"
    
    # Open the URL in the default web browser
    webbrowser.open(url)


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
        
        
        tab = 'tab-1'
        
        ddi.debug_print("", debug=Debug) 
        ddi.debug_print(("Time computation=", time.time()-start_time), debug=Debug) 
        ddi.debug_print(colored("=====================  Tab1_content  =========================", "yellow"), debug=Debug) 
        
        # Print all ids
        component_ids = dci.get_component_ids(app.layout)
        ddi.debug_print(("Component IDs:", component_ids), debug=Debug) 
        
        ddi.debug_print(colored("==================== End Tab1_content ========================", "yellow"), debug=Debug) 
        
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
            )
    
            
        data_table_df1_num = tds.dropdown_table(df1_num_description, 'table-df1', tab,
                                         dark_dropdown_style, uniform_style, False)[1]    
        data_table_df1_obj = tds.dropdown_table(df1_obj_description, 'table-df1', tab,
                                         dark_dropdown_style, uniform_style, False)[1]
        
    
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
            )
    
        
        return html.Div([
    
            html.Div([
                html.H2(file_name, style={"color": "#4682B4", 'fontSize': '36px'}, className="text-light"),
                html.P(f"Total memory usage: {total_memory_kb:.2f} KB", style={'fontSize': '24px'}),
                html.P(f"Columns / Rows: {nb_col_df1} / {nb_row_df1}", style={'fontSize': '24px'}),
            ], style={"margin-bottom": "20px"}),  # Added margin bottom
                
            html.Div([
                html.H2("Feature Characteristics:", style={"color": "#FFD700"}, className="text-light"),
            ], style={"margin-bottom": "20px"}),  # Added margin bottom
    
            html.Div(style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '20px'}, children=[
                # Graph on the left for data types distribution
                html.Div(
                    [dcc.Graph(id='dtype-df1', style={'height': '500px'},
                               figure=fig_dtype_df1)], 
                    style={'width': '48%'}
                ),
                # Graph on the right for missing values bar chart
                html.Div(
                    [dcc.Graph(id='heatmap-df1', style={'height': '500px'},
                               figure=fig_missing_bar)], 
                    style={'width': '48%'}
                ),
            ]),
    
            html.Div([
                html.P("Numeric Summary:", style={"color": "#FFD700"}, className="text-light"),
                # html.P(Text5),
            ]),
    
            html.Div([
                html.Div(style={'display': 'flex', 'margin-top': '10px', 'overflowX': 'auto'}, children=[
                    html.Div(data_table_df1_num, style={'width': '100%'})  # Table display
                ])
            ]),
    
            html.Div([
                html.P("\nObject Summary:", style={"color": "#FFD700"}, className="text-light"),
                # html.P(Text5),
            ]),
    
            html.Div([
                html.Div(style={'display': 'flex', 'margin-top': '10px', 'overflowX': 'auto'}, children=[
                    html.Div(data_table_df1_obj, style={'width': '100%'})  # Table display
                ])
            ]),
            
            html.Div([
                html.H2("Feature/Target correlation:", style={"color": "#FFD700"}, className="text-light"),
            ]),
    
    
            html.Div(
                style={'display': 'flex', 'alignItems': 'flex-start'},  # Align at the top if needed
                children=[
                    html.Div(
                        [dcc.Graph(id='correlation-heatmap-df2', style={'width': '100%', 'height': '700px'},
                                   figure=fig_correlation_heatmap)],
                        style={'flex': '1', 'margin-right': '5px'}  # Use flexbox to take full width and small margin
                    ),
                    html.Div(
                        # Dropdown for selecting crosstab figures
                        [
                            dcc.Dropdown(
                                id='crosstab-dropdown',
                                options=crosstab_options,
                                placeholder='Select a Crosstab Heatmap',
                                style={'width': '100%'}  # Full width in the dropdown container
                            ),
                            html.Div(
                                id='selected-crosstab-heatmap',
                                style={'height': '700px', 'overflowY': 'scroll'}
                            )
                        ],
                        style={'flex': '1', 'margin-left': '5px'}  # Use flex for equal width and consistent margin
                    ) if len(object_cols) > 1 else html.Div([
                        html.P("One object column is not enough for crosstab visualization.", style={"color": "#FFD700"}, className="text-light"),
                    ]),
                ]
            ),
    
    
            html.Div([
                html.H3("Hypothesis Testing Methods:", style={"color": "#FFD700"}, className="text-light"),
                # html.P(Text5),
            ]),
                        
                                
            html.Div([
                # Container Div for Target and Feature Selection
                html.Div([
                    # Target Selection Section
                    html.Div([
                        html.Div([
                            html.P("Dependent variable:", className="text-light"),
                        ], style={'margin-right': '10px'}),  # Inline header
            
                        dcc.Dropdown(
                            id='target-value',
                            options=[{'label': val, 'value': val} for val in df1.columns],
                            placeholder='Select variable',
                            style={**dark_dropdown_style, **uniform_style, 'width': '160px'}  # Adjust width as necessary
                        ),
            
                        html.Div([
                            html.P("Select the variable type:", className="text-light"),
                        ], style={'margin-left': '20px', 'margin-right': '10px'}),  # Inline header
            
                        dcc.Dropdown(
                            id='target-type',
                            options=[{'label': val, 'value': val} for val in ["Numerical", "Ordinal", "Nominal"]],
                            placeholder='Select Type',
                            style={**dark_dropdown_style, **uniform_style, 'width': '160px'}  # Adjust width as necessary
                        )
                    ], style={'display': 'flex', 'alignItems': 'center', 'margin-bottom': '20px'}),  # Spacing
            
                    # Feature Selection Section
                    html.Div([
                        html.Div([
                            html.P("Independent variable:", className="text-light"),
                        ], style={'margin-right': '10px'}),  # Inline header
            
                        dcc.Dropdown(
                            id='feature-value',
                            options=[{'label': val, 'value': val} for val in df1.columns],
                            placeholder='Select variable',
                            style={**dark_dropdown_style, **uniform_style, 'width': '160px'}  # Adjust width as necessary
                        ),
            
                        html.Div([
                            html.P("Select the variable type:", className="text-light"),
                        ], style={'margin-left': '20px', 'margin-right': '10px'}),  # Inline header
            
                        dcc.Dropdown(
                            id='feature-type',
                            options=[{'label': val, 'value': val} for val in ["Numerical", "Ordinal", "Nominal"]],
                            placeholder='Select Type',
                            style={**dark_dropdown_style, **uniform_style, 'width': '160px'}  # Adjust width as necessary
                        )
                    ], style={'display': 'flex', 'alignItems': 'center'}),
                ], style={'flex': '1', 'display': 'flex', 'flexDirection': 'column'}),  # Column layout for sections
            
                # Result Brace Div (the "}" character)
                html.Div(id='result-brace'),
                # html.Script(src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS_HTML")
            ], style={'display': 'flex', 'alignItems': 'center', 'padding': '20px', 'border': '1px solid #ccc', 'border-radius': '5px', 'background-color': '#2E2E2E'}),  # Overall styling
    
            
            html.Div(id='dynamic-content-tab1')
    
    
        ], style={'padding': '20px'})
    
    
    @app.callback(
        Output('selected-crosstab-heatmap', 'children'),
        Input('crosstab-dropdown', 'value')
    )
    def update_crosstab_heatmap(selected_index):
        if selected_index is not None:
            # Retrieve col1 and col2 based on selected_index
            col1, col2 = column_pairs[selected_index]
    
            # Create crosstab result
            crosstab_result = pd.crosstab(df1[col1], df1[col2])
    
            # Create heatmap for the crosstab result
            fig_crosstab_object_heatmap = px.imshow(
                crosstab_result,
                color_continuous_scale='RdBu',  # Color scale
                labels=dict(x=col2, y=col1, color='Count'),
                title=f'Crosstab Heatmap: {col1} vs {col2}'
            )
            fig_crosstab_object_heatmap.update_layout(
                plot_bgcolor='#1e1e1e',  # Darker background for the plot area
                paper_bgcolor='#101820',  # Dark gray for the paper
                font=dict(color='white')   # White text color
            )
    
            return dcc.Graph(figure=fig_crosstab_object_heatmap)
        
        return html.Div("Please select a crosstab heatmap")
    
    
    # Callback to update UI based on input value in Tab 3
    @app.callback(
        [Output('dynamic-content-tab1', 'children'),
        Output('result-brace', 'children')],
        Input('target-value', 'value'),
        Input('feature-value', 'value'),
        Input('target-type', 'value'),
        Input('feature-type', 'value')
    )
    def update_feature_value_type(target_input_value, feature_input_value, target_input_type, feature_input_type):
    
        if not feature_input_value or not target_input_type or not feature_input_type:  # Return nothing if input is empty or None
            return '', ''
        
        ddi.debug_print("", debug=Debug) 
        ddi.debug_print(colored("------------ callback update_feature_value_type ------------", "red"), debug=Debug) 

        Hypothesis_test, messages, normality_fig, outlier_table = htm.Hypothesis_Testing_Methods(df1, target_input_value, feature_input_value, target_input_type, feature_input_type)
    
        Hypothesis_test_explanation = htm.get_explanation_on_Hypothesis_test(Hypothesis_test)
        
        return_block = html.Div([
            html.Div(style={'display': 'flex', 'alignItems': 'center'}),  # Flex container for alignment
            # Hypothesis test result without color on "} "
            html.Span("} ", style={'fontSize': '100px', 'color': 'white'}),  # This part is in plain color
            html.Span(Hypothesis_test, style={'fontSize': '70px', 'color': '#006400'}),  # Dark green color
            html.Div(style={'width': '5px', 'backgroundColor': '#FFD700', 'height': '120px', 'marginLeft': '20px', 'marginRight': '20px'}),  # Vertical dark blue line
            # Explanation of the test with equations
            dcc.Markdown(
                Hypothesis_test_explanation, mathjax=True,  # Get the explanation text
                style={'color': 'white', 'fontSize': '20px', 'marginLeft': '10px'}  # Style for the explanation block
            )
        ], style={'display': 'flex', 'alignItems': 'center'})
        
        
        html_output = html.Div([
            dcc.Markdown(msg, className="text-light", mathjax=True) for msg in messages
        ]) 
        # Check if outlier_table is not None
        if outlier_table is not None:
            html_output = html.Div([
                html_output,  # The existing output containing messages and the graph
                html.Div(style={'display': 'flex', 'margin-top': '10px', 'overflowX': 'auto'}, children=[
                    html.Div(outlier_table, style={'width': '100%'})  # Adjusted to take full width
                ])
            ])    
    
        if normality_fig is not None:
            # Prepare the HTML output with messages and the normality figure
            html_output = html.Div([
                html_output
            ] + [
                dcc.Graph(figure=normality_fig)  # Add the generated Plotly figure
            ])
            
        return html.Div([
           
            html_output
    
    
        ], style={'padding': '20px'}), return_block
    
    
    
    
    
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
        
        ddi.debug_print("", debug=Debug) 
        ddi.debug_print(("Time computation=", time.time()-start_time), debug=Debug) 
        ddi.debug_print(colored("=====================  Tab2_content  =========================", "yellow"), debug=Debug) 

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

        # df1.select_dtypes(include=['float64', 'int64']).columns, df1.select_dtypes(include=['object', 'int64']).columns, df1.select_dtypes(include=['object']).columns,    
        button_dropdown_classification_tab2 = fds.button_modal_board_for_machine_learning(
            "maclearning-"+tab, "Machine learning model", "Create your Machine learning model",
            [],[],[],
            "SimpleImputer(strategy = 'mean')", "SimpleImputer(strategy = 'most_frequent')", "SimpleImputer(strategy = 'most_frequent')",
            "StandardScaler()", "OneHotEncoder(handle_unknown='ignore', sparse_output = False)", "OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)",
            ["Classification", "Regression"], "Write your model like: DecisionTreeClassifier()", 0.2, df1.columns,
            dark_dropdown_style, uniform_style)
    

        button_dropdown_smoothing_tab2 = fds.button_modal_dropdown_input("smoothing-"+tab,  "Smoothing", 
                                                                         ["Savitzky-Golay Filter"], "Enter an order if needed",
                                                                         "Select a smoothing function", dark_dropdown_style, uniform_style)
    
        button_subplot_tab2 = fds.button_modal_subplot_creation("subplot-"+tab,  "Subplot creation", 
                                                                         "Number of subplot", "Number of rows", "Number of columns",
                                                                         "Configuration of the subplot figure", dark_dropdown_style, uniform_style)
    
        button_saving_figure_tab2 = fds.button_modal_input_text_dropdown("save-figure-"+tab,  "Save Figure", 
                                                                         ["png", "jpeg", "webp", "svg", "pdf", "gif", "mp4"], "Enter the figure name",
                                                                         "png", "Save Figure", dark_dropdown_style, uniform_style)
    
    
    
        component_ids = dci.get_component_ids(app.layout)
        ddi.debug_print(("Component IDs:", component_ids), debug=Debug) 
        ddi.debug_print(colored("==================== End Tab2_content ========================", "yellow"), debug=Debug) 
        return html.Div([
            html.Div([
                fds.figure_position_dash(tab,
                                         'graph-output-'+tab,
                                         dropdowns_with_labels_for_fig_tab2,
                                         dropdowns_with_labels_for_fig_filter_tab2,
                                         button_dropdown_function_tab2,
                                         button_dropdown_classification_tab2,
                                         button_dropdown_smoothing_tab2,
                                         button_subplot_tab2,
                                         button_saving_figure_tab2
                                         )
            ], style={'padding': '20px'}),
            
            html.Div(id='dynamic-content-tab2')
            
        ], style={'padding': '20px'})
        
    
    
    
    @app.callback(
        Output('dynamic-content-tab2', 'children'), 
        Input('datatable-button-tab-2', "n_clicks"),
        [Input(f'fig-dropdown-{col}-tab-2', 'value') for col in List_col_tab2]
    )
    def update_tableau_show(n_clic, *args):
        
        
        if not n_clic or n_clic % 2 == 0:
            return ''
    
        ddi.debug_print(colored("------------ callback update_tableau_show ------------", "red"), debug=Debug) 
    
        tab = "tab-2"
        
        filter_values = list(args[0:len(List_col_tab2)])
        filter_values = {List_col_tab2[i]: (filter_values[i] if filter_values[i] != '' else None) for i in range(min(len(List_col_tab2), len(filter_values)))}    
        df1_filtered = od.apply_filter(df1, filter_values)
        
        # Create the table with the appropriate dropdowns for each column
        data_table_df1_ta2 = tds.table_with_filter_action(df1_filtered, 'table-df1-tab2', tab, dark_dropdown_style, uniform_style, False)
                  
    
        return html.Div([
    
            html.Div(style={'display': 'flex', 'margin-top': '10px', 'overflowX': 'auto'}, children=[
                html.Div(data_table_df1_ta2, style={'width': '100%'})  # Adjusted to take full width
            ]),
    
                            
        ], style={'padding': '20px'})
    
    
    @app.callback(
        Output('datatable-button-tab-2', 'children'), 
        Input('datatable-button-tab-2', "n_clicks")
    )
    def update_tableau_show(n_clic):
        
        
        if not n_clic or n_clic % 2 == 0:
            return "Show Data Table"
        else:
            return "Hide Data Table"
    
    
    
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
        ddi.debug_print(("Submit button clicks:", n_clicks), debug=Debug) 
        ddi.debug_print(("Function Name:", func_name), debug=Debug) 
        ddi.debug_print(("Input Value:", input_value), debug=Debug)         
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
        ddi.debug_print("", debug=Debug) 
        ddi.debug_print(colored("------------ callback update_x_dropdown_tab2 ------------", "red"), debug=Debug) 
        return [{'label': col, 'value': col} for col in df1.columns if col not in List_col_exclude_tab2]
        
    #  -----------------------------------------------------------------
    
    @app.callback(
        Output("modal-maclearning-tab-2", "is_open"),
        [Input("open-modal-maclearning-tab-2", "n_clicks"), Input("submit-button-maclearning-tab-2", "n_clicks"), Input("close-button-maclearning-tab-2", "n_clicks")],
        [State("modal-maclearning-tab-2", "is_open")]
    )
    def toggle_modal(open_clicks, submit_clicks, close_clicks, is_open):
        if open_clicks or submit_clicks or close_clicks:
            return not is_open
        return is_open


    @app.callback(
        Output('target-maclearning-tab-2', 'options'),
        [Input('x-dropdown-tab-2', 'value')]+
        [Input('y-dropdown-tab-2', 'value')]+
        [Input('z-dropdown-tab-2', 'value')]+
        [Input('t-dropdown-tab-2', 'value')]
    )
    def update_ml_option_target(selected_x, selected_y, selected_z, selected_t):
        ddi.debug_print("", debug=Debug) 
        ddi.debug_print(colored("------------ callback update_ml_option_target ------------", "red"), debug=Debug) 

        # List of selected values
        selected_values = [selected_x, selected_y, selected_z, selected_t]
        
        # Filter out None values
        non_none_selected = [val for val in selected_values if val in df1.columns]

        return non_none_selected


    @app.callback(
        [Output("target-type-maclearning-tab-2", "options"),
         Output("num-features-maclearning-tab-2", "options"),
         Output("ode-features-maclearning-tab-2", "options"),
         Output("ohe-features-maclearning-tab-2", "options")],
        Input("target-maclearning-tab-2", "value"),
        Input("target-maclearning-tab-2", "options")
    )
    def update_dropdown_modal(target_value, target_option):

        ddi.debug_print("", debug=Debug) 
        ddi.debug_print(colored("-------- callback update_dropdown_modal --------", "red"), debug=Debug) 
        
        num_fea_option = df1[target_option].select_dtypes(include=['float64', 'int64']).columns
        ode_fea_option = df1[target_option].select_dtypes(include=['object', 'int64']).columns
        ohe_fea_option = df1[target_option].select_dtypes(include=['object']).columns
        
        tar_typ_option = []
        if target_value is not None:
            # Determine the type of target_value and update tar_typ_option
            if pd.api.types.is_float_dtype(df1[target_value].dtype):
                tar_typ_option = ["numerical"]
            elif pd.api.types.is_integer_dtype(df1[target_value].dtype):
                tar_typ_option = ["numerical", "ordinal"]
            elif pd.api.types.is_object_dtype(df1[target_value].dtype):
                tar_typ_option = ["ordinal", "nominal"]       
        
            # Remove target_value from num_fea_option if it is present
            num_fea_option_updated = [option for option in num_fea_option if option != target_value]
            
            # Remove target_value from ode_fea_option if it is present
            ode_fea_option_updated = [option for option in ode_fea_option if option != target_value]
            
            # Remove target_value from ohe_fea_option if it is present
            ohe_fea_option_updated = [option for option in ohe_fea_option if option != target_value]
            
            return tar_typ_option, num_fea_option_updated, ode_fea_option_updated, ohe_fea_option_updated        

        else:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update


    @app.callback(
        Output("type-model-maclearning-tab-2", "options"),
        Input("target-type-maclearning-tab-2", "value")
    )
    def update_dropdown_model_type_modal(target_type_value):
                    
        if target_type_value == "numerical":
            return ["Regression"]
        elif target_type_value == "nominal":
            return ["Classification"]
        else:
            return ["Classification", "Regression"]


    @app.callback(
        Output("model-maclearning-tab-2", "value"),
        Input("type-model-maclearning-tab-2", "value")
    )
    def update_input_modal(type_model):
        if type_model == "Regression":
            return "LinearRegression()"
        if type_model == "Classification":
            return "DecisionTreeClassifier()"
        if type_model == "Clustering":
            return "KMeans(n_clusters=3, random_state=42)"
        return None
    
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
        Output("modal-save-figure-tab-2", "is_open"),
        [Input("open-modal-save-figure-tab-2", "n_clicks"), Input("submit-button-save-figure-tab-2", "n_clicks")],
        [State("modal-save-figure-tab-2", "is_open")]
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
        
        ddi.debug_print(colored("-------------- callback update_output --------------", "red"), debug=Debug) 
        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        ddi.debug_print(("Triggered component:", triggered_id), debug=Debug) 
        ddi.debug_print("", debug=Debug) 
        
        ddi.debug_print(("Submit button clicks:", n_clicks), debug=Debug) 
        ddi.debug_print(("Inputs Name:", [input_1_value, input_2_value, input_3_value]), debug=Debug) 
        
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
                buttons_subplot_tab2 = fds.buttons_subplots("Figure-tab-2-subplot-", "Subplot ",
                                                            input_1_value, input_2_value, input_3_value, dark_dropdown_style, uniform_style)
                
                previous_clicks = [0] * input_1_value
                return buttons_subplot_tab2
            except Exception as e:
                return f"Error: {str(e)}"
        return ""
    
    
    #  -----------------------------------------------------------------
    
    @app.callback(
        [Output('x-dropdown-tab-2', 'value'),
        Output('y-dropdown-tab-2', 'value'),
        Output('z-dropdown-tab-2', 'value'),
        Output('t-dropdown-tab-2', 'value')],
        [Input('x-dropdown-tab-2', 'value')]+
        [Input('y-dropdown-tab-2', 'value')]+
        [Input('z-dropdown-tab-2', 'value')]+
        [Input('t-dropdown-tab-2', 'value')]+
        [Input('tabs', 'value')]
    )
    def update_value_default_tab2(selected_x, selected_y, selected_z, selected_t, selected_tab):
        ddi.debug_print("", debug=Debug) 
        ddi.debug_print(colored("------------ callback update_value_default_tab2 ------------", "red"), debug=Debug) 
        ddi.debug_print(("Active Tab=", selected_tab), debug=Debug) 
        ddi.debug_print(("Time computation=", time.time()-start_time), debug=Debug) 

        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        ddi.debug_print(("Triggered component:", triggered_id), debug=Debug) 
        ddi.debug_print("", debug=Debug) 
        
        
        if selected_tab == 'tab-2':
                        
            if selected_x is None:
                ddi.debug_print("X Dropdown Value is None, returning an empty list [].", debug=Debug) 
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update
            
            elif selected_x is not None and triggered_id == 'x-dropdown-tab-2' and selected_z is None:
                return dash.no_update, "count", dash.no_update, dash.no_update
                        
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    
    @app.callback(
        Output('y-dropdown-tab-2', 'options'),
        [Input('x-dropdown-tab-2', 'value')]+
        [Input('tabs', 'value')]
    )
    def update_y_dropdown_tab2(selected_x, selected_tab):
        ddi.debug_print("", debug=Debug) 
        ddi.debug_print(colored("------------ callback update_y_dropdown_tab2 ------------", "red"), debug=Debug) 
        ddi.debug_print(("Active Tab=", selected_tab), debug=Debug) 
        ddi.debug_print(("Time computation=", time.time()-start_time), debug=Debug) 
        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        ddi.debug_print(("Triggered component:", triggered_id), debug=Debug) 
        ddi.debug_print("", debug=Debug) 

        if selected_tab == 'tab-2':
            if selected_x is None:
                ddi.debug_print("X Dropdown Value is None, returning an empty list [].", debug=Debug) 
                return []
            ddi.debug_print(f"Selected X: {selected_x}", debug=Debug) 
            exclude_cols=List_col_exclude_tab2
            return update_y_dropdown_utility(selected_x, df1.columns.tolist(), exclude_cols)
        return dash.no_update
    
    
    @app.callback(
        Output('z-dropdown-tab-2', 'options'),
        [Input('x-dropdown-tab-2', 'value'),
         Input('y-dropdown-tab-2', 'value'),
        Input('tabs', 'value')]
    )
    def update_z_dropdown_tab2(selected_x, selected_y, selected_tab):
        ddi.debug_print("", debug=Debug) 
        ddi.debug_print(colored("------------ callback update_z_dropdown_tab2 ------------", "red"), debug=Debug) 
        ddi.debug_print(("Active Tab=", selected_tab), debug=Debug) 
        ddi.debug_print(("Time computation=", time.time()-start_time), debug=Debug) 

        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        ddi.debug_print(("Triggered component:", triggered_id), debug=Debug) 
        ddi.debug_print("", debug=Debug) 
        if selected_tab == 'tab-2':
            if selected_y is None:
                ddi.debug_print("Y Dropdown Value is None, returning an empty list [].", debug=Debug) 
                return []
            ddi.debug_print(f"Selected y: {selected_y}", debug=Debug) 
            exclude_cols=List_col_exclude_tab2
            return update_z_dropdown_utility(selected_x, selected_y, df1.columns.tolist(), exclude_cols)
        return dash.no_update
    
    
    @app.callback(
        Output('t-dropdown-tab-2', 'options'),
        [Input('x-dropdown-tab-2', 'value'),
         Input('y-dropdown-tab-2', 'value'),
         Input('z-dropdown-tab-2', 'value'),
        Input('tabs', 'value')]
    )
    def update_t_dropdown_tab2(selected_x, selected_y, selected_z, selected_tab):
        ddi.debug_print("", debug=Debug) 
        ddi.debug_print(colored("------------ callback update_t_dropdown_tab2 ------------", "red"), debug=Debug) 
        ddi.debug_print(("Active Tab=", selected_tab), debug=Debug) 
        ddi.debug_print(("Time computation=", time.time()-start_time), debug=Debug) 
        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        ddi.debug_print(("Triggered component:", triggered_id), debug=Debug) 
        ddi.debug_print("", debug=Debug) 
        if selected_tab == 'tab-2':
            if selected_y is None:
                ddi.debug_print("Y Dropdown Value is None, returning an empty list [].", debug=Debug) 
                return []
            ddi.debug_print(f"Selected y: {selected_y}", debug=Debug) 
            exclude_cols=List_col_exclude_tab2
            return update_t_dropdown_utility(selected_x, selected_y, selected_z, df1.columns.tolist(), exclude_cols)
        return dash.no_update
    
    
    @app.callback(
        [Output('Func on y-dropdown-tab-2', 'options'),
        Output('Func on y-dropdown-tab-2', 'value')],
        Input('y-dropdown-tab-2', 'value'),
        Input('tabs', 'value')  # Include tab value to conditionally trigger callback
    )
    def update_yfunc_dropdown_tab2(selected_y, selected_tab):
        ddi.debug_print("", debug=Debug) 
        ddi.debug_print(colored("------------ callback update_yfunc_dropdown_tab2 ------------", "red"), debug=Debug) 
        ddi.debug_print(("Active Tab=", selected_tab), debug=Debug) 
        ddi.debug_print(("Time computation=", time.time()-start_time), debug=Debug) 
        if selected_tab == 'tab-2':
            if selected_y is None:
                ddi.debug_print("Y Dropdown Value is None, returning an empty list [].", debug=Debug) 
                return [], []
            ddi.debug_print(f"Selected Y: {selected_y}", debug=Debug)   # Additional debugging
            
            function_on_y = ["Avg"]
            
            return update_func_dropdown_utility(selected_y, function_on_y, None)
        return dash.no_update, dash.no_update
    
    @app.callback(
        [Output('Func on z-dropdown-tab-2', 'options'),
        Output('Func on z-dropdown-tab-2', 'value')],
        Input('z-dropdown-tab-2', 'value'),
        Input('tabs', 'value')  # Include tab value to conditionally trigger callback
    )
    def update_zfunc_dropdown_tab2(selected_z, selected_tab):
        ddi.debug_print("", debug=Debug) 
        ddi.debug_print(colored("------------ callback update_zfunc_dropdown_tab2 ------------", "red"), debug=Debug) 
        ddi.debug_print(("Active Tab=", selected_tab), debug=Debug) 
        ddi.debug_print(("Time computation=", time.time()-start_time), debug=Debug) 
        if selected_tab == 'tab-2':
            if selected_z is None:
                ddi.debug_print("Z Dropdown Value is None, returning an empty list [].", debug=Debug) 
                return [], []  # Return an empty options list if the DF is not ready
            # Proceed to get options based on selected_x and stored_df1..
            ddi.debug_print(f"Selected Z: {selected_z}", debug=Debug)   # Additional debugging
            
            function_on_z = ["Avg"]
            
            return update_func_dropdown_utility(selected_z, function_on_z, 'Avg')
        return dash.no_update, dash.no_update
    
    
    @app.callback(
        [Output('Func on t-dropdown-tab-2', 'options'),
        Output('Func on t-dropdown-tab-2', 'value')],
        Input('t-dropdown-tab-2', 'value'),
        Input('tabs', 'value')  # Include tab value to conditionally trigger callback
    )
    def update_tfunc_dropdown_tab2(selected_t, selected_tab):
        ddi.debug_print("", debug=Debug) 
        ddi.debug_print(colored("------------ callback update_tfunc_dropdown_tab2 ------------", "red"), debug=Debug) 
        ddi.debug_print(("Active Tab=", selected_tab), debug=Debug) 
        ddi.debug_print(("Time computation=", time.time()-start_time), debug=Debug) 
        if selected_tab == 'tab-2':
            if selected_t is None:
                ddi.debug_print("T Dropdown Value is None, returning an empty list [].", debug=Debug) 
                return [], []  # Return an empty options list if the DF is not ready
            # Proceed to get options based on selected_x and stored_df1...
            ddi.debug_print(f"Selected T: {selected_t}", debug=Debug)   # Additional debugging
            
            function_on_t = ["Avg"]
            
            return update_func_dropdown_utility(selected_t, function_on_t, 'Avg')
        return dash.no_update, dash.no_update
    
    
    
    
    @app.callback(
        Output('Dim-dropdown-tab-2', 'options'),
        Input('y-dropdown-tab-2', 'value'),
        Input('tabs', 'value')  # Include tab value to conditionally trigger callback
    )
    def update_dim_dropdown_tab2(selected_y, selected_tab):
        ddi.debug_print("", debug=Debug) 
        ddi.debug_print(colored("-------- callback update_dim_dropdown_tab2 --------", "red"), debug=Debug) 
        ddi.debug_print(("Active Tab=", selected_tab), debug=Debug) 
        ddi.debug_print(("Time computation=", time.time()-start_time), debug=Debug) 
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
        ddi.debug_print("", debug=Debug) 
        ddi.debug_print(colored("------------ callback update_graph_dropdown_tab2 ------------", "red"), debug=Debug) 
        ddi.debug_print(("Active Tab=", selected_tab), debug=Debug) 
        ddi.debug_print(("Time computation=", time.time()-start_time), debug=Debug) 
        if selected_tab == 'tab-2':
            if selected_dim == "1D":
                return [{'label': col, 'value': col} for col in List_graph_type if col not in ("Colormesh", "Pie")], 'Histogram'
            if selected_dim == "2D":
                return [{'label': col, 'value': col} for col in List_graph_type if col not in ("Histogram", "Curve", "Scatter", "Histogram Movie", "Curve Movie", "Scatter Movie", "Boxes")], None
            if selected_dim == "3D":
                return [{'label': col, 'value': col} for col in ["Scatter"]], None
        return dash.no_update, dash.no_update 
 
    @app.callback(
        Output('graph-output-tab-2', 'figure'), Output('figure-store-tab-2', 'data'),
        [Input('tabs', 'value'),
         Input('x-dropdown-tab-2', 'value'),
         Input('y-dropdown-tab-2', 'value'),
         Input('z-dropdown-tab-2', 'value'),
         Input('t-dropdown-tab-2', 'value'),
         Input('Func on y-dropdown-tab-2', 'value'),
         Input('Func on z-dropdown-tab-2', 'value'),
         Input('Func on t-dropdown-tab-2', 'value'),
         Input('Graph-dropdown-tab-2', 'value'),
         Input('Dim-dropdown-tab-2', 'value'),
         
         Input('type-model-maclearning-tab-2', "value"),
         Input('target-maclearning-tab-2', "value"),
         Input('target-type-maclearning-tab-2', "value"),
         Input('test-size-maclearning-tab-2', "value"),
         Input('num-features-maclearning-tab-2', "value"),
         Input('num-imputer-maclearning-tab-2', "value"),
         Input('num-encoder-maclearning-tab-2', "value"),
         Input('ode-features-maclearning-tab-2', "value"),
         Input('ode-imputer-maclearning-tab-2', "value"),
         Input('ode-encoder-maclearning-tab-2', "value"),
         Input('ohe-features-maclearning-tab-2', "value"),
         Input('ohe-imputer-maclearning-tab-2', "value"),
         Input('ohe-encoder-maclearning-tab-2', "value"),
         Input('model-maclearning-tab-2', "value"),
         Input("submit-button-maclearning-tab-2", "n_clicks"),
         
         Input("dropdown-smoothing-tab-2", "value"),
         Input("input-smoothing-tab-2", "value"),
         Input("submit-button-smoothing-tab-2", "n_clicks"),
         Input("input_1-subplot-tab-2", "value"),
         Input("input_2-subplot-tab-2", "value"),
         Input("input_3-subplot-tab-2", "value"),
         Input("dropdown-save-figure-tab-2", "value"),
         Input("input-save-figure-tab-2", "value"),
         Input("submit-button-save-figure-tab-2", "n_clicks"),     
         Input("submit-button-filter-tab-2", "n_clicks")] +
        [Input(f'fig-dropdown-{col}-tab-2', 'value') for col in List_col_tab2] +
        [Input({'type': 'subplot-button', 'index': ALL}, 'n_clicks')],
        State('graph-output-tab-2', 'figure'),
        State('figure-store-tab-2', 'data')
        )
    def update_graph_tab2(selected_tab, x_dropdown_value, y_dropdown_value, z_dropdown_value, t_dropdown_value,
                          yfunc_dropdown_value, zfunc_dropdown_value, tfunc_dropdown_value, graph_dropdown_value, dim_dropdown_value,
                          type_model, cla_tar, cla_tar_type, cla_size, 
                          cla_num_fea, cla_num_imp, cla_num_enc,
                          cla_ode_fea, cla_ode_imp, cla_ode_enc,
                          cla_ohe_fea, cla_ohe_imp, cla_ohe_enc,
                          cla_model, cla_submit,
                          smt_dropdown_value, smt_order_value, sub_bot_smt_value,
                          nb_subplots, nb_subplots_row, nb_subplots_col,
                          svf_dropdown_value, svf_order_value, sub_bot_svf_value,
                          sub_bot_filter_value, *args):
    
        global previous_clicks, last_clicked_index
        ddi.debug_print("", debug=Debug) 
        ddi.debug_print(colored("------------ callback update_graph_tab2 ------------", "red"), debug=Debug) 
        current_fig = args[-2]
        data_for_plot = args[-1]
        filter_values = list(args[0:len(List_col_tab2)])
        filter_values = {List_col_tab2[i]: (filter_values[i] if filter_values[i] != '' else None) for i in range(min(len(List_col_tab2), len(filter_values)))}
        subplot_button_clicks = list(args[len(List_col_tab2):-2])
        
        # Now to get the flat list
        if subplot_button_clicks and isinstance(subplot_button_clicks, list):
            subplot_button_clicks = subplot_button_clicks[0]  # Access the first element
            ddi.debug_print(("Subplot Button Clicks:", subplot_button_clicks), debug=Debug) 
        else:
            subplot_button_clicks = []  # Handle cases where subplot_button_clicks might be empty or wrongly structured
            ddi.debug_print(("No subplot button clicks found."), debug=Debug) 
    
        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        ddi.debug_print(("Triggered component:", triggered_id), debug=Debug) 
        
        
        if triggered_id in ["dropdown-smoothing-tab-2", "input-smoothing-tab-2", "input_1-subplot-tab-2", "input_2-subplot-tab-2", "input_3-subplot-tab-2", "dropdown-save-figure-tab-2", "input-save-figure-tab-2" ]:
            return dash.no_update


        if triggered_id in ['type-model-maclearning-tab-2', 'target-maclearning-tab-2', 'target-type-maclearning-tab-2',  'test-size-maclearning-tab-2',
                            'num-features-maclearning-tab-2', 'num-imputer-maclearning-tab-2', 'num-encoder-maclearning-tab-2',
                            'ode-features-maclearning-tab-2', 'ode-imputer-maclearning-tab-2', 'ode-encoder-maclearning-tab-2',
                            'ohe-features-maclearning-tab-2', 'ohe-imputer-maclearning-tab-2', 'ohe-encoder-maclearning-tab-2',
                            'model-maclearning-tab-2']:
            return dash.no_update

    
        df_col_numeric = df1.select_dtypes(include=['float64', 'int64']).columns.tolist()
        df_col_numeric.append('count')
        df_col_all = df1.columns.tolist()
        df_col_all.append('count')
        df_col_string = [col for col in df_col_all if col not in df_col_numeric]   

        ddi.debug_print(("Active Tab=", selected_tab), debug=Debug) 
        ddi.debug_print(("Time computation=", time.time()-start_time), debug=Debug) 
    
        df1_filtered = od.apply_filter(df1, filter_values)
            
        if graph_dropdown_value is None:
            ddi.debug_print("Please select a graphic type.", debug=Debug) 
            return dash.no_update
        
        if triggered_id in list([f'fig-dropdown-{col}-tab-2' for col in List_col_tab2]):
            return dash.no_update
    
        if triggered_id == "submit-button-maclearning-tab-2":
            return update_graph_minor_change_utility(x_dropdown_value, y_dropdown_value, z_dropdown_value, t_dropdown_value,
                                                     yfunc_dropdown_value, zfunc_dropdown_value, tfunc_dropdown_value, 
                                                     graph_dropdown_value, dim_dropdown_value,
                                                     type_model, cla_tar, cla_tar_type, cla_size, 
                                                     cla_num_fea, cla_num_imp, cla_num_enc,
                                                     cla_ode_fea, cla_ode_imp, cla_ode_enc,
                                                     cla_ohe_fea, cla_ohe_imp, cla_ohe_enc,
                                                     cla_model,
                                                     current_fig, df1_filtered, df_col_string)
    
    
        if  triggered_id == "submit-button-save-figure-tab-2":
            fig_json_serializable = go.Figure(current_fig)
            filename = svf_order_value
            format_fig = svf_dropdown_value
            fig_json_serializable["layout"]["updatemenus"] = [] 
            
            if format_fig != "gif" and format_fig != "mp4":
                pio.write_image(fig_json_serializable, f"{filename}.{format_fig}")
                
            else:
                # Step 1: Save each frame as an image
                frame_filenames = []  # Keep track of frame filenames
                current_fig = fig_json_serializable
                for i, frame in enumerate(fig_json_serializable.frames):
                    
                    # Create a new figure for the current frame
                    frame_fig = go.Figure()
                    
                    # Append the current frame's data to the new figure
                    for data in frame.data:
                        frame_fig.add_trace(data)
                    
                    frame_fig.update_layout(current_fig.layout)
                    frame_filename = f"frame_{i}.png"  # Specify a filename
                    pio.write_image(frame_fig, frame_filename)  # Save the current frame
                    frame_filenames.append(frame_filename)  # Add to the list
                
                if format_fig == "gif":
                    duration_frame = round(10 / i)
                    
                    # Step 2: Create a GIF from the saved frames
                    gif_filename = f"{filename}.{format_fig}"
                    with imageio.get_writer(gif_filename, mode='I', duration=duration_frame) as writer:  # Set duration for frame display
                        for frame_filename in frame_filenames:
                            image = imageio.imread(frame_filename)
                            writer.append_data(image)
                            
                if format_fig == "mp4":
                    if i<20:
                        frame_per_second = i
                    else:
                        frame_per_second = 10
                    
                    # Step 2: Create an MP4 from the saved frames
                    video_filename = f"{filename}.{format_fig}"
                    frame = cv2.imread(frame_filenames[0])  # Read the first frame to get dimensions
                    height, width, layers = frame.shape
                    
                    # Create a video writer object
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
                    video = cv2.VideoWriter(video_filename, fourcc, frame_per_second, (width, height))  # 10 fps
                    
                    # Write each frame to the video
                    for frame_filename in frame_filenames:
                        frame = cv2.imread(frame_filename)
                        video.write(frame)
                    
                    # Release the video writer object
                    video.release()
                    cv2.destroyAllWindows()
                    
                    print(f"Video saved as {video_filename}")        
                
                
                
            
            return dash.no_update
        
        
        ###################### Subplot part ######################
        
        
        # Parse the triggered_id if it is a JSON string
        try:
            parsed_id = json.loads(triggered_id)
        except json.JSONDecodeError:
            ddi.debug_print("Could not decode JSON.", debug=Debug) 
            parsed_id = {}
    
        # Check if the parsed_id corresponds to a subplot button
        if parsed_id.get('type') == 'subplot-button':
            ddi.debug_print('subplot-button type, return no update.', debug=Debug) 
            if all(click == 0 for click in subplot_button_clicks) :
    
                return update_graph_subplot_creation(x_dropdown_value, y_dropdown_value, z_dropdown_value, t_dropdown_value,
                                            yfunc_dropdown_value, zfunc_dropdown_value, tfunc_dropdown_value,
                                            graph_dropdown_value, dim_dropdown_value,
                                            nb_subplots, nb_subplots_row, nb_subplots_col,
                                            current_fig, data_for_plot)        
            
            else:
                for index, (prev, curr) in enumerate(zip(previous_clicks, subplot_button_clicks)):
                    if curr > prev:  # If current clicks > previous clicks, this button was clicked
                        last_clicked_index = index
                        ddi.debug_print(f"Subplot button at index {index} was clicked.", debug=Debug) 
        
                        # Update the previous clicks
                        previous_clicks = subplot_button_clicks.copy()
                return dash.no_update
        
        # Check whether subplot_button_clicks is valid and not empty
        if not subplot_button_clicks:
            ddi.debug_print("No subplot buttons have been clicked. The figure is unique.", debug=Debug) 
        elif all(x == 0 for x in subplot_button_clicks):
            ddi.debug_print("Subplot buttons are all 0.", debug=Debug) 
            return update_graph_subplot(x_dropdown_value, y_dropdown_value, z_dropdown_value, t_dropdown_value,
                                        yfunc_dropdown_value, zfunc_dropdown_value, tfunc_dropdown_value,
                                        graph_dropdown_value, dim_dropdown_value,
                                        smt_dropdown_value, smt_order_value, sub_bot_smt_value,
                                        0, nb_subplots, nb_subplots_row, nb_subplots_col,
                                        df1_filtered, df_col_string, current_fig, data_for_plot, Large_file_memory)
        else:           
            # Use last_clicked_index for any needed logic
            if last_clicked_index is not None:
                ddi.debug_print(f"Last clicked subplot button index: {last_clicked_index}", debug=Debug) 
                # Additional logic based on the last clicked button can go here
    
            return update_graph_subplot(x_dropdown_value, y_dropdown_value, z_dropdown_value, t_dropdown_value,
                                        yfunc_dropdown_value, zfunc_dropdown_value, tfunc_dropdown_value,
                                        graph_dropdown_value, dim_dropdown_value,
                                        smt_dropdown_value, smt_order_value, sub_bot_smt_value,
                                        last_clicked_index, nb_subplots, nb_subplots_row, nb_subplots_col,
                                        df1_filtered, df_col_string, current_fig, data_for_plot, Large_file_memory)
                
                
        
        return update_graph_utility(x_dropdown_value, y_dropdown_value, z_dropdown_value, t_dropdown_value, yfunc_dropdown_value, zfunc_dropdown_value, tfunc_dropdown_value, graph_dropdown_value, dim_dropdown_value, smt_dropdown_value, smt_order_value, sub_bot_smt_value, df1_filtered, df_col_string, Large_file_memory)
    
    
    
    # =============================================================================
    # Utility Function for Graphs
    # =============================================================================
    
    def update_y_dropdown_utility(selected_x, List_cols, exclude_cols):
        """
        Utility function to generate dropdown options for the y-axis based on the selected x-axis column and dataframe.
        """
        List_cols.append('count')
        return [{'label': col, 'value': col} for col in List_cols 
                        if col != selected_x and col not in exclude_cols]
    
    def update_z_dropdown_utility(selected_x, selected_y, List_cols, exclude_cols):
        """
        Utility function to generate dropdown options for the z-axis based on the selected x-axis and y-axis column and dataframe.
        """
        List_cols.append('count')
        return [{'label': col, 'value': col} for col in List_cols 
                        if col not in (selected_x, selected_y) and col not in exclude_cols]
    
    def update_t_dropdown_utility(selected_x, selected_y, selected_z, List_cols, exclude_cols):
        """
        Utility function to generate dropdown options for the t-axis based on the selected x-axis, y-axis and z-axis column and dataframe.
        """
        List_cols.append('count')
        return [{'label': col, 'value': col} for col in List_cols 
                        if col not in (selected_x, selected_y, selected_z) and col not in exclude_cols]
    
    def update_func_dropdown_utility(selected_y, function_on_axi, initial_value=None):
        """
        Utility function to generate dropdown options for the function based on the selected y-axis column.
        """
        
        df_col_numeric = df1.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if selected_y not in df_col_numeric:  # Check if y column is not numeric
            return [], None
        else:
            return [{'label': col, 'value': col} for col in function_on_axi], initial_value
    
    def update_graph_utility(x_column, y_column, z_column, t_column, yfunc_column, zfunc_column, tfunc_column, graph_type, dim_type, smt_dropdown_value, smt_order_value, sub_bot_smt_value, df, df_col_string, large_file_memory):
    
        """
        Utility function to generate a graph based on the provided parameters.
        """  
        
        if df is None:  # Check if stored_df is None or empty
            filtered_data_graph = None
        else:
            # Create a copy of the DataFrame to avoid modifying the original stored data
            filtered_data_graph = df.copy()
        # Create the figure based on filtered data
        fig, data_for_plot = fc.create_figure(filtered_data_graph, df_col_string, x_column, y_column, z_column, t_column, yfunc_column, zfunc_column, tfunc_column, graph_type, dim_type, smt_dropdown_value, smt_order_value, sub_bot_smt_value, large_file_memory)
        return fig, data_for_plot
    
    def update_graph_minor_change_utility(x_column, y_column, z_column, t_column, yfunc_column, zfunc_column, tfunc_column, graph_type, dim_type, 
                                          type_model, cla_tar, cla_tar_type, cla_size, 
                                          cla_num_fea, cla_num_imp, cla_num_enc,
                                          cla_ode_fea, cla_ode_imp, cla_ode_enc,
                                          cla_ohe_fea, cla_ohe_imp, cla_ohe_enc,
                                          cla_model,
                                          fig_json_serializable, data_for_plot, df_col_string):
        """
        Utility function to update a graph based on the provided parameters.
        """
        fig, data_for_plot = fc.figure_add_trace(fig_json_serializable, data_for_plot, df_col_string, x_column, y_column, z_column, t_column, yfunc_column, zfunc_column, tfunc_column, graph_type, dim_type,
                                                 type_model, cla_tar, cla_tar_type, cla_size, 
                                                 cla_num_fea, cla_num_imp, cla_num_enc,
                                                 cla_ode_fea, cla_ode_imp, cla_ode_enc,
                                                 cla_ohe_fea, cla_ohe_imp, cla_ohe_enc,
                                                 cla_model
                                                 )
        return fig, data_for_plot
    
    def update_graph_subplot_creation(x_column, y_column, z_column, t_column, yfunc_column, zfunc_column, tfunc_column, graph_type, dim_type,
                                nb_subplots, nb_subplots_row, nb_subplots_col, current_fig, data_for_plot):
        """
        Utility function to update a graph based on the provided parameters.
        """
        fig, data_for_plot = fc.figure_add_subplot(current_fig, data_for_plot, x_column, y_column, z_column, t_column, yfunc_column, zfunc_column, tfunc_column, graph_type, dim_type, nb_subplots, nb_subplots_row, nb_subplots_col)
        
        return fig, data_for_plot
    
    def update_graph_subplot(x_column, y_column, z_column, t_column, yfunc_column, zfunc_column, tfunc_column, graph_type, dim_type,
                             smt_dropdown_value, smt_order_value, sub_bot_smt_value,
                             index_subplot, nb_subplots, nb_subplots_row, nb_subplots_col, df, df_col_string, current_fig, data_for_plot, large_file_memory):
        """
        Utility function to update a graph based on the provided parameters.
        """
        fig, data_for_plot = fc.figure_update_subplot(df, df_col_string, current_fig, data_for_plot, x_column, y_column, z_column, t_column,
                                                      yfunc_column, zfunc_column, tfunc_column, graph_type, dim_type, 
                                                      smt_dropdown_value, smt_order_value, sub_bot_smt_value,
                                                      index_subplot, nb_subplots, nb_subplots_row, nb_subplots_col, large_file_memory)
        return fig, data_for_plot
    
    # =============================================================================
    # End Utility Function for Graphs
    # =============================================================================