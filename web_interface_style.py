#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 18:05:42 2024

@author: quentin
"""

"""#=============================================================================
   #=============================================================================
   #=============================================================================

    Dictionnary of functions for Dash app creation.

#=============================================================================
   #=============================================================================
   #============================================================================="""


import dash
from dash import dcc, html, Input, Output, dash_table, callback, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import webbrowser


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def web_interface_style():

    """
    Goal: Create the Dash app and define the style and theme.

    Parameters:
    - None

    Returns:
    - app: The Dash app.
    - dark_dropdown_style: Color style of the dropdown.
    - uniform_style: Color style of the dropdown.    
    """        

    # Initialize the Dash app with the dark theme
    app = dash.Dash(__name__, suppress_callback_exceptions=True)  # Use the DARKLY theme from Bootstrap
        
    # Define dark theme styles
    dark_dropdown_style = {
        'backgroundColor': '#1E1E78',  # Dark background for dropdown
        'color': '#f8f9fa',  # White text color
        'border': '1px solid #555',  # Border for dropdown
        'borderRadius': '5px',
        'width': '160px',
    }

    # Define a consistent style for both input and dropdown elements
    uniform_style = {
        'minWidth': '160px',  # Set a minimum width
        'minHeight': '40px',  # Set a minimum height
        'borderRadius': '5px',  # Rounded corners
    }


    #Creation of the app layout
    app.layout = html.Div([
        html.Div(
            className='header-container',
            style={'display': 'flex', 'alignItems': 'center', 'padding': '20px'},  # Flexbox layout for header
            children=[
                # IMDB Logo
                html.Img(src='assets/maxresdefault.jpg', style={'height': '140px', 'borderRadius': '14px 14px 14px 14px'}),  # Replace with your logo's path and adjust the size
                # Container for Tabs
                html.Div(
                    className='tabs-container',
                    children=[
                        # Tabs Component
                        dcc.Tabs(id='tabs', value='tab-1', style={'height': 'auto'}, children=[
                            dcc.Tab(id='tabs-1', label='🏠 Home', value='tab-1', 
                                     style={
                                         'backgroundColor': '#DAA520',  # Dark black background
                                         'color': 'black',
                                         'border': 'none',
                                         'borderRadius': '10px 10px 10px 10px',  
                                         'width': '200px',                    # Width relative to container
                                         'height': '100px',              # Adjusted height
                                         'display': 'flex',
                                         'alignItems': 'center',            # Center vertically
                                         'justifyContent': 'center',        # Center horizontally
                                         'margin-right': '10px',
                                         'font-size': '20px',   #Text size */
                                         'font-weight': 'bold',   #Text size */
                                        },
                                     selected_style={
                                         'backgroundColor': '#228B22',  # Slightly lighter for selected tab
                                         'color': 'black',
                                         'border': 'none',
                                         'borderRadius': '10px 10px 10px 10px',  
                                         'width': '200px',                    # Width relative to container
                                         'height': '100px',                  # Adjusted height
                                         'display': 'flex',
                                         'alignItems': 'center',            # Center vertically
                                         'justifyContent': 'center',        # Center horizontally
                                         'margin-right': '10px',
                                         'font-size': '20px',   #Text size */
                                         'font-weight': 'bold',   #Text size */
                                         # 'borderBottom': '2px solid white',
                                         # 'borderRight': '2px solid white',
                                        }
                                     ),
                            dcc.Tab(id='tabs-2', label='📈 Analytics', value='tab-2', 
                                     style={
                                         'backgroundColor': '#DAA520',
                                         'color': 'black',
                                         'border': 'none',
                                         'borderRadius': '10px 10px 10px 10px',  
                                         'width': '200px',                    # Width relative to container
                                         'height': '100px',                  # Adjusted height
                                         'display': 'flex',
                                         'alignItems': 'center',            # Center vertically
                                         'justifyContent': 'center',        # Center horizontally
                                         'margin-right': '10px',
                                         'font-size': '20px',   #Text size */
                                         'font-weight': 'bold',   #Text size */
                                        },
                                     selected_style={
                                         'backgroundColor': '#228B22',
                                         'color': 'black',
                                         'border': 'none',
                                         'borderRadius': '10px 10px 10px 10px',  
                                         'width': '200px',                    # Width relative to container
                                         'height': '100px',                  # Adjusted height
                                         'display': 'flex',
                                         'alignItems': 'center',            # Center vertically
                                         'justifyContent': 'center',        # Center horizontally
                                         'margin-right': '10px',
                                         'font-size': '20px',   #Text size */
                                         'font-weight': 'bold',   #Text size */
                                         # 'borderBottom': '2px solid white',
                                         # 'borderRight': '2px solid white',
                                     }
                                     ),
                            dcc.Tab(id='tabs-3', label='🔍 Data table', value='tab-3', 
                                     style={
                                         'backgroundColor': '#DAA520',
                                         'color': 'black',
                                         'border': 'none',
                                         'borderRadius': '10px 10px 10px 10px',  
                                         'width': '200px',                    # Width relative to container
                                         'height': '100px',                  # Adjusted height'
                                         'display': 'flex',
                                         'alignItems': 'center',            # Center vertically
                                         'justifyContent': 'center',        # Center horizontally
                                         'font-size': '20px',   #Text size */
                                         'font-weight': 'bold',   #Text size */
                                        },
                                     selected_style={
                                         'backgroundColor': '#228B22',
                                         'color': 'black',
                                         'border': 'none',
                                         'borderRadius': '10px 10px 10px 10px',  
                                         'width': '200px',                    # Width relative to container
                                         'height': '100px',                  # Adjusted height
                                         'display': 'flex',
                                         'alignItems': 'center',            # Center vertically
                                         'justifyContent': 'center',        # Center horizontally
                                         'font-size': '20px',   #Text size */
                                         'font-weight': 'bold',   #Text size */
                                         # 'borderBottom': '2px solid white',
                                         # 'borderRight': '2px solid white',
                                     }
                                     ),
                            ])
                ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'flex-start', 'height': '50px', 'margin': '50px'}  # Adjust according to height
        
               ),
            ]
        ),
        
        
        # Hidden store to hold df2 data
        dcc.Store(id='stored-df2', data=None),
    
        # Content Div for Tabs
        html.Div(id='tabs-content'),
        
    ])

    return app, dark_dropdown_style, uniform_style
    