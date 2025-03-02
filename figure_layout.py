#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 15:56:27 2024

@author: quentin
"""

"""#=============================================================================
   #=============================================================================
   #=============================================================================

    Dictionnary of functions to update graphic layout.

#=============================================================================
   #=============================================================================
   #============================================================================="""


import dash
from dash import dcc, html, Input, Output, dash_table, callback, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.io as pio
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import linear_model as lm, tree, neighbors
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from scipy import signal

from termcolor import colored

import matplotlib.pyplot as plt
import plotly.tools as tls  # For converting Matplotlib to Plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import Function_dataframe as fd
import Function_errors as fe
import data_plot_preparation as dpp
import figure_layout as fl


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def fig_update_layout(fig_json_serializable, data_for_plot,figname,xlabel,ylabel,zlabel,x_column,y_column,z_column,g_column,d_column, df_col_string):

    """
    Goal: Update the layout of the dash figure.

    Parameters:
    - fig_json_serializable: Dash figure.
    - figname: The name of the Figure.
    - xlabel: The xlabel of the axis (can be None).
    - ylabel: The ylabel of the axis (can be None).
    - zlabel: The zlabel of the axis (can be None).
    - x_column: Column in the dataframe
    - g_column: Type of Graphyque for the figure.
    - d_column: Graphyque dimension for the figure.
    - df_col_string: List of columns in the DataFrame that are of object type.

    Returns:
    - fig_json_serializable: Dash figure updated.
    """
    
    
    modified_xlabel = xlabel.replace(' ', '_') if xlabel is not None else None
    modified_ylabel = ylabel.replace(' ', '_') if ylabel is not None else None
    modified_zlabel = zlabel.replace(' ', '_') if zlabel is not None else None
    modified_glabel = g_column.replace(' ', '_') if g_column is not None else None
    
    print("figpath = ", 'x_'+str(modified_xlabel)+'_y_'+str(modified_ylabel)+'_z_'+str(modified_zlabel)+'_g_'+str(modified_glabel)+'_d_'+str(d_column))
    
    df_col_string = [col + '_split' for col in df_col_string]

    fig_json_serializable.update_layout(
        plot_bgcolor='#1e1e1e',  # Darker background for the plot area
        paper_bgcolor='#101820',  # Dark gray for the paper
        font=dict(color='white'),  # White text color
        # title = figname,
        # title_font=dict(size=20, color='white')
        )

    if x_column is not None and (d_column =="1D"or d_column =="2D") and g_column != 'Pie':
        fig_json_serializable.update_layout(
            plot_bgcolor='#1e1e1e',  # Darker background for the plot area
            paper_bgcolor='#101820',  # Dark gray for the paper
            font=dict(color='white'),  # White text color
            # title = figname,
            # title_font=dict(size=20, color='white'),  # Title styling
            xaxis=dict(
                # range=[0, 2000] if g_column == 'Histogram Movie' else None,
                title=dict(text=xlabel, font=dict(size=20, color='white')),  # X-axis label styling
                tickfont=dict(color='white', size=18),  # X-axis tick color
                tickangle=0,  # Rotate the x-axis labels for better readability
                showgrid=True,  # Grid styling
                gridcolor='gray',  # Grid color
                categoryorder='category ascending',  # Ensures categorical x-values are treated correctly
            ),
            yaxis=dict(
                title=dict(text=ylabel, font=dict(size=20, color='white')),  # Y-axis label styling
                tickfont=dict(color='white', size=18),  # Y-axis tick color
                tickangle=0,  # Rotate the x-axis labels for better readability
                showgrid=True,  # Grid styling
                gridcolor='gray',  # Grid color
                categoryorder='total ascending' if x_column in df_col_string else 'category ascending',  # Ensures categorical x-values are treated correctly
                
            )
            
        )
        # if y_column is not None:
        #     fig_json_serializable.update_layout(
        #         updatemenus=[
        #             dict(
        #                 buttons=list([
        #                     dict(
        #                         args=[{"marker.colorscale": "Viridis", "coloraxis.colorbar.title": y_column}],
        #                         label="Linear Scale",
        #                         method="restyle"
        #                     ),
        #                     dict(
        #                         args=[{"marker.colorscale": "Viridis", "marker.colors": data_for_plot[y_column].apply(lambda x: max(x, 1e-10)), "coloraxis.colorbar.title": y_column}],
        #                         label="Log Scale",
        #                         method="restyle"
        #                     )
        #                 ]),
        #                 direction="down",
        #                 pad={"r": 10, "t": 10},
        #                 showactive=True,
        #                 x=0.1,           # position of the dropdown
        #                 xanchor="left",
        #                 y=1.1,           # position of the dropdown
        #                 yanchor="top"
        #             ),
        #         ],
        #         coloraxis_colorbar=dict(title=y_column)  # Add the color bar title
        #     )
        
    elif x_column is not None and d_column =="3D":
        fig_json_serializable.update_layout(
            plot_bgcolor='#1e1e1e',  # Darker background for the plot area
            paper_bgcolor='#101820',  # Dark gray for the paper
            font=dict(color='white'),  # White text color
            # title = figname,
            # title_font=dict(size=20, color='white'),  # Title styling
            scene=dict(
                    xaxis=dict(
                        title=dict(text=xlabel, font=dict(size=18, color='white')),  # X-axis label styling
                        tickmode='array',
                        tickfont=dict(color='white', size=14),  # X-axis tick color
                        tickangle=0,  # Rotate the x-axis labels for better readability
                        showgrid=True,  # Grid styling
                        gridcolor='gray',  # Grid color
                        categoryorder='category ascending',  # Ensures categorical x-values are treated correctly
                    ),
                    yaxis=dict(
                        title=dict(text=ylabel, font=dict(size=18, color='white')),  # Y-axis label styling
                        tickmode='array',
                        tickfont=dict(color='white', size=14),  # Y-axis tick color
                        tickangle=0,  # Rotate the x-axis labels for better readability
                        showgrid=True,  # Grid styling
                        gridcolor='gray',  # Grid color
                        categoryorder='category ascending',  # Ensures categorical x-values are treated correctly
                    ),
                    zaxis=dict(
                        title=dict(text='Count', font=dict(size=18, color='white')),
                        tickmode='array',
                        tickfont=dict(color='white', size=14)  # Z-axis tick color
                    )
            )
        )
        
    if g_column == 'Colormesh':    

        # Update 3D scene options
        fig_json_serializable.update_scenes(
            aspectratio=dict(x=1, y=1, z=0.7),
            aspectmode="manual"
        )
        
        # Add dropdowns
        button_layer_1_height = 1.08
        
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=["colorscale", "Viridis"],
                        label="Viridis",
                        method="restyle"
                    ),
                    dict(
                        args=["colorscale", "Cividis"],
                        label="Cividis",
                        method="restyle"
                    ),
                    dict(
                        args=["colorscale", "Blues"],
                        label="Blues",
                        method="restyle"
                    ),
                    dict(
                        args=["colorscale", "Greens"],
                        label="Greens",
                        method="restyle"
                    ),
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=button_layer_1_height,
                yanchor="top"
            ),
            dict(
                buttons=list([
                    dict(
                        args=["reversescale", False],
                        label="False",
                        method="restyle"
                    ),
                    dict(
                        args=["reversescale", True],
                        label="True",
                        method="restyle"
                    )
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.37,
                xanchor="left",
                y=button_layer_1_height,
                yanchor="top"
            ),
            dict(
                buttons=list([
                    dict(
                        args=[{"contours.showlines": False, "type": "contour"}],
                        label="Hide lines",
                        method="restyle"
                    ),
                    dict(
                        args=[{"contours.showlines": True, "type": "contour"}],
                        label="Show lines",
                        method="restyle"
                    ),
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.58,
                xanchor="left",
                y=button_layer_1_height,
                yanchor="top"
            ),
        ]
    
        fig_json_serializable.update_layout(
        updatemenus=updatemenus
        )
        
        
    if g_column == 'Histogram Movie':
        fig_json_serializable.update_layout(
        margin=dict(l=150, r=20, t=20, b=20)
        )
        


    if d_column == "3D":
        name = 'default'
        # Default parameters which are used when `layout.scene.camera` is not provided
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=1.25, z=1.25)
        )
        
        fig_json_serializable.update_layout(scene_camera=camera) #, title=name