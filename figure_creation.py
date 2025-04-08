#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 23:18:23 2024

@author: quentin
"""


"""#=============================================================================
   #=============================================================================
   #=============================================================================

    Dictionnary of functions for visualisation of the dataframe.

#=============================================================================
   #=============================================================================
   #============================================================================="""


import dash
from dash import dcc, html, Input, Output, dash_table, callback, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.io as pio
import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn import linear_model as lm, tree, neighbors
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from scipy import signal
from scipy.interpolate import griddata

from termcolor import colored

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.tools as tls  # For converting Matplotlib to Plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


import Exploratory_Data_Analysis.function_dataframe as fd
import Exploratory_Data_Analysis.data_plot_preparation as dpp
import Exploratory_Data_Analysis.figure_layout as fl
import Exploratory_Data_Analysis.machine_learning_functions as mlf
import Exploratory_Data_Analysis.debug_dash_infos as ddi
import Exploratory_Data_Analysis.app_state as aps

cmaps = [('Perceptually Uniform Sequential', [
            'viridis', 'plasma', 'inferno', 'magma']),
         ('Sequential', [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
         ('Sequential (2)', [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']),
         ('Diverging', [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
         ('Qualitative', [
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c']),
         ('Miscellaneous', [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])]


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def create_figure(df, df_col_string, x_column, y_column, z_column, t_column, yf_column, zf_column, tf_column, g_column, d_column, smt_dropdown_value, smt_order_value, sub_bot_smt_value, Large_file_memory):

    """
    Goal: Create a sophisticated figure which adapt to any input variable.

    Parameters:
    - df: dataframe
    - x_column: Column in the dataframe
    - y_column: Column in the dataframe (default count)
    - z_column: Column in the dataframe (can be None)
    - t_column: Column in the dataframe (can be None)
    - yf_column: Function to operate on y_column with the rest of the dataframe
    - zf_column: Function to operate on z_column with the rest of the dataframe
    - tf_column: Function to operate on t_column with the rest of the dataframe
    - g_column: Type of Graphyque for the figure.
    - d_column: Graphyque dimension for the figure.
    - sub_bot_smt_value: Button to apply the smoothing.
    - smt_dropdown_value: Type of smoothing for the data.
    - smt_dropdown_value: Order of the smoothing for the data.
    - Large_file_memory: Estimate if the file is too large to be open with panda

    Returns:
    - fig_json_serializable: The finalized plotly figure. 
    """

    Debug = aps.Debug

    # =============================================================================
    ddi.debug_print(colored("========================= Start figure creation =========================", "green"), debug=Debug)
    # =============================================================================      
    # Create a Dash compatible Plotly graph figure
    fig_json_serializable = go.Figure()  # This figure can now be used with dcc.Graph in Dash
    
    # Create the label of the figure
    figname, xlabel, ylabel, zlabel, tlabel = label_fig(x_column, y_column, z_column, t_column, yf_column, zf_column, g_column, d_column, True, df_col_string)  
    ddi.debug_print("label_fig done", debug=Debug)
    data_for_plot = []
    if x_column is not None: 
        ddi.debug_print("Extract from data base the required column and prepare them for the figure.", debug=Debug)
        Para, data_for_plot, x_column, y_column, z_column, t_column = dpp.data_preparation_for_plot(df, df_col_string , x_column, y_column, z_column, t_column, yf_column, zf_column, tf_column, g_column, Large_file_memory)
        ddi.debug_print("The data ready to be ploted is:", debug=Debug)
        ddi.debug_print(data_for_plot, debug=Debug)
        ddi.debug_print("", debug=Debug)
        # Add the core of the figure
        ddi.debug_print("############## Core figure creation ##############", debug=Debug)
        fig_json_serializable, data_for_plot, xlabel, ylabel, zlabel, tlabel = figure_plotly(fig_json_serializable, x_column, y_column, z_column, t_column, yf_column, zf_column, tf_column, g_column, d_column, smt_dropdown_value, smt_order_value, sub_bot_smt_value, data_for_plot, xlabel, ylabel, zlabel, tlabel, df_col_string)
    
    # Update the figure layout
    ddi.debug_print("############## Update figure layout ##############", debug=Debug)
    fl.fig_update_layout(fig_json_serializable, data_for_plot,figname,xlabel,ylabel,zlabel,tlabel,x_column,y_column,z_column,t_column,g_column,d_column,df_col_string)       
    plt.close()
    # =============================================================================
    ddi.debug_print(colored("=============================================================================", "green"), debug=Debug)
    if x_column is None: 
        return fig_json_serializable, None
    
    return fig_json_serializable, data_for_plot.to_dict(orient='records')


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def label_fig(x_column, y_column, z_column, t_column, yf_column, zf_column, g_column, d_column, init, df_col_string):

    """
    Goal: Create the figure labels.

    Parameters:
    - x_column: Column in the dataframe (can be None).
    - y_column: Column in the dataframe (can be None).
    - z_column: Column in the dataframe (can be None).
    - t_column: Column in the dataframe (can be None).
    - yf_column: Function to operate on y_column with the rest of the dataframe
    - zf_column: Function to operate on z_column with the rest of the dataframe
    - g_column: Type of Graphyque for the figure.
    - d_column: Graphyque dimension for the figure.

    Returns:
    - figname: The name of the Figure.
    - xlabel: The xlabel of the axis (can be None).
    - ylabel: The ylabel of the axis (can be None).
    - zlabel: The zlabel of the axis (can be None).
    """
        
    
    df_col_string = [col[:-6] if col.endswith('_split') else col for col in df_col_string]

    if init == False:
        if x_column is not None: 
            figname = "x"+str(x_column)+"y"+str(y_column)+"z"+str(z_column)+"t"+str(t_column)
    
            if 'avg_' in x_column:
                xlabel = 'Average '+x_column[4:]#+' of the movies'
            else:
                xlabel = x_column
            
            if 'avg_' in y_column:
                ylabel = 'Average '+y_column[4:]#+' of the movies'              
            else:
                ylabel = y_column
            
            if z_column is not None:
                if 'avg_' in z_column:
                    zlabel = 'Average '+z_column[4:]#+' of the movies'
                else:
                    zlabel = z_column
            else:
                zlabel = "None"

            if t_column is not None:
                if 'avg_' in t_column:
                    tlabel = 'Average '+t_column[4:]#+' of the movies'
                else:
                    tlabel = t_column
            else:
                tlabel = "None"
            
            
            if d_column == "2D":
                if g_column == 'Colormesh':
                    ylabel = y_column
                else:
                    ylabel = "None"
                zlabel = "None"
    
    
        else: 
            figname = 'No data selected'
            xlabel, ylabel, zlabel, tlabel = "None","None","None","None"
    
    else:
        figname = 'No data selected'
        xlabel, ylabel, zlabel, tlabel = "None","None","None","None"  
    
    if x_column in df_col_string:
        xlabel_temp = xlabel
        xlabel = ylabel
        ylabel = xlabel_temp
    
    return figname, xlabel, ylabel, zlabel, tlabel


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def figure_plotly(plotly_fig, x_column, y_column, z_column, t_column, yf_column, zf_column, tf_column, g_column, d_column, smt_dropdown_value, smt_order_value, sub_bot_smt_value, data_for_plot, xlabel, ylabel, zlabel, tlabel, df_col_string):

    """
    Goal: Create the plot inside the figure regarding the inputs.

    Parameters:
    - plotly_fig: Dash figure.
    - x_column: Column in the dataframe
    - y_column: Column in the dataframe (can be None)
    - z_column: Column in the dataframe (can be None)
    - t_column: Column in the dataframe (can be None)
    - yf_column: Function to operate on y_column with the rest of the dataframe
    - zf_column: Function to operate on z_column with the rest of the dataframe
    - tf_column: Function to operate on t_column with the rest of the dataframe
    - g_column: Type of Graphyque for the figure.
    - d_column: Graphyque dimension for the figure.
    - sub_bot_smt_value: Button to apply the smoothing.
    - smt_dropdown_value: Type of smoothing for the data.
    - smt_dropdown_value: Order of the smoothing for the data.
    - data_for_plot: Data to plot.
    - xlabel: The xlabel of the axis (can be None).
    - ylabel: The ylabel of the axis (can be None).
    - zlabel: The zlabel of the axis (can be None).
    - tlabel: The tlabel of the axis (can be None).
    - df_col_string: List of columns in the DataFrame that are of object type.

    Returns:
    - plotly_fig: The core figure.
    """
    
    Debug = aps.Debug
    
    df_col_string = [col + '_split' for col in df_col_string]

    # Define a list of colors for the bars
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    legend = "None"
    
    x_axis = x_column
    y_axis = y_column
    z_axis = z_column
    t_axis = t_column

    if yf_column == "Avg":
        y_axis = 'avg_' + y_column
    if zf_column == "Avg":
        z_axis = 'avg_' + z_column
    if tf_column == "Avg":
        t_axis = 'avg_' + t_column

    ddi.debug_print(f"x_axis = {x_axis}", debug=Debug)
    ddi.debug_print(f"y_axis = {y_axis}", debug=Debug)
    if str(y_column)!='None':
        ddi.debug_print(f"z_axis = {z_axis}", debug=Debug)
    if str(z_column)!='None':
        ddi.debug_print(f"t_axis = {t_axis}", debug=Debug)


    # Rename the label of the figure
    figname, xlabel, ylabel, zlabel, tlabel = label_fig(x_axis, y_axis, z_axis, t_axis, yf_column, zf_column, g_column, d_column, False, df_col_string)  
    
    ddi.debug_print("label_fig done", debug=Debug)
    
    if d_column=="1D": 
                
        ddi.debug_print("", debug=Debug)
        if str(z_column) == 'None':
            
            data_for_plot = smoothing_data(sub_bot_smt_value, smt_dropdown_value, smt_order_value, data_for_plot, x_axis, y_axis, z_axis, df_col_string)
            
            if g_column=="Histogram":
                plotly_fig = px.bar(
                    data_for_plot, 
                    x=x_axis, 
                    y=y_axis
                    )
            if g_column=="Curve":
                plotly_fig = px.line(
                    data_for_plot, 
                    x=x_axis, 
                    y=y_axis
                    ) #, color=y_column, symbol="country"
            if g_column=="Scatter":
                plotly_fig = px.scatter(
                    data_for_plot,
                    x=x_axis,
                    y=y_axis,
                    # log_x=True,
                    size_max=60
                    )
            if "Boxes" in g_column:
                plotly_fig = px.box(
                    data_for_plot, 
                    x=x_axis, 
                    y=y_axis,
                    points=False)
                
        #Case where y_column is None and z_column is None
        elif str(z_column)!='None' and str(t_column) == 'None':           

            # if x_column in df_col_string and "Movie" not in g_column:
            #     # Grouping y_column values
            #     n = 10  # Number of top categories to keep
            #     data_for_plot = fd.group_small_values(data_for_plot, y_axis, x_axis, n)

            # if y_column in df_col_string and "Movie" not in g_column:
            #     # Grouping y_column values
            #     n = 7  # Number of top categories to keep
            #     data_for_plot = fd.group_small_values(data_for_plot, z_axis, y_axis, n, x_axis)


            data_for_plot = smoothing_data(sub_bot_smt_value, smt_dropdown_value, smt_order_value, data_for_plot, x_axis, y_axis, z_axis, df_col_string)
            
            if "Histogram" in g_column:
                plotly_fig = px.bar(
                   data_for_plot, 
                   x=x_axis, 
                   y=y_axis,
                   color=z_axis if "Movie" not in g_column else None,
                   animation_frame=z_axis if "Movie" in g_column else None,
                   range_x=[data_for_plot[x_axis].min(), data_for_plot[x_axis].max()] if "Movie" in g_column else None,
                   range_y=[data_for_plot[y_axis].min(), data_for_plot[y_axis].max()] if "Movie" in g_column else None
                   )
            if "Curve" in g_column:
                plotly_fig = px.line(
                    data_for_plot, 
                    x=x_axis, 
                    y=y_axis,
                    color=z_axis if "Movie" not in g_column else None,
                    animation_frame=z_axis if "Movie" in g_column else None,
                    line_group=g_column if "Movie" in g_column else None
                    ) #symbol="country"
            if "Scatter" in g_column:
                plotly_fig = px.scatter(
                    data_for_plot,
                    x=x_axis,
                    y=y_axis,
                    size_max=60,
                    # log_x=True,
                    color=z_axis if "Movie" not in g_column else None,
                    animation_frame=z_axis if "Movie" in g_column else None
                    )
            if "Boxes" in g_column:
                plotly_fig = px.box(
                    data_for_plot, 
                    x=x_axis, 
                    y=y_axis,
                    facet_row=z_axis,
                    points=False)
                    

        #Case where z_column is not None
        elif str(t_column)!='None':
                        
            # if y_column in df_col_string:
            #     # Grouping y_column values
            #     n = 7  # Number of top categories to keep
            #     if zf_column == "Avg":
            #         data_for_plot = fd.group_small_values(data_for_plot, z_axis, y_axis, n, x_axis)
            #     elif zf_column == "Avg on the ordinate":
            #         data_for_plot = fd.group_small_values(data_for_plot, z_axis, t_axis, n, x_axis)
            #     elif zf_column == "Weight on y":
            #         data_for_plot = fd.group_small_values(data_for_plot, z_axis, y_axis, n, x_axis)

            data_for_plot = smoothing_data(sub_bot_smt_value, smt_dropdown_value, smt_order_value, data_for_plot, x_axis, y_axis, z_axis, df_col_string)
                        
            if "Histogram" in g_column:
                plotly_fig = px.bar(
                   data_for_plot, 
                   x=x_axis, 
                   y=y_axis,
                   color=z_axis,
                   animation_frame=t_axis,
                   range_x=[data_for_plot[x_axis].min(), data_for_plot[x_axis].max()] if "Movie" in g_column else None,
                   range_y=[data_for_plot[y_axis].min(), data_for_plot[y_axis].max()] if "Movie" in g_column else None
                   )
            elif "Curve" in g_column:
                if "Movie" not in g_column and t_axis+"_split" not in df_col_string:
                    plotly_fig = go.Figure()
                    # Add traces for each unique group
                    for key in data_for_plot[z_axis].unique():
                        group = data_for_plot[data_for_plot[z_axis] == key]
                        plotly_fig.add_trace(go.Scatter(
                            x=group[x_axis],
                            y=group[y_axis],
                            mode='lines',
                            name=key,
                            line=dict(width=group[t_axis].mean())  # Set line width based on avg thickness
                        ))
                else:
                    plotly_fig = px.line(
                        data_for_plot, 
                        x=x_axis, 
                        y=y_axis,
                        color=z_axis,
                        animation_frame=t_axis,
                        range_x=[data_for_plot[x_axis].min(), data_for_plot[x_axis].max()] if "Movie" in g_column else None,
                        range_y=[data_for_plot[y_axis].min(), data_for_plot[y_axis].max()] if "Movie" in g_column else None
                        )
            elif "Scatter" in g_column:
                plotly_fig = px.scatter(
                   data_for_plot, 
                   x=x_axis, 
                   y=y_axis,
                   color=z_axis,
                   size_max=60,
                   size=t_axis if ("Movie" not in g_column and t_axis+"_split" not in df_col_string) else None,
                   animation_frame=t_axis if ("Movie" in g_column or t_axis+"_split" in df_col_string) else None,
                   range_x=[data_for_plot[x_axis].min(), data_for_plot[x_axis].max()] if ("Movie" in g_column or t_axis+"_split" in df_col_string) else None,
                   range_y=[data_for_plot[y_axis].min(), data_for_plot[y_axis].max()] if ("Movie" in g_column or t_axis+"_split" in df_col_string) else None
                   )
                

    if g_column=="Pie": #d_column=="2D" and 

        if x_column in df_col_string:
            # Grouping y_column values
            n = 24  # Number of top categories to keep
            data_for_plot = fd.group_small_values(data_for_plot, x_column, 'count', n)

        # x_values,fig_x_value,y_values,fig_y_value=None,None,None,None
        if str(y_column)=='count':
            plotly_fig = px.pie(
                data_for_plot, 
                values="count", 
                names=x_column
                )
        elif str(y_column)!='count':
            plotly_fig = px.sunburst(
                data_for_plot,
                path=[x_axis, y_axis],  # Define the hierarchy
                values=y_axis  # Use a column for values
            )
        elif str(z_column)!='None':
            plotly_fig = px.sunburst(
                data_for_plot,
                path=[x_axis, y_axis, z_axis],  # Define the hierarchy
                values=z_axis  # Use a column for values
            )
        elif str(t_column)!='None':
            plotly_fig = px.sunburst(
                data_for_plot,
                path=[x_axis, y_axis, z_axis, t_axis],  # Define the hierarchy
                values=t_axis  # Use a column for values
            )


    if d_column=="2D" and g_column=="Colormesh":     
        
        px_fig = px.density_heatmap(
            data_for_plot, 
            x=x_axis, 
            y=y_axis, 
            # nbinsx=100, nbinsy=100, 
            z=z_axis,
            color_continuous_scale="Viridis")

        # Get the z data from the px figure
        z_data = px_fig.data[0].z  # Access the z values (the counts)
        x_values = px_fig.data[0].x  # Access the x values (start years)
        y_values = px_fig.data[0].y  # Access the y values (runtime minutes)
        
        # Create a Go Figure and add the Heatmap trace
        plotly_fig = go.Figure()
        
        # Add Heatmap trace
        plotly_fig.add_trace(go.Heatmap(
            x=x_values,
            y=y_values,
            z=z_data,
            colorscale='Viridis'
        ))

                    
    if d_column == "3D":
        
        # if g_column == "Histogram":
        
        #     # Pivoting the DataFrame to create a grid for surface plot
        #     pivoted_data = data_for_plot.pivot(index=y_column, columns=x_column, values='count')
        #     # Fill NaN values with zeros or an appropriate value for the surface
        #     pivoted_data = pivoted_data.fillna(0)
        #     # Now, create the surface plot
    
        #     plotly_fig = go.Figure(
        #         data=[go.Surface(z=pivoted_data.values, x=pivoted_data.columns, y=pivoted_data.index)])



        if "Scatter" in g_column:
                        
            print(x_axis,y_axis,z_axis,t_axis)
            plotly_fig = px.scatter_3d(
                data_frame=data_for_plot,
                x=x_axis,
                y=y_axis,
                z=z_axis,
                color=t_axis if t_axis is not None else None)



    return plotly_fig, data_for_plot, xlabel, ylabel, zlabel, tlabel  


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def smoothing_data(sub_bot_smt_value, smt_dropdown_value, smt_order_value, data_for_plot, x_axis, y_axis, z_axis, df_col_string):

    """
    Goal: Apply a filter on the data.

    Parameters:
    - sub_bot_smt_value: Button to apply the smoothing.
    - smt_dropdown_value: Type of smoothing for the data.
    - smt_dropdown_value: Order of the smoothing for the data.
    - data_for_plot: Dataframe which will be filtered.
    - x_axis: Column in the dataframe.
    - y_axis: Column in the dataframe.
    - z_axis: Column in the dataframe.
    - df_col_string: List of columns in the DataFrame that are of object type.

    Returns:
    - data_for_plot: Dataframe updated.
    """
    
    Debug = aps.Debug
    
    if sub_bot_smt_value % 2 == 1:
        
        ddi.debug_print("############## Smoothing #################", debug=Debug)
        
        data_for_plot['original_index'] = data_for_plot.index
        
        if z_axis is None or z_axis not in df_col_string:
            window_length = len(data_for_plot[x_axis])//5
            data_for_plot[y_axis] = signal.savgol_filter(data_for_plot[y_axis],
                                   window_length, # window size used for filtering
                                   smt_order_value)
            ddi.debug_print(f"window_length = {window_length}", debug=Debug)
            ddi.debug_print("Data updated by the smoothing", debug=Debug)
            ddi.debug_print(data_for_plot, debug=Debug)

        else:
            # Function to apply savgol_filter
            def apply_savgol_filter(group):
                # Calculate window length based on the size of the group
                window_length = len(group)//5
                
                # Ensure that window_length is odd and less than or equal to the total group length
                if window_length < 3:  # Savitzky-Golay filter needs at least a size of 3
                    return group  # Skip filtering for groups too small
                
                if window_length % 2 == 0:
                    window_length -= 1  # Make sure window_length is odd
                
                ddi.debug_print(f"window_length = {window_length}", debug=Debug)
                ddi.debug_print(("Amount of data", len(group[y_axis])), debug=Debug)
                ddi.debug_print("", debug=Debug)
                
                # Apply the savgol_filter
                filtered_values = signal.savgol_filter(group[y_axis], window_length, smt_order_value)
    
                # Replace the original 'count' with the filtered values
                group[y_axis] = filtered_values
                                
                return group
            
            # Apply the filter to each genre
            data_for_plot_filtered = data_for_plot.groupby(z_axis, as_index=False, group_keys=False).apply(apply_savgol_filter)
                        
            # Sort the DataFrame by the original index
            data_for_plot_filtered.sort_values(by='original_index', inplace=True)
                        
            # Drop the 'original_index' column if you no longer need it
            data_for_plot_filtered.drop(columns='original_index', inplace=True)
                
            data_for_plot = data_for_plot_filtered
        
            
            ddi.debug_print("Data updated by the smoothing", debug=Debug)
            ddi.debug_print(data_for_plot, debug=Debug)
            
    return data_for_plot
            

"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def figure_add_trace(fig_json_serializable, data_for_plot, df_col_string, x_column, y_column, z_column, t_column, yf_column, zf_column, tf_column, graph_type, dim_type, 
                     type_model, ml_tar, ml_tar_type, ml_size, 
                     ml_num_fea, ml_num_imp, ml_num_enc,
                     ml_ode_fea, ml_ode_imp, ml_ode_enc,
                     ml_ohe_fea, ml_ohe_imp, ml_ohe_enc,
                     ml_model):
    
    
    """
    Goal: Add a trace inside the figure regarding the inputs.

    Parameters:
    - fig_json_serializable: Dash figure.
    - data_for_plot: Dataframe which has been use to create the figure that is re-opened in this function.
    - df_col_string: List of columns in the DataFrame that are of object type.
    - x_column: Column in the dataframe
    - y_column: Column in the dataframe (can be None)
    - z_column: Column in the dataframe (can be None)
    - t_column: Column in the dataframe (can be None)
    - yf_column: Function to operate on y_column with the rest of the dataframe
    - zf_column: Function to operate on z_column with the rest of the dataframe
    - tf_column: Function to operate on t_column with the rest of the dataframe
    - graph_type: Type of Graphyque for the figure.
    - dim_type: Graphyque dimension for the figure.

    - type_model: Type of the machine learning problem (Regression/Classification).
    - ml_tar: The target value.
    - ml_tar_type: Nature of the target variable ("numerical", "ordinal", "nominal").
    - ml_size: The ratio of testing value for the fit.

    - ml_num_fea: Value for numerical features.
    - ml_ode_fea: Value for nominal features.
    - ml_ohe_fea: Value for ordinal features.
    
    - ml_num_imp: Imputer for numerical features.
    - ml_ode_imp: Imputer for nominal features.
    - ml_ohe_imp: Imputer for ordinal features.

    - ml_num_enc: Encoder for numerical features.
    - ml_ode_enc: Encoder for nominal features.
    - ml_ohe_enc: Encoder for ordinal features.

    - ml_model: Model of classification for the data.


    Returns:
    - fig_json_serializable: Dash figure updated with the trace.
    - data_for_plot: Dataframe updated with the trace.
    """
    
    Debug = aps.Debug
    
    plotly_fig = go.Figure(fig_json_serializable)

    df_col_string = [col + '_split' for col in df_col_string]
    
    # Define a list of colors for the bars
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'] 
    
    x_axis = x_column
    y_axis = y_column
    z_axis = z_column
    t_axis = t_column

    if yf_column == "Avg":
        y_axis = 'avg_' + y_column
    if zf_column == "Avg":
        z_axis = 'avg_' + z_column
    if tf_column == "Avg":
        t_axis = 'avg_' + t_column     

    ddi.debug_print(("x_axis=", x_axis), debug=Debug)
    ddi.debug_print(("y_axis=", y_axis), debug=Debug)
    ddi.debug_print(("z_axis=", z_axis), debug=Debug)
    ddi.debug_print(("t_axis=", t_axis), debug=Debug)

    # Creating a DataFrame
    data_for_plot = pd.DataFrame(data_for_plot)

    # Resetting the index to have a clean index
    data_for_plot.reset_index(drop=True, inplace=True)

            
    # Make ML classification
    X, y, preprocessor, ml_model = mlf.make_model(type_model, data_for_plot,
                                  ml_tar, ml_tar_type, ml_size, 
                                  ml_num_fea, ml_num_imp, ml_num_enc,
                                  ml_ode_fea, ml_ode_imp, ml_ode_enc,
                                  ml_ohe_fea, ml_ohe_imp, ml_ohe_enc,
                                  ml_model)

        
    if type_model == "Regression":
        
        X_reduced = X.values

        if ml_tar == x_axis and z_axis is None:
            unique_y = X[y_axis].unique()
        elif ml_tar == y_axis  and z_axis is None:
            unique_x = X[x_axis].unique()
        elif ml_tar == x_axis  and z_axis is not None and t_axis is None:
            unique_y = X[y_axis].unique()
            unique_z = X[z_axis].unique()
        elif ml_tar == y_axis  and z_axis is not None and t_axis is None:
            unique_x = X[x_axis].unique()
            unique_z = X[z_axis].unique()
        elif ml_tar == z_axis and t_axis is None:
            unique_x = X[x_axis].unique()
            unique_y = X[y_axis].unique() 
        elif ml_tar == x_axis and t_axis is not None:
            unique_y = X[y_axis].unique()
            unique_z = X[z_axis].unique() 
            unique_t = X[t_axis].unique() 
        elif ml_tar == y_axis and t_axis is not None:
            unique_x = X[x_axis].unique()
            unique_z = X[z_axis].unique() 
            unique_t = X[t_axis].unique() 
        elif ml_tar == z_axis and t_axis is not None:
            unique_x = X[x_axis].unique()
            unique_y = X[y_axis].unique() 
            unique_t = X[t_axis].unique() 
        elif ml_tar == t_axis:
            unique_x = X[x_axis].unique()
            unique_y = X[y_axis].unique() 
            unique_z = X[z_axis].unique() 
        
        nb_linspace = 100
        # Handling for meshgrid based on types
        if ml_tar != x_axis:
            if np.issubdtype(X[x_axis].dtype, np.number):
                x_min, x_max = X[x_axis].min() - 1, X[x_axis].max() + 1
                xx = np.linspace(x_min, x_max, nb_linspace)
            else:
                xx = unique_x
        
        if ml_tar != y_axis:
            if np.issubdtype(X[y_axis].dtype, np.number):
                y_min, y_max = X[y_axis].min() - 1, X[y_axis].max() + 1
                yy = np.linspace(y_min, y_max, nb_linspace)
            else:
                yy = unique_y

        if ml_tar != z_axis and z_axis is not None:
            if np.issubdtype(X[z_axis].dtype, np.number):
                z_min, z_max = X[z_axis].min() - 1, X[z_axis].max() + 1
                zz = np.linspace(z_min, z_max, nb_linspace)
            else:
                zz = unique_z

        if ml_tar != t_axis and t_axis is not None:
            if np.issubdtype(X[t_axis].dtype, np.number):
                t_min, z_max = X[t_axis].min() - 1, X[t_axis].max() + 1
                tt = np.linspace(t_min, t_max, nb_linspace)
            else:
                tt = unique_t


        if len(X.columns.tolist()) == 1:
            if ml_tar != x_axis:
                grid = xx
            else:
                grid = yy
            
        elif len(X.columns.tolist()) == 2:
            
            if ml_tar == x_axis:
                grid_y, grid_z = np.meshgrid(yy, zz)
                # Predict the full grid
                grid = np.c_[grid_y.ravel(), grid_z.ravel()]
                
            if ml_tar == y_axis:
                grid_x, grid_z = np.meshgrid(xx, zz)
                # Predict the full grid
                grid = np.c_[grid_x.ravel(), grid_z.ravel()]     
                
            if ml_tar == z_axis:
                grid_x, grid_y = np.meshgrid(xx, yy)
                # Predict the full grid
                grid = np.c_[grid_x.ravel(), grid_y.ravel()]


        elif len(X.columns.tolist()) == 3:
            
            if ml_tar == x_axis:
                grid_y, grid_z, grid_t = np.meshgrid(yy, zz, tt)
                # Predict the full grid
                grid = np.c_[grid_y.ravel(), grid_z.ravel(), grid_t.ravel()]
                
            if ml_tar == y_axis:
                grid_x, grid_z, grid_t = np.meshgrid(xx, zz, tt)
                # Predict the full grid
                grid = np.c_[grid_x.ravel(), grid_z.ravel(), grid_t.ravel()]   
                
            if ml_tar == z_axis:
                grid_x, grid_y, grid_t = np.meshgrid(xx, yy, tt)
                # Predict the full grid
                grid = np.c_[grid_x.ravel(), grid_y.ravel(), grid_t.ravel()]

            if ml_tar == t_axis:
                grid_x, grid_y, grid_z = np.meshgrid(xx, yy, zz)
                # Predict the full grid
                grid = np.c_[grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]
        
        print(grid)
            
        # Convert the grid into a pandas DataFrame
        # grid_df = pd.DataFrame(grid, columns=[x_col, y_col])
        grid_df = pd.DataFrame(grid, columns=X.columns.tolist())
        
        Z = ml_model.predict(preprocessor.fit_transform(grid_df))
        
        df_with_model = grid_df.copy()  # Start with a copy of the grid_df
        df_with_model['Prediction'] = Z   # Add the predictions as a new column  
        
        
        print(df_with_model)
        
        if (ml_tar == x_axis or ml_tar == y_axis) and t_axis is None:
            line_trace = px.line(
                df_with_model, 
                x='Prediction' if ml_tar == x_axis else x_axis, 
                y='Prediction' if ml_tar == y_axis else y_axis, 
                color=z_axis if (z_axis is not None) else None
                )
            
            print(line_trace)
            for trace in line_trace.data:
                plotly_fig.add_trace(trace)
            
        elif ml_tar == z_axis and t_axis is None:
        
            Z_numeric = Z

            heatmap_trace = px.density_heatmap(df_with_model, 
                                              x=x_axis, 
                                              y=y_axis, 
                                              z='Prediction')

            for data in heatmap_trace.data:  # Loop through existing heatmap data
                plotly_fig.add_trace(data)  # Add each trace to the main figure


        elif (ml_tar == x_axis or ml_tar == y_axis or ml_tar == z_axis) and t_axis is not None:

            # Get unique values for the t_axis
            unique_t_values = df_with_model[t_axis].unique()
        
            print(df_with_model)
            print(unique_t_values)

            
            color_map = {t_value: color for t_value, color in zip(unique_t_values, colors)}


            for t_value in unique_t_values:
                print(t_value, t_axis)
                
                filtered_data_for_plot = data_for_plot[df_with_model[t_axis] == t_value]

                # Filter the DataFrame for the current t_axis value
                filtered_df = df_with_model[df_with_model[t_axis] == t_value]
                
                
                conditions = []
                if ml_tar != x_axis:
                    x_min = filtered_data_for_plot[x_axis].min()
                    x_max = filtered_data_for_plot[x_axis].max()                    
                    conditions.append((filtered_df[x_axis] < x_max) & (filtered_df[x_axis] > x_min))
                
                if ml_tar != y_axis:
                    y_min = filtered_data_for_plot[y_axis].min()
                    y_max = filtered_data_for_plot[y_axis].max()  
                    conditions.append((filtered_df[y_axis] < y_max) & (filtered_df[y_axis] > y_min))

                if ml_tar != z_axis:
                    z_min = filtered_data_for_plot[z_axis].min()
                    z_max = filtered_data_for_plot[z_axis].max()  
                    conditions.append((filtered_df[z_axis] < z_max) & (filtered_df[z_axis] > z_min))
                    
                # Combine all conditions using the AND operator
                if conditions:
                    final_condition = conditions[0]
                    for cond in conditions[1:]:
                        final_condition &= cond  # Combine conditions
            
                    filtered_df = filtered_df[final_condition]
        
                # Use x, y for 2D surface plotting
                x_surface = filtered_df['Prediction' if ml_tar == x_axis else x_axis]
                y_surface = filtered_df['Prediction' if ml_tar == y_axis else y_axis]
                z_surface = filtered_df['Prediction' if ml_tar == z_axis else z_axis]  # Predictions corresponding to x and y
        
                # Reshape to create a grid for surface plotting
                unique_x = np.unique(x_surface)
                unique_y = np.unique(y_surface)
                
                # Create a grid of x and y values
                X_grid, Y_grid = np.meshgrid(unique_x, unique_y)
        
                # Interpolate Z values (predictions) over the grid
                Z_grid = griddata((x_surface, y_surface), z_surface, (X_grid, Y_grid), method='linear')
        
                
                surface_color = color_map[t_value]  
        
                # Add surface plot for the current unique t value
                plotly_fig.add_trace(go.Surface(
                    z=Z_grid,
                    x=X_grid,
                    y=Y_grid,
                    name=f'Surface for {t_axis} = {t_value}',
                    # colorscale=[[0, surface_color], [1, surface_color]],
                    opacity=0.5,
                    # showscale=True,
                    hovertemplate=f'{t_axis}: {t_value}<br>Feature 1: %{{x}}<br>Feature 2: %{{y}}<br>Prediction: %{{z}}<extra></extra>'
                ))
           



        
    if type_model == "Classification":
        
        X_reduced = X.values

        # Create a meshgrid
        x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
        if len(X.columns.tolist())>1:
            y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
        if len(X.columns.tolist()) == 3 and dim_type == '3D':
            z_min, z_max = X_reduced[:, 2].min() - 1, X_reduced[:, 2].max() + 1
        
        print(X.columns.tolist())
        print(len(X.columns.tolist()))
        
        if len(X.columns.tolist()) == 1:
            
            xx = grid = np.arange(x_min, x_max, 0.01)
            
        elif len(X.columns.tolist()) == 2:
            
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                                 np.arange(y_min, y_max, 0.01))
            # Predict the full grid
            grid = np.c_[xx.ravel(), yy.ravel()]
                    
        elif len(X.columns.tolist()) == 3 and dim_type == '3D':
            xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, 0.01),
                                     np.arange(y_min, y_max, 0.01),
                                     np.arange(z_min, z_max, 0.01))            
            # Predict the full grid
            grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

        else:
            print("Build a PCA", "yellow")
            return fig_json_serializable, data_for_plot.to_dict(orient='records') 
        
        print(grid)
            
        # Convert the grid into a pandas DataFrame
        grid_df = pd.DataFrame(grid, columns=X.columns.tolist())
        
        Z = ml_model.predict(preprocessor.fit_transform(grid_df))
        
        df_with_model = grid_df.copy()  # Start with a copy of the grid_df
        df_with_model['Prediction'] = Z   # Add the predictions as a new column  
        
        
        print(df_with_model)
        print(ml_tar, x_axis, y_axis, z_axis)
        
        if (ml_tar == x_axis or ml_tar == y_axis):
            line_trace = px.line(
                df_with_model, 
                x='Prediction' if ml_tar == x_axis else x_axis, 
                y='Prediction' if ml_tar == y_axis else y_axis, 
                color=z_axis if (z_axis is not None and df_with_model[z_axis].dtype == 'object') else None
                )
            plotly_fig.add_trace(line_trace.data[0])
            
        elif ml_tar == z_axis:
        
            # Map class names to unique numeric values for heatmap
            unique_classes = np.unique(Z)
            class_map = {cls: idx for idx, cls in enumerate(unique_classes)}
            Z_numeric = np.vectorize(class_map.get)(Z).reshape(xx.shape)
    
            # Assuming Z_numeric, xx, yy, and unique_classes are already defined
            unique_classes = np.unique(data_for_plot[ml_tar])  # Adjust this line if necessary to get unique classes
            num_classes = len(unique_classes)
            
            colorscale = []
            
            # Define base colors to include based on the number of unique classes
            if num_classes >= 1:
                colorscale.append([0, 'blue'])  # Class 0
            if num_classes >= 2:
                colorscale.append([1 / (num_classes - 1), 'red'])  # Class 1
            if num_classes >= 3:
                colorscale.append([2 / (num_classes - 1), 'green'])  # Class 2
            if num_classes >= 4:
                colorscale.append([3 / (num_classes - 1), 'purple'])  # Class 3
            if num_classes >= 5:
                colorscale.append([4 / (num_classes - 1), 'orange'])  # Class 4
            if num_classes >= 6:
                colorscale.append([5 / (num_classes - 1), 'pink'])  # Class 5
        
            plotly_fig.add_trace(go.Heatmap(
                z=Z_numeric,
                x=np.unique(xx[0]),
                y=np.unique(yy[:, 0]),
                colorscale=colorscale,
                opacity=0.5 ,
                colorbar=dict(title='Classes', tickvals=np.arange(len(unique_classes)), ticktext=unique_classes),
                hovertemplate='Predicted Class: %{z}<br>Feature 1: %{x}<br>Feature 2: %{y}<extra></extra>',
                showscale=True
            ))

    
    
    fig_json_serializable = plotly_fig.to_dict()

    
    plt.close()
    # =============================================================================
    ddi.debug_print(colored("=============================================================================", "green"), debug=Debug)
    
    return fig_json_serializable, data_for_plot.to_dict(orient='records') 


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def figure_add_subplot(fig_json_serializable, data_for_plot, 
                       x_column, y_column, z_column, t_column, yfunc_column, zfunc_column, tfunc_column, graph_type, dim_type,
                       nb_subplots, nb_subplots_row, nb_subplots_col):

    """
    Goal: Create a subplot figure where the original figure, fig_json_serializable, is included as the first subplot.

    Parameters:
    - fig_json_serializable: Dash figure.
    - data_for_plot: Data to plot.
    - x_column: Column in the dataframe
    - y_column: Column in the dataframe (can be None)
    - z_column: Column in the dataframe (can be None)
    - t_column: Column in the dataframe (can be None)
    - yf_column: Function to operate on y_column with the rest of the dataframe
    - zf_column: Function to operate on z_column with the rest of the dataframe
    - tf_column: Function to operate on t_column with the rest of the dataframe
    - graph_type: Type of Graphyque for the figure.
    - dim_type: Graphyque dimension for the figure.
    - nb_subplots: Amount of subplots in the figure.
    - nb_subplots_row: Amount of subplots per row.
    - nb_subplots_col: Amount of subplots per coulumn.
    
    Returns:
    - plotly_fig: The core figure with subplot updated.
    - data_for_plot: Data to plot updated.
    """
    
    plotly_fig = go.Figure(fig_json_serializable)
        
    # Create a subplot figure
    # For example, creating a 2x1 grid of subplots
    fig_with_subplots = make_subplots(rows=nb_subplots_row, cols=nb_subplots_col)
        
    # Add a trace from your existing figure to the first subplot
    for trace in plotly_fig.data:
        fig_with_subplots.add_trace(trace, row=1, col=1)

    # Add empty traces for each subplot cell except for (1, 1)
    for row in range(1, nb_subplots_row + 1):
        for col in range(1, nb_subplots_col + 1):
            if (row, col) != (1, 1):  # Skip the first cell (1, 1)
                # Create an empty trace
                empty_trace = go.Scatter(x=[], y=[], mode='lines', showlegend=False)  # Example empty trace
                fig_with_subplots.add_trace(empty_trace, row=row, col=col)    

    
    # Update selected layout properties of fig_with_subplots from plotly_fig
    fig_with_subplots.update_layout(
        xaxis_title=plotly_fig.layout.xaxis.title.text if plotly_fig.layout.xaxis.title else 'X-Axis',
        yaxis_title=plotly_fig.layout.yaxis.title.text if plotly_fig.layout.yaxis.title else 'Y-Axis',
        plot_bgcolor=plotly_fig.layout.plot_bgcolor,
        paper_bgcolor=plotly_fig.layout.paper_bgcolor,
        font=plotly_fig.layout.font
    )  
    
    plt.close()       
    
    return fig_with_subplots, data_for_plot


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def get_subplot_position(index_subplot, nb_subplots, nb_subplots_row, nb_subplots_col):

    """
    Goal: Determine the row and column position of the subplot corresponding to the index index_subplot.

    Parameters:
    - index_subplot: Index of the subplot been updated.
    - nb_subplots: Amount of subplots in the figure.
    - nb_subplots_row: Amount of subplots per row.
    - nb_subplots_col: Amount of subplots per coulumn.
    
    Returns:
    - row: The row position of subplot index_subplot.
    - col: The column position of subplot index_subplot.
    """    

    # Check if index_subplot is within the valid range
    if index_subplot < 0 or index_subplot >= nb_subplots:
        raise ValueError("index_subplot must be in the range [0, nb_subplots-1]")
    
    # Calculate the row and column positions
    row = index_subplot // nb_subplots_col + 1     # add 1 to convert to 1-based index
    col = index_subplot % nb_subplots_col + 1      # add 1 to convert to 1-based index
    
    return row, col 


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def clean_trace(fig_with_subplots, index_subplot):

    """
    Goal: Clean the subplot from the previous trace.
        
    Parameters:
    - fig_with_subplots: The dash subplot figure.
    - index_subplot: Index of the subplot been updated.
    
    Returns:
    - fig_with_subplots: The cleaned dash subplot figure.
    """        

    # Identify the corresponding xaxis and yaxis labels
    if index_subplot == 0:
        xaxis_to_remove = 'x'  # Use just 'x' for index 0
        yaxis_to_remove = 'y'  # Use just 'y' for index 0
    else:
        xaxis_to_remove = f'x{index_subplot + 1}'  # e.g., 'x2' for index 1
        yaxis_to_remove = f'y{index_subplot + 1}'  # e.g., 'y2' for index 1
        
    # Filter out traces that correspond to the specified subplot
    fig_with_subplots['data'] = [
        trace for trace in fig_with_subplots['data']
        if trace.get('xaxis') != xaxis_to_remove and trace.get('yaxis') != yaxis_to_remove
    ]
    
    return fig_with_subplots


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def transform_trace_to_format(trace, index_subplot):

    """
    Goal: Convert different trace types into a uniform structure that is compatible with subplot configurations.
        
    Parameters:
    - trace: The subplot trace.
    - index_subplot: Index of the subplot been updated.
    
    Returns:
    - new_trace: The subplot updated trace.
    """    

    # Generate axis labels based on index_subplot
    xaxis_label = f'x{index_subplot + 1}'  # For example, x1, x2, etc.
    yaxis_label = f'y{index_subplot + 1}'  # For example, y1, y2, etc.

    new_trace = {
        'type': trace.type,
        'name': trace.name,
        'hovertemplate': getattr(trace, 'hovertemplate', ''),
        'marker': getattr(trace, 'marker', {}),
        'showlegend': getattr(trace, 'showlegend', True),
        'textposition': getattr(trace, 'textposition', ''),
        'xaxis': xaxis_label,  # Use dynamic xaxis_label
        'yaxis': yaxis_label,  # Use dynamic yaxis_label
        'x': trace.x.tolist() if hasattr(trace, 'x') and isinstance(trace.x, np.ndarray) else trace.x,
        'y': trace.y.tolist() if hasattr(trace, 'y') and isinstance(trace.y, np.ndarray) else trace.y
    }
    
    print("trace.type",trace.type)
    
    # If it's a bar trace, eliminate any unsupported properties
    if trace.type == 'bar':
        new_trace.pop('mode', None)  # Bar traces do not have a 'mode'
        new_trace.pop('z', None)   # Bar does not use 'z'

    # If it's a bar trace, eliminate any unsupported properties
    elif trace.type == 'line':
        new_trace.pop('mode', None)  # Bar traces do not have a 'mode'
        new_trace.pop('z', None)   # Bar does not use 'z'

    elif trace.type == 'scatter':
        new_trace.pop('mode', None)  # Bar traces do not have a 'mode'
        # new_trace['size_max'] = getattr(trace, 'size_max', None)
        # new_trace['fillcolor'] = getattr(trace, 'fillcolor', None)
        # new_trace['animation_frame'] = getattr(trace, 'animation_frame', None)
        new_trace.pop('z', None)   # Bar does not use 'z'
    
    elif trace.type == 'heatmap':
        new_trace.pop('marker', None)  # Bar traces do not have a 'mode'
        new_trace.pop('textposition', None)  # Bar traces do not have a 'mode'
    
    return new_trace


"""#=============================================================================
   #=============================================================================
   #============================================================================="""

def figure_update_subplot(df, df_col_string, fig_with_subplots, data_for_plot, 
                       x_column, y_column, z_column, t_column, yf_column, zf_column, tf_column, graph_type, dim_type,
                       smt_dropdown_value, smt_order_value, sub_bot_smt_value,
                       index_subplot, nb_subplots, nb_subplots_row, nb_subplots_col, Large_file_memory):
    
    """
    Goal: Update one subplot inside the figure regarding the inputs.

    Parameters:
    - df: dataframe.
    - df_col_string: List of columns in the DataFrame that are of object type.
    - fig_with_subplots: Dash figure with subplots.
    - data_for_plot: Data to plot.
    - x_column: Column in the dataframe
    - y_column: Column in the dataframe (can be None)
    - z_column: Column in the dataframe (can be None)
    - t_column: Column in the dataframe (can be None)
    - yf_column: Function to operate on y_column with the rest of the dataframe
    - zf_column: Function to operate on z_column with the rest of the dataframe
    - tf_column: Function to operate on t_column with the rest of the dataframe
    - graph_type: Type of Graphyque for the figure.
    - dim_type: Graphyque dimension for the figure.
    - sub_bot_smt_value: Button to apply the smoothing.
    - smt_dropdown_value: Type of smoothing for the data.
    - smt_dropdown_value: Order of the smoothing for the data.
    - index_subplot: Index of the subplot been updated.
    - nb_subplots: Amount of subplots in the figure.
    - nb_subplots_row: Amount of subplots per row.
    - nb_subplots_col: Amount of subplots per coulumn.
    
    Returns:
    - plotly_fig: The core figure updated.
    - data_for_plot: Data to plot updated.
    """
        
    Debug = aps.Debug
    
    row_index, col_index = get_subplot_position(index_subplot, nb_subplots, nb_subplots_row, nb_subplots_col)
    ddi.debug_print(f"Row: {row_index}, Column: {col_index}", debug=Debug)    

    fig_json_serializable = go.Figure()
    # Create the label of the figure
    figname, xlabel, ylabel, zlabel, tlabel = label_fig(x_column, y_column, z_column, t_column, yf_column, zf_column, graph_type, dim_type, True, df_col_string)  
    
    if x_column is not None: 
        ddi.debug_print("Extract from data base the required column and prepare them for the figure.", debug=Debug)
        Para, data_for_plot, x_column, y_column, z_column, t_column = dpp.data_preparation_for_plot(df, df_col_string , x_column, y_column, z_column, t_column, yf_column, zf_column, tf_column, graph_type, Large_file_memory)
        ddi.debug_print("The data ready to be ploted is:", debug=Debug)
        ddi.debug_print(data_for_plot, debug=Debug)
        ddi.debug_print("", debug=Debug)
        # Add the core of the figure
        ddi.debug_print("############## Core figure creation ##############", debug=Debug)
        figure_returned, data_for_plot, xlabel, ylabel, zlabel, tlabel = figure_plotly(fig_json_serializable, x_column, y_column, z_column, t_column, yf_column, zf_column, tf_column, graph_type, dim_type, smt_dropdown_value, smt_order_value, sub_bot_smt_value, data_for_plot, xlabel, ylabel, zlabel, tlabel, df_col_string)       
        fl.fig_update_layout(figure_returned, data_for_plot,figname,xlabel,ylabel,zlabel,tlabel,x_column,y_column,z_column,t_column,graph_type, dim_type,df_col_string)   
        ddi.debug_print("", debug=Debug)
        
    traces = figure_returned.data    
    if len(traces) == 0:  # Check if there is any trace
        ddi.debug_print("No traces found in the figure returned.", debug=Debug)
        return fig_with_subplots, data_for_plot  # Nothing to add, return as is

    fig_with_subplots = clean_trace(fig_with_subplots, index_subplot)
    for trace in traces:
        modified_trace = transform_trace_to_format(trace, index_subplot)
    
        # Add the modified trace to the figure's data
        fig_with_subplots['data'].append(modified_trace)
    
    
    # Now create the figure using the cleaned data
    plotly_fig = go.Figure(fig_with_subplots)

    
    # Determine the specific xaxis and yaxis labels based on index_subplot
    if index_subplot == 0:
        xaxis_name = 'xaxis'
        yaxis_name = 'yaxis'
    else:
        xaxis_name = f'xaxis{index_subplot+1}'
        yaxis_name = f'yaxis{index_subplot+1}'
    
    # Update selected layout properties of fig_with_subplots from plotly_fig for the specified subplot
    plotly_fig.update_layout(
        **{
            f'{xaxis_name}_title': figure_returned.layout['xaxis'].title.text,
            f'{yaxis_name}_title': figure_returned.layout['yaxis'].title.text,
            'plot_bgcolor': figure_returned.layout.plot_bgcolor,
            'paper_bgcolor': figure_returned.layout.paper_bgcolor,
            'font': figure_returned.layout.font
        }
    )
    
    ddi.debug_print(plotly_fig, debug=Debug)
    
    # Ensure data_for_plot is serializable
    if isinstance(data_for_plot, pd.DataFrame):
        data_for_plot = data_for_plot.to_dict(orient='records')  # Convert DataFrame to a dictionary
        
    return plotly_fig, data_for_plot
        
