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
import Machine_learning_functions as mlf


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


def create_figure(df, df_col_string, x_column, y_column, z_column, yf_column, zf_column, g_column, d_column, smt_dropdown_value, smt_order_value, sub_bot_smt_value, Large_file_memory):

    """
    Goal: Create a sophisticated figure which adapt to any input variable.

    Parameters:
    - df: dataframe
    - x_column: Column in the dataframe
    - y_column: Column in the dataframe (can be None)
    - z_column: Column in the dataframe (can be None)
    - yf_column: Function to operate on y_column with the rest of the dataframe
    - zf_column: Function to operate on z_column with the rest of the dataframe
    - g_column: Type of Graphyque for the figure.
    - d_column: Graphyque dimension for the figure.
    - sub_bot_smt_value: Button to apply the smoothing.
    - smt_dropdown_value: Type of smoothing for the data.
    - smt_dropdown_value: Order of the smoothing for the data.
    - Large_file_memory: Estimate if the file is too large to be open with panda

    Returns:
    - fig_json_serializable: The finalized plotly figure. 
    """


    # =============================================================================
    print(colored("========================= Start figure creation =========================", "green"))
    # =============================================================================      
    # Create a Dash compatible Plotly graph figure
    fig_json_serializable = go.Figure()  # This figure can now be used with dcc.Graph in Dash
    
    # Create the label of the figure
    figname, xlabel, ylabel, zlabel = label_fig(x_column, y_column, z_column, yf_column, zf_column, g_column, d_column, True, df_col_string)  
    data_for_plot = []
    if x_column is not None: 
        print("Extract from data base the required column and prepare them for the figure.")
        Para, data_for_plot, x_column, y_column, z_column = dpp.data_preparation_for_plot(df, df_col_string , x_column, y_column, z_column, yf_column, zf_column, g_column, Large_file_memory)
        print("The data ready to be ploted is:")
        print(data_for_plot)
        print()
        # Add the core of the figure
        print("############## Core figure creation ##############")
        fig_json_serializable, data_for_plot, xlabel, ylabel, zlabel = figure_plotly(fig_json_serializable, x_column, y_column, z_column, yf_column, zf_column, g_column, d_column, smt_dropdown_value, smt_order_value, sub_bot_smt_value, data_for_plot, xlabel, ylabel, zlabel, df_col_string)
    
    # Update the figure layout
    print("############## Update figure layout ##############")
    fl.fig_update_layout(fig_json_serializable, data_for_plot,figname,xlabel,ylabel,zlabel,x_column,y_column,z_column,g_column,d_column,df_col_string)       
    plt.close()
    # =============================================================================
    print(colored("=============================================================================", "green"))
    if x_column is None: 
        return fig_json_serializable, None
    
    return fig_json_serializable, data_for_plot.to_dict(orient='records')


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def label_fig(x_column, y_column, z_column, yf_column, zf_column, g_column, d_column, init, df_col_string):

    """
    Goal: Create the figure labels.

    Parameters:
    - x_column: Column in the dataframe (can be None).
    - y_column: Column in the dataframe (can be None).
    - z_column: Column in the dataframe (can be None).
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
    

    # columns = [x_column, y_column, z_column]
    
    # # Replace 'tconst' with 'productions' in each element of the list, if it's not None
    # columns = [col.replace('tconst', 'productions') if col is not None else col for col in columns]
    
    # # Unpack results back to original variables if needed
    # x_column, y_column, z_column = columns
    
    
    name_to_work_on = "production" #  production, directors, writers
    
    df_col_string = [col[:-6] if col.endswith('_split') else col for col in df_col_string]

    if init == False:
        if x_column is not None: 
            figname = 'Amount of '+name_to_work_on+' over the ' + x_column
    
            if x_column == 'count':
                xlabel = 'Amount of '+name_to_work_on
            elif 'avg_' in x_column:
                xlabel = 'Average '+x_column[4:]#+' of the movies'
            else:
                xlabel = x_column
            
            if y_column == 'count':
                ylabel = 'Amount of '+name_to_work_on
            elif 'avg_' in y_column:
                ylabel = 'Average '+y_column[4:]#+' of the movies'              
            else:
                ylabel = y_column
            
            if z_column is not None:
                if z_column == 'count':
                    zlabel = 'Amount of '+name_to_work_on
                elif 'avg_' in z_column:
                    zlabel = 'Average '+z_column[4:]#+' of the movies'
                else:
                    zlabel = z_column
                
                if yf_column == 'Avg' and zf_column == 'Weight on y':
                    ylabel = 'Amount of '+name_to_work_on
                if yf_column == 'Avg on the ordinate' and zf_column == 'Weight on y':
                    ylabel = 'Average '+y_column[4:]#+' of the movies'
                    
            else:
                zlabel = None

            if d_column == "2D":
                if g_column == 'Colormesh':
                    ylabel = y_column
                else:
                    ylabel = "None"
                zlabel = "None"
    
    
        else: 
            figname = 'No data selected'
            xlabel, ylabel, zlabel = "None","None","None"
    
    else:
        figname = 'No data selected'
        xlabel, ylabel, zlabel = "None","None","None"        
    
    if x_column in df_col_string:
        xlabel_temp = xlabel
        xlabel = ylabel
        ylabel = xlabel_temp
    
    return figname, xlabel, ylabel, zlabel


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def figure_plotly(plotly_fig, x_column, y_column, z_column, yf_column, zf_column, g_column, d_column, smt_dropdown_value, smt_order_value, sub_bot_smt_value, data_for_plot, xlabel, ylabel, zlabel, df_col_string):

    """
    Goal: Create the plot inside the figure regarding the inputs.

    Parameters:
    - plotly_fig: Dash figure.
    - x_column: Column in the dataframe
    - y_column: Column in the dataframe (can be None)
    - z_column: Column in the dataframe (can be None)
    - yf_column: Function to operate on y_column with the rest of the dataframe
    - zf_column: Function to operate on z_column with the rest of the dataframe
    - g_column: Type of Graphyque for the figure.
    - d_column: Graphyque dimension for the figure.
    - sub_bot_smt_value: Button to apply the smoothing.
    - smt_dropdown_value: Type of smoothing for the data.
    - smt_dropdown_value: Order of the smoothing for the data.
    - data_for_plot: Data to plot.
    - xlabel: The xlabel of the axis (can be None).
    - ylabel: The ylabel of the axis (can be None).
    - zlabel: The zlabel of the axis (can be None).
    - df_col_string: List of columns in the DataFrame that are of object type.

    Returns:
    - plotly_fig: The core figure.
    """
    
    df_col_string = [col + '_split' for col in df_col_string]

    # Define a list of colors for the bars
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    legend = "None"
    
    x_axis = x_column
    y_axis = 'count'
    z_axis = None
    t_axis = None
    if str(y_column)!='None' and yf_column != "Value in x_y interval":
        z_axis = y_column
    if x_column in df_col_string:
        x_axis = 'count'
        y_axis = x_column

    if yf_column == "Avg":
        z_axis = 'avg_' + y_column

    if yf_column == "Avg on the ordinate":
        x_axis = x_column
        y_axis = 'avg_' + y_column
        z_axis = 'count'
        if x_column in df_col_string:
            x_axis = 'avg_' + y_column
            y_axis = x_column
            z_axis = 'count'
        
    if z_column is not None and zf_column == "Avg" and yf_column != "Value in x_y interval":
        t_axis = 'avg_' + z_column
    if z_column is not None and zf_column == "Avg" and yf_column == "Value in x_y interval":
        z_axis = 'avg_' + z_column
    if z_column is not None and zf_column == "Avg on the ordinate" and yf_column != "Value in x_y interval":
        y_axis = 'avg_' + z_column
        t_axis = 'count'
    if z_column is not None and zf_column == "Avg on the ordinate" and yf_column == "Value in x_y interval":
        y_axis = 'avg_' + z_column
        z_axis = 'count'
        
    if z_column is not None and zf_column == "Weight on y":
        # y_axis = 'sum_' + z_column
        t_axis = 'standard_error'   

    if d_column=="2D" and g_column=="Colormesh":    
        x_axis = x_column
        y_axis = y_column
        z_axis = 'count'
        if z_column is not None:
            z_axis = 'avg_' + z_column

    print("x_axis=", x_axis)
    print("y_axis=", y_axis)
    if str(y_column)!='None':
        print("z_axis=", z_axis)
    if str(z_column)!='None':
        print("t_axis=", t_axis)


    # Rename the label of the figure
    figname, xlabel, ylabel, zlabel = label_fig(x_axis, y_axis, z_axis, yf_column, zf_column, g_column, d_column, False, df_col_string)  
    
    
    if d_column=="1D": 
        
        # Check if 'startYear' is in the DataFrame
        if 'startYear' in data_for_plot.columns:
            # Check if startYear 0 exists and drop it if it does
            if 0 in data_for_plot['startYear'].values:
                data_for_plot.drop(data_for_plot[data_for_plot['startYear'] == 0].index, inplace=True)
            
            # Resetting index if needed
            data_for_plot.reset_index(drop=True, inplace=True)
        
        
        if str(y_column)=='None':
            
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
        
        #Case where y_column is None and z_column is None
        elif str(y_column)!='None' and str(z_column)=='None':           

            if x_column in df_col_string and "Movie" not in g_column:
                # Grouping y_column values
                n = 10  # Number of top categories to keep
                data_for_plot = fd.group_small_values(data_for_plot, y_axis, x_axis, n)

            if y_column in df_col_string and "Movie" not in g_column:
                # Grouping y_column values
                n = 7  # Number of top categories to keep
                data_for_plot = fd.group_small_values(data_for_plot, z_axis, y_axis, n, x_axis)


            data_for_plot = smoothing_data(sub_bot_smt_value, smt_dropdown_value, smt_order_value, data_for_plot, x_axis, y_axis, z_axis, df_col_string)

            if "Histogram" in g_column:
                if "Movie" in g_column:    
                    # First, create a new DataFrame to calculate cumulative sums
                    cumulative_data = data_for_plot.groupby([y_axis, z_axis])[x_axis].sum().reset_index()
                    # Sort the DataFrame by year (z_axis) for cumulative calculation
                    cumulative_data = cumulative_data.sort_values(by=[y_axis, z_axis])
                    # Calculate the cumulative sum for each genre (x_axis)
                    cumulative_data['count'] = cumulative_data.groupby(y_axis)[x_axis].cumsum()
                    # Use cumulative_data for the plot
                    data_for_plot = cumulative_data

                    # Create the animation frames by grouping your data by the z_axis
                    frames = data_for_plot.groupby(z_axis)
                    # For each frame, sort the data based on the y values or any other criteria you choose
                    sorted_frames = {}
                    for name, group in frames:
                        sorted_group = group.sort_values(by=y_axis, ascending=False)  # Sort by y_axis for consistency, adjust criteria as needed
                        sorted_frames[name] = sorted_group
                    # Concatenate sorted frames back into a single DataFrame
                    data_for_plot = pd.concat(sorted_frames.values(), ignore_index=True)

                plotly_fig = px.bar(
                   data_for_plot, 
                   x=x_axis, 
                   y=y_axis,
                   color=z_axis if "Movie" not in g_column else None,
                   animation_frame=z_axis if "Movie" in g_column else None,
                   range_y=[data_for_plot[y_axis].min(), data_for_plot[y_axis].max()] if "Movie" in g_column else None
                   )
                
                if "Movie" in g_column:    
                    # fd.make_movie(plotly_fig)
                    plotly_fig.write_html(x_axis+'_'+y_axis+'_'+z_axis+"_animation_plot.html")
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
                if x_column in df_col_string:
                    x_axis = y_column
                    xlabel = y_column
                else:
                    y_axis = y_column
                    ylabel = y_column
                plotly_fig = px.box(
                    data_for_plot, 
                    x=x_axis, 
                    y=y_axis,
                    points=False)

        #Case where z_column is not None
        else:
                        
            if y_column in df_col_string:
                # Grouping y_column values
                n = 7  # Number of top categories to keep
                if zf_column == "Avg":
                    data_for_plot = fd.group_small_values(data_for_plot, z_axis, y_axis, n, x_axis)
                elif zf_column == "Avg on the ordinate":
                    data_for_plot = fd.group_small_values(data_for_plot, z_axis, t_axis, n, x_axis)
                elif zf_column == "Weight on y":
                    data_for_plot = fd.group_small_values(data_for_plot, z_axis, y_axis, n, x_axis)

            data_for_plot = smoothing_data(sub_bot_smt_value, smt_dropdown_value, smt_order_value, data_for_plot, x_axis, y_axis, z_axis, df_col_string)
                        
            # y_values = data_for_plot[y_column].unique()
            if g_column=="Histogram" and (zf_column == "Avg" or zf_column == "Avg on the ordinate"):
                if "Movie" in g_column:    
                    # Create the animation frames by grouping your data by the z_axis
                    frames = data_for_plot.groupby(z_axis)
                    # For each frame, sort the data based on the y values or any other criteria you choose
                    sorted_frames = {}
                    for name, group in frames:
                        sorted_group = group.sort_values(by=y_axis, ascending=False)  # Sort by y_axis for consistency, adjust criteria as needed
                        sorted_frames[name] = sorted_group
                    # Concatenate sorted frames back into a single DataFrame
                    data_for_plot = pd.concat(sorted_frames.values())
                plotly_fig = px.bar(
                   data_for_plot, 
                   x=x_axis, 
                   y=y_axis,
                   color=z_axis if "Movie" not in g_column else None,
                   animation_frame=z_axis if "Movie" in g_column else None,
                   range_y=[sorted_data_for_plot[y_axis].min(), sorted_data_for_plot[y_axis].max()] if "Movie" in g_column else None
                   )
            elif g_column=="Curve" and (zf_column == "Avg"):
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
            elif g_column=="Curve" and (zf_column == "Avg on the ordinate"):
                plotly_fig = px.line(
                    data_for_plot, 
                    x=x_axis, 
                    y=y_axis,
                    color=z_axis if "Movie" not in g_column else None,
                    animation_frame=z_axis if "Movie" in g_column else None,
                    line_group=g_column if "Movie" in g_column else None
                    )
            elif g_column=="Scatter" and (zf_column == "Avg" or zf_column == "Avg on the ordinate" or zf_column == "Weight on y"):
                plotly_fig = px.scatter(
                    data_for_plot,
                    x=x_axis,
                    y=y_axis,
                    size=t_axis if zf_column == "Avg" or zf_column == "Weight on y" else None,
                    size_max=60,
                    color=z_axis if "Movie" not in g_column else None,
                    animation_frame=z_axis if "Movie" in g_column else None
                )

    if g_column=="Pie": #d_column=="2D" and 

            if x_column in df_col_string:
                # Grouping y_column values
                n = 24  # Number of top categories to keep
                data_for_plot = fd.group_small_values(data_for_plot, x_column, 'count', n)

            # x_values,fig_x_value,y_values,fig_y_value=None,None,None,None
            plotly_fig = px.pie(
                data_for_plot, 
                values="count", 
                names=x_column
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

                    
    if d_column == "3D" and g_column == "Histogram":
        
        # Pivoting the DataFrame to create a grid for surface plot
        pivoted_data = data_for_plot.pivot(index=y_column, columns=x_column, values='count')
        # Fill NaN values with zeros or an appropriate value for the surface
        pivoted_data = pivoted_data.fillna(0)
        # Now, create the surface plot

        plotly_fig = go.Figure(
            data=[go.Surface(z=pivoted_data.values, x=pivoted_data.columns, y=pivoted_data.index)])

    return plotly_fig, data_for_plot, xlabel, ylabel, zlabel  


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
    
    if sub_bot_smt_value % 2 == 1:
        
        print("############## Smoothing #################")
        
        data_for_plot['original_index'] = data_for_plot.index
        
        if z_axis is None or z_axis not in df_col_string:
            window_length = len(data_for_plot[x_axis])//5
            data_for_plot[y_axis] = signal.savgol_filter(data_for_plot[y_axis],
                                   window_length, # window size used for filtering
                                   smt_order_value)
            print("window_length=",window_length)
            print("Data updated by the smoothing")
            print(data_for_plot)

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
                
                print("window_length=",window_length)
                print("Amount of data", len(group[y_axis]))
                print()
                
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
        
            
            print("Data updated by the smoothing")
            print(data_for_plot)
            
    return data_for_plot
            

"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def figure_add_trace(fig_json_serializable, data_for_plot, df_col_string, x_column, y_column, z_column, yf_column, zf_column, graph_type, dim_type, reg_type, reg_order, test_size_val=0.2):

    """
    Goal: Add a trace inside the figure regarding the inputs.

    Parameters:
    - fig_json_serializable: Dash figure.
    - data_for_plot: Dataframe which has been use to create the figure that is re-opened in this function.
    - df_col_string: List of columns in the DataFrame that are of object type.
    - x_column: Column in the dataframe
    - y_column: Column in the dataframe (can be None)
    - z_column: Column in the dataframe (can be None)
    - yf_column: Function to operate on y_column with the rest of the dataframe
    - zf_column: Function to operate on z_column with the rest of the dataframe
    - graph_type: Type of Graphyque for the figure.
    - dim_type: Graphyque dimension for the figure.
    - reg_type: Type of regression for the data.
    - reg_order: Order of the regression for the data.
    - test_size_val: The ratio of testing value for the fit.

    Returns:
    - fig_json_serializable: Dash figure updated with the trace.
    - data_for_plot: Dataframe updated with the trace.
    """
    
    
    plotly_fig = go.Figure(fig_json_serializable)

    df_col_string = [col + '_split' for col in df_col_string]
    
    # Define a list of colors for the bars
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'] 
    
    x_axis = x_column
    y_axis = 'count'
    z_axis = None
    t_axis = None
    if str(y_column)!='None':
        z_axis = y_column
    if x_column in df_col_string:
        x_axis = 'count'
        y_axis = x_column

    if yf_column == "Avg":
        z_axis = 'avg_' + y_column

    if yf_column == "Avg on the ordinate":
        x_axis = x_column
        y_axis = 'avg_' + y_column
        z_axis = 'count'
        if x_column in df_col_string:
            x_axis = 'avg_' + y_column
            y_axis = x_column
            z_axis = 'count'
    
    if z_column is not None and zf_column == "Avg":
        t_axis = 'avg_' + z_column
    if z_column is not None and zf_column == "Avg on the ordinate":
        y_axis = 'avg_' + z_column
        t_axis = 'count'
    if z_column is not None and zf_column == "Weight on y":
        # y_axis = 'sum_' + z_column
        t_axis = 'standard_error'        

    print("x_axis=", x_axis)
    print("y_axis=", y_axis)
    print("z_axis=", z_axis)
    print("t_axis=", t_axis)

    # Creating a DataFrame
    data_for_plot = pd.DataFrame(data_for_plot)

    # Resetting the index to have a clean index
    data_for_plot.reset_index(drop=True, inplace=True)
    
    # Calculate the offset
    offset = data_for_plot[x_axis].values.min()
    
    # Adjust x values by subtracting the minimum value
    x_offset = data_for_plot[x_axis].values - offset
    x = x_offset.reshape(-1, 1)
    y = data_for_plot[y_axis].values.reshape(-1, 1)
    
    weights = None
    if z_column is not None and (zf_column == "Weight on y" or zf_column == "Avg"):
        if yf_column == "Avg":
            y = data_for_plot[z_axis].values.reshape(-1, 1)
        elif yf_column == "Avg on the ordinate":
            y = data_for_plot[y_axis].values.reshape(-1, 1)
        weights = data_for_plot[t_axis].values.reshape(-1, 1).flatten()    
    
    # Make ML regression
    mlf.make_regression_model(data_for_plot,x,y,weights,reg_type,reg_order,test_size_val)

    # Plotly figure with the original data and the regression line
    plotly_fig.add_trace(go.Scatter(
        x=data_for_plot[x_axis],
        y=data_for_plot['predicted_count'],
        mode='lines',
        name=reg_type if reg_type in ['Linear Regression', 'Decision Tree','k-NN'] else "Poly deg "+str(reg_order),#polynomial_equation,
        line=dict(dash='dash', width=2)  # Customizing line color and width
    ))
    
        
    fig_json_serializable = plotly_fig.to_dict()
        
    plt.close()
    # =============================================================================
    print(colored("=============================================================================", "green"))
    
    return fig_json_serializable, data_for_plot.to_dict(orient='records') 


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def figure_add_subplot(fig_json_serializable, data_for_plot, 
                       x_column, y_column, z_column, yfunc_column, zfunc_column, graph_type, dim_type,
                       nb_subplots, nb_subplots_row, nb_subplots_col):

    """
    Goal: Create a subplot figure where the original figure, fig_json_serializable, is included as the first subplot.

    Parameters:
    - fig_json_serializable: Dash figure.
    - data_for_plot: Data to plot.
    - x_column: Column in the dataframe
    - y_column: Column in the dataframe (can be None)
    - z_column: Column in the dataframe (can be None)
    - yf_column: Function to operate on y_column with the rest of the dataframe
    - zf_column: Function to operate on z_column with the rest of the dataframe
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

    # If it's a bar trace, eliminate any unsupported properties
    if trace.type == 'bar':
        new_trace.pop('mode', None)  # Bar traces do not have a 'mode'
        new_trace.pop('z', None)   # Bar does not use 'z'

    # If it's a bar trace, eliminate any unsupported properties
    elif trace.type == 'line':
        new_trace.pop('mode', None)  # Bar traces do not have a 'mode'
        new_trace.pop('z', None)   # Bar does not use 'z'

    if trace.type == 'scatter':
        new_trace.pop('mode', None)  # Bar traces do not have a 'mode'
        # new_trace['size_max'] = getattr(trace, 'size_max', None)
        # new_trace['fillcolor'] = getattr(trace, 'fillcolor', None)
        # new_trace['animation_frame'] = getattr(trace, 'animation_frame', None)
        new_trace.pop('z', None)   # Bar does not use 'z'
    
    if trace.type == 'heatmap':
        new_trace.pop('marker', None)  # Bar traces do not have a 'mode'
        new_trace.pop('textposition', None)  # Bar traces do not have a 'mode'
    
    return new_trace


"""#=============================================================================
   #=============================================================================
   #============================================================================="""

def figure_update_subplot(df, df_col_string, fig_with_subplots, data_for_plot, 
                       x_column, y_column, z_column, yf_column, zf_column, graph_type, dim_type,
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
    - yf_column: Function to operate on y_column with the rest of the dataframe
    - zf_column: Function to operate on z_column with the rest of the dataframe
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
        
    row_index, col_index = get_subplot_position(index_subplot, nb_subplots, nb_subplots_row, nb_subplots_col)
    print(f"Row: {row_index}, Column: {col_index}")    

    fig_json_serializable = go.Figure()
    # Create the label of the figure
    figname, xlabel, ylabel, zlabel = label_fig(x_column, y_column, z_column, yf_column, zf_column, graph_type, dim_type, True, df_col_string)  
    
    if x_column is not None: 
        print("Extract from data base the required column and prepare them for the figure.")
        Para, data_for_plot, x_column, y_column, z_column = dpp.data_preparation_for_plot(df, df_col_string , x_column, y_column, z_column, yf_column, zf_column, graph_type, Large_file_memory)
        print("The data ready to be ploted is:")
        print(data_for_plot)
        print()
        # Add the core of the figure
        print("############## Core figure creation ##############")
        figure_returned, data_for_plot, xlabel, ylabel, zlabel = figure_plotly(fig_json_serializable, x_column, y_column, z_column, yf_column, zf_column, graph_type, dim_type, smt_dropdown_value, smt_order_value, sub_bot_smt_value, data_for_plot, xlabel, ylabel, zlabel, df_col_string)       
        fl.fig_update_layout(figure_returned, data_for_plot,figname,xlabel,ylabel,zlabel,x_column,y_column,z_column,graph_type, dim_type,df_col_string)   
        print()
        
    traces = figure_returned.data    
    if len(traces) == 0:  # Check if there is any trace
        print("No traces found in the figure returned.")
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
    
    print(plotly_fig)
    
    # Ensure data_for_plot is serializable
    if isinstance(data_for_plot, pd.DataFrame):
        data_for_plot = data_for_plot.to_dict(orient='records')  # Convert DataFrame to a dictionary
        
    return plotly_fig, data_for_plot
        
