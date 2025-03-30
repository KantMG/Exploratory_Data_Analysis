#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 17:50:57 2024

@author: quentin
"""


"""#=============================================================================
   #=============================================================================
   #=============================================================================

    Dictionnary of functions for dataframe preparation before plot

#=============================================================================
   #=============================================================================
   #============================================================================="""

import pandas as pd
import numpy as np

import Exploratory_Data_Analysis.function_dataframe as fd

"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def data_preparation_for_plot(df_temp, df_col_string, x_column, y_column, z_column, t_column, yf_column, zf_column, tf_column, g_column, Large_file_memory):

    """
    Goal: Get the pivot of the Count table of the dataframe.
    From a table of dimension x with n indexes to a table of dimension x+1 with n-1 index

    Parameters:
    - df_temp: dataframe which has been created temporary
    - df_col_string: List of columns in the DataFrame that are of object type.
    - x_column: Column in the dataframe
    - y_column: Column in the dataframe (can be None)
    - z_column: Column in the dataframe (can be None)
    - t_column: Column in the dataframe (can be None)
    - yf_column: Function to operate on y_column with the rest of the dataframe
    - zf_column: Function to operate on z_column with the rest of the dataframe
    - tf_column: Function to operate on t_column with the rest of the dataframe
    - g_column: Type of Graphyque for the figure.
    - Large_file_memory: Estimate if the file is too large to be open with panda and use dask instead.

    Returns:
    - Para: List of column in the dataframe (can be different of [x_column,y_column])
    - data_for_plot: Data to plot.
    - x_column: Column in the dataframe (it could have change)
    - y_column: Column in the dataframe (it could have change)
    - z_column: Column in the dataframe (it could have change)
    - t_column: Column in the dataframe (it could have change)
    """
        
    df_col_string = [col + '_split' for col in df_col_string]
    
    # print("Delete the rows with unknown value and split the column with multiple value per cell.")
    Para, df_temp, x_column, y_column, z_column, t_column = delete_rows_unknow_and_split(df_temp, x_column, y_column, z_column, t_column, Large_file_memory)

    # if yf_column == "Value in x_y interval":
    #     data_for_plot, x_column, y_column, z_column, t_column = count_value_x_y_interval(df_temp, x_column, y_column, z_column, t_column)
    #     return Para, data_for_plot, x_column, y_column, z_column
    
    print("delete_rows_unknow_and_split done")
         
    
    #Case where y_column is None
    if str(y_column)=='count' and str(z_column)=='None':
        # Get the Count table of the dataframe  
        data_for_plot=df_temp.value_counts(dropna=False).reset_index(name='count') #dropna=False to count nan value
        # sort the data in function of column Para_sorted
        data_for_plot = data_for_plot.sort_values(by=Para[0], ascending=True)
        
    elif str(x_column)=='count' and str(z_column)=='None':
        # Get the Count table of the dataframe  
        data_for_plot=df_temp.value_counts(dropna=False).reset_index(name='count') #dropna=False to count nan value
        # sort the data in function of column Para_sorted
        data_for_plot = data_for_plot.sort_values(by=Para[1], ascending=True)


    #Case where y_column is not None and z_column is None
    elif str(y_column)=='count' and str(z_column)!='None' and str(t_column)=='None':   

        if zf_column == "Avg":
            data_for_plot = df_temp.groupby([x_column]).agg(
                avg_z_column=('{}'.format(z_column), 'mean'),
                count=('{}'.format(z_column), 'size')
            ).reset_index()
            avg_col_name = 'avg_' + z_column
            data_for_plot.rename(columns={'avg_z_column': avg_col_name}, inplace=True)        
        else:
            data_for_plot = df_temp.groupby([x_column, z_column]).size().reset_index(name='count')


    elif str(z_column)=='count' and str(t_column)=='None':   

        if yf_column == "Avg":
            data_for_plot = df_temp.groupby([x_column]).agg(
                avg_y_column=('{}'.format(y_column), 'mean'),
                count=('{}'.format(y_column), 'size')
            ).reset_index()
            avg_col_name = 'avg_' + y_column
            data_for_plot.rename(columns={'avg_y_column': avg_col_name}, inplace=True)        
        else:
            data_for_plot = df_temp.groupby([x_column, y_column]).size().reset_index(name='count')


    elif str(z_column)=='No count' and str(t_column)=='None':   

        data_for_plot = df_temp[[x_column, y_column]]

            
    # #Case where z_column is not None
    elif str(y_column)=='count' and str(t_column)!='None':   

        
        if zf_column=="Avg" and tf_column == "Avg":
            
            data_for_plot = df_temp.groupby([x_column]).agg(
                avg_z_column=('{}'.format(z_column), 'mean'),
                avg_t_column=('{}'.format(t_column), 'mean'),
                count=('{}'.format(t_column), 'size')
            ).reset_index()
                        
            # Renaming the columns
            data_for_plot.rename(columns={
                'avg_z_column': 'avg_' + z_column,
                'avg_t_column': 'avg_' + t_column
            }, inplace=True)
           
            
        elif zf_column=="Avg" and tf_column == "Weight on y":

            data_for_plot = df_temp.groupby([x_column]).agg(
                avg_z_column=('{}'.format(z_column), 'mean'),
                sum_t_column=('{}'.format(t_column), 'sum'),
                count=('{}'.format(t_column), 'size')
            ).reset_index()

            # We'll use sum_numVotes as the number of observations for each startYear
            data_for_plot['sum_t_column'] = data_for_plot['avg_z_column'] / np.sqrt(data_for_plot['sum_t_column'])
                        
            # Renaming the columns
            data_for_plot.rename(columns={
                'avg_z_column': 'avg_' + z_column,
                'sum_t_column': 'standard_error',
            }, inplace=True)       


        elif zf_column is None and tf_column == "Avg":

            data_for_plot = df_temp.groupby([x_column, z_column]).agg(
                avg_t_column=('{}'.format(t_column), 'mean'),
                count=('{}'.format(t_column), 'size')
            ).reset_index()
                        
            # Renaming the columns
            data_for_plot.rename(columns={
                'avg_t_column': 'avg_' + t_column
            }, inplace=True)


        elif zf_column == "Avg" and tf_column is None:

            data_for_plot = df_temp.groupby([x_column, t_column]).agg(
                avg_z_column=('{}'.format(z_column), 'mean'),
                count=('{}'.format(z_column), 'size')
            ).reset_index()
                        
            # Renaming the columns
            data_for_plot.rename(columns={
                'avg_z_column': 'avg_' + z_column
            }, inplace=True)

        else:
            data_for_plot = df_temp.groupby([x_column, z_column, t_column]).size().reset_index(name='count')




    # #Case where z_column is not None
    elif str(z_column)=='count' and str(t_column)!='None':   

        
        if yf_column=="Avg" and tf_column == "Avg":
            
            data_for_plot = df_temp.groupby([x_column]).agg(
                avg_y_column=('{}'.format(y_column), 'mean'),
                avg_t_column=('{}'.format(t_column), 'mean'),
                count=('{}'.format(t_column), 'size')
            ).reset_index()
                        
            # Renaming the columns
            data_for_plot.rename(columns={
                'avg_y_column': 'avg_' + y_column,
                'avg_t_column': 'avg_' + t_column
            }, inplace=True)
           
            
        elif yf_column=="Avg" and tf_column == "Weight on y":

            data_for_plot = df_temp.groupby([x_column]).agg(
                avg_y_column=('{}'.format(y_column), 'mean'),
                sum_t_column=('{}'.format(t_column), 'sum'),
                count=('{}'.format(t_column), 'size')
            ).reset_index()

            # We'll use sum_numVotes as the number of observations for each startYear
            data_for_plot['sum_t_column'] = data_for_plot['avg_y_column'] / np.sqrt(data_for_plot['sum_t_column'])
                        
            # Renaming the columns
            data_for_plot.rename(columns={
                'avg_y_column': 'avg_' + y_column,
                'sum_t_column': 'standard_error',
            }, inplace=True)  
        
        else:
            data_for_plot = df_temp.groupby([x_column, y_column, t_column]).size().reset_index(name='count')
    


    # #Case where z_column is not None
    elif str(t_column)=='count':   
        
        print(x_column,y_column,z_column,t_column)
        print(yf_column,zf_column)
        print(yf_column is None)
        
        if yf_column=="Avg" and zf_column == "Avg":
            
            data_for_plot = df_temp.groupby([x_column]).agg(
                avg_y_column=('{}'.format(y_column), 'mean'),
                avg_z_column=('{}'.format(z_column), 'mean'),
                count=('{}'.format(y_column), 'size')
            ).reset_index()
                        
            # Renaming the columns
            data_for_plot.rename(columns={
                'avg_y_column': 'avg_' + y_column,
                'avg_z_column': 'avg_' + z_column
            }, inplace=True)
           
            
        elif yf_column=="Avg" and zf_column == "Weight on y":

            data_for_plot = df_temp.groupby([x_column]).agg(
                avg_y_column=('{}'.format(y_column), 'mean'),
                sum_z_column=('{}'.format(z_column), 'sum'),
                count=('{}'.format(y_column), 'size')
            ).reset_index()

            # We'll use sum_numVotes as the number of observations for each startYear
            data_for_plot['sum_z_column'] = data_for_plot['avg_y_column'] / np.sqrt(data_for_plot['sum_z_column'])
                        
            # Renaming the columns
            data_for_plot.rename(columns={
                'avg_y_column': 'avg_' + y_column,
                'sum_z_column': 'standard_error',
            }, inplace=True)  
            
        
        elif yf_column is None and zf_column == "Avg":
            
            data_for_plot = df_temp.groupby([x_column, y_column]).agg(
                avg_z_column=('{}'.format(z_column), 'mean'),
                count=('{}'.format(y_column), 'size')
            ).reset_index()

            # Renaming the columns
            data_for_plot.rename(columns={
                'avg_z_column': 'avg_' + z_column,
            }, inplace=True) 

        else:
            data_for_plot = df_temp.groupby([x_column, y_column, z_column]).size().reset_index(name='count')


    elif str(t_column)=='No count':   

        data_for_plot = df_temp[[x_column, y_column, z_column]]

    

    if x_column in df_col_string:
        data_for_plot = data_for_plot.sort_values(by='count', ascending=False)

    # # Remove rows with any NaN values
    # data_for_plot = data_for_plot.dropna()
    
    print("Para=",Para)
    
    return Para, data_for_plot, x_column, y_column, z_column, t_column


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def delete_rows_unknow_and_split(df_temp, x_column, y_column, z_column, t_column, Large_file_memory):

    """
    Goal: Delete the rows in a dataframe which correspond to '\\N'.

    Parameters:
    - df_temp: dataframe which has been created temporary.
    - x_column: Column in the dataframe.
    - y_column: Column in the dataframe (can be None).
    - z_column: Column in the dataframe (can be None).
    - t_column: Column in the dataframe (can be None).
    - Large_file_memory: Estimate if the file is too large to be open with panda and use dask instead.
    
    Returns:
    - Para: List of column in the dataframe (can be different of [x_column,y_column]).
    - df_temp: dataframe which has been created temporary.
    - x_column: Column in the dataframe (it could have change)
    - y_column: Column in the dataframe (it could have change)
    - z_column: Column in the dataframe (it could have change)
    - t_column: Column in the dataframe (it could have change)
    """
    
    
    df_col_numeric = df_temp.select_dtypes(include=['float64', 'int64']).columns.tolist()
    df_col_all = df_temp.columns.tolist()
    df_col_string = [col for col in df_col_all if col not in df_col_numeric]   

    print("df_col_numeric=",df_col_numeric)
    print("df_col_string=",df_col_string)
    
    if str(x_column) in df_col_string:
        df_temp[x_column] = df_temp[x_column].replace('', 'Unknown').astype(str)
    if str(y_column) in df_col_string:
        df_temp[y_column] = df_temp[y_column].replace('', 'Unknown').astype(str)
    if str(z_column) in df_col_string:
        df_temp[z_column] = df_temp[z_column].replace('', 'Unknown').astype(str)        
        
    if Large_file_memory==True:
        #Convert the Dask DataFrame to a Pandas DataFrame
        df_temp = df_temp.compute()
    
    if x_column in df_col_string:
        #To count individual elements when multiple elements are stored in a single cell 
        df_temp, element_counts = fd.explode_dataframe(df_temp, x_column)
        x_column = x_column+'_split'

    if y_column in df_col_string:
        #To count individual elements when multiple elements are stored in a single cell 
        df_temp, element_counts = fd.explode_dataframe(df_temp, y_column)
        y_column = y_column+'_split'

    if z_column in df_col_string:
        #To count individual elements when multiple elements are stored in a single cell 
        df_temp, element_counts = fd.explode_dataframe(df_temp, z_column)
        z_column = z_column+'_split'

    #Case where y_column is None
    if str(y_column)=='count' and str(z_column)=='None':    
        df_temp = df_temp[[x_column]]
        Para=[x_column, y_column]

    elif str(x_column)=='count' and str(y_column)!='None' and str(z_column)=='None':    
        df_temp = df_temp[[y_column]]
        Para=[x_column, y_column]
        

    elif str(x_column)=='count' and str(z_column)!='None' and str(t_column)=='None':
        df_temp = df_temp[[y_column, z_column]]
        Para=[x_column, y_column, z_column]    

    elif str(y_column)=='count' and str(z_column)!='None' and str(t_column)=='None':
        df_temp = df_temp[[x_column, z_column]]
        Para=[x_column, y_column, z_column]

    elif str(z_column)=='count' and str(t_column)=='None':
        df_temp = df_temp[[x_column, y_column]]
        Para=[x_column, y_column, z_column]

    elif str(z_column)=='No count' and str(t_column)=='None':
        df_temp = df_temp[[x_column, y_column]]
        Para=[x_column, y_column, z_column]
        

    elif str(x_column)=='count' and str(z_column)!='None' and str(t_column)!='None':
        df_temp = df_temp[[y_column, z_column, t_column]]
        Para=[x_column, y_column, z_column, t_column]    

    elif str(y_column)=='count' and str(z_column)!='None' and str(t_column)!='None':
        df_temp = df_temp[[x_column, z_column, t_column]]
        Para=[x_column, y_column, z_column, t_column]

    elif str(z_column)=='count' and str(t_column)!='None':
        df_temp = df_temp[[x_column, y_column, t_column]]
        Para=[x_column, y_column, z_column, t_column]
        
    elif str(t_column)=='count':
        df_temp = df_temp[[x_column, y_column, z_column]]
        Para=[x_column, y_column, z_column, t_column]        

    elif str(t_column)=='No count':
        df_temp = df_temp[[x_column, y_column, z_column]]
        Para=[x_column, y_column, z_column, t_column]  
    
    print("Para=",Para)
    
    return Para, df_temp, x_column, y_column, z_column, t_column


"""#=============================================================================
   #=============================================================================
   #============================================================================="""
