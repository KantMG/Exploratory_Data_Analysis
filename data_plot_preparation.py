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


import Function_dataframe as fd
import pandas as pd
import numpy as np

"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def data_preparation_for_plot(df_temp, df_col_string, x_column, y_column, z_column, yf_column, zf_column, g_column, Large_file_memory):

    """
    Goal: Get the pivot of the Count table of the dataframe.
    From a table of dimension x with n indexes to a table of dimension x+1 with n-1 index

    Parameters:
    - df_temp: dataframe which has been created temporary
    - df_col_string: List of columns in the DataFrame that are of object type.
    - x_column: Column in the dataframe
    - y_column: Column in the dataframe (can be None)
    - z_column: Column in the dataframe (can be None)
    - yf_column: Function to operate on y_column with the rest of the dataframe
    - zf_column: Function to operate on z_column with the rest of the dataframe
    - g_column: Type of Graphyque for the figure.
    - Large_file_memory: Estimate if the file is too large to be open with panda and use dask instead.

    Returns:
    - Para: List of column in the dataframe (can be different of [x_column,y_column])
    - data_for_plot: Data to plot.
    - x_column: Column in the dataframe (it could have change)
    - y_column: Column in the dataframe (it could have change)
    - z_column: Column in the dataframe (it could have change)
    """
        
    df_col_string = [col + '_split' for col in df_col_string]
    
    # print("Delete the rows with unknown value and split the column with multiple value per cell.")
    Para, df_temp, x_column, y_column, z_column = delete_rows_unknow_and_split(df_temp, x_column, y_column, z_column, Large_file_memory)

    if yf_column == "Value in x_y interval":
        data_for_plot, x_column, y_column, z_column = count_value_x_y_interval(df_temp, x_column, y_column, z_column)
        return Para, data_for_plot, x_column, y_column, z_column
    
    if x_column not in df_col_string:
        df_temp = df_temp[df_temp[x_column] >= 0]
    if str(y_column)!='None':
        if y_column not in df_col_string:
            df_temp = df_temp[df_temp[y_column] >= 0]   
    if str(z_column)!='None':
        if z_column not in df_col_string:
            df_temp = df_temp[df_temp[z_column] >= 0]        

    #Case where y_column is None
    if str(y_column)=='None':
        # Get the Count table of the dataframe  
        data_for_plot=df_temp.value_counts(dropna=False).reset_index(name='count') #dropna=False to count nan value
        # sort the data in function of column Para_sorted
        data_for_plot = data_for_plot.sort_values(by=Para[0], ascending=True)
        
    #Case where y_column is not None and z_column is None
    elif str(y_column)!='None' and str(z_column)=='None':   
        
        if yf_column == "Avg" or yf_column == "Avg on the ordinate":
            data_for_plot = df_temp.groupby([x_column]).agg(
                avg_y_column=('{}'.format(y_column), 'mean'),
                count=('{}'.format(y_column), 'size')
            ).reset_index()
            avg_col_name = 'avg_' + y_column
            data_for_plot.rename(columns={'avg_y_column': avg_col_name}, inplace=True)        
        else:
            data_for_plot = df_temp.groupby([x_column, y_column]).size().reset_index(name='count')
            
    #Case where z_column is not None
    else:
        # Calculate average z_column and count for each (x_column, y_column) combination
        if not yf_column and "Avg" in zf_column:
                        
            data_for_plot = df_temp.groupby([x_column, y_column]).agg(
                avg_z_column=('{}'.format(z_column), 'mean'),
                count=('{}'.format(z_column), 'size')
            ).reset_index()
            avg_col_name = 'avg_' + z_column
            data_for_plot.rename(columns={'avg_z_column': avg_col_name}, inplace=True)

        elif "Avg" in yf_column and zf_column == "Avg":
            
            data_for_plot = df_temp.groupby([x_column]).agg(
                avg_y_column=('{}'.format(y_column), 'mean'),
                avg_z_column=('{}'.format(z_column), 'mean'),
                count=('{}'.format(z_column), 'size')
            ).reset_index()
                        
            # Renaming the columns
            data_for_plot.rename(columns={
                'avg_y_column': 'avg_' + y_column,
                'avg_z_column': 'avg_' + z_column
            }, inplace=True)
           
        elif "Avg" in yf_column and zf_column == "Weight on y":

            data_for_plot = df_temp.groupby([x_column]).agg(
                avg_y_column=('{}'.format(y_column), 'mean'),
                sum_z_column=('{}'.format(z_column), 'sum'),
                count=('{}'.format(z_column), 'size')
            ).reset_index()

            # We'll use sum_numVotes as the number of observations for each startYear
            data_for_plot['sum_z_column'] = data_for_plot['avg_y_column'] / np.sqrt(data_for_plot['sum_z_column'])
                        
            # Renaming the columns
            data_for_plot.rename(columns={
                'avg_y_column': 'avg_' + y_column,
                'sum_z_column': 'standard_error',
            }, inplace=True)        

    if x_column in df_col_string:
        data_for_plot = data_for_plot.sort_values(by='count', ascending=False)

    # Remove rows with any NaN values
    data_for_plot = data_for_plot.dropna()
    
    return Para, data_for_plot, x_column, y_column, z_column


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def delete_rows_unknow_and_split(df_temp, x_column, y_column, z_column, Large_file_memory):

    """
    Goal: Delete the rows in a dataframe which correspond to '\\N'.

    Parameters:
    - df_temp: dataframe which has been created temporary.
    - x_column: Column in the dataframe.
    - y_column: Column in the dataframe (can be None).
    - z_column: Column in the dataframe (can be None).
    - Large_file_memory: Estimate if the file is too large to be open with panda and use dask instead.
    
    Returns:
    - Para: List of column in the dataframe (can be different of [x_column,y_column]).
    - df_temp: dataframe which has been created temporary.
    - x_column: Column in the dataframe (it could have change)
    - y_column: Column in the dataframe (it could have change)
    - z_column: Column in the dataframe (it could have change)
    """
    
    
    df_col_numeric = df_temp.select_dtypes(include=['float64', 'int64']).columns.tolist()
    df_col_all = df_temp.columns.tolist()
    df_col_string = [col for col in df_col_all if col not in df_col_numeric]   

    # if str(x_column) in df_col_numeric:
    #     df_temp[x_column] = df_temp[x_column].replace('', '0').fillna('0').astype(int)
    # if str(y_column) in df_col_numeric:
    #     df_temp[y_column] = df_temp[y_column].replace('', '0').fillna('0').astype(int)
    # if str(z_column) in df_col_numeric:
    #     df_temp[z_column] = df_temp[z_column].replace('', '0').fillna('0').astype(int)
        
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
    if str(y_column)=='None':    
        df_temp = df_temp[[x_column]]
        Para=[x_column]
    elif str(y_column)!='None' and str(z_column)=='None':
        df_temp = df_temp[[x_column, y_column]]
        Para=[x_column, y_column]
    else:
        df_temp = df_temp[[x_column, y_column, z_column]]
        Para=[x_column, y_column, z_column]        

    return Para, df_temp, x_column, y_column, z_column


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def count_value_x_y_interval(df, x_column, y_column, z_column):
    
    """
    Goal: Analyze a dataset to count entities based on specified criteria over a defined range and calculate the average of a given attribute. 
    The function processes temporary data to provide updated metrics and insights.

    Parameters:
    - df_temp: dataframe which has been created temporary.
    - x_column: Column in the dataframe.
    - y_column: Column in the dataframe (can be None).
    - z_column: Column in the dataframe (can be None).
    
    Returns:
    - df_temp: dataframe which has been updated.
    - x_column: Column in the dataframe (it could have change)
    - y_column: Column in the dataframe (it could have change)
    - z_column: Column in the dataframe (it could have change)
    """    
    
    
    # Determine the range for years based on birthYear and deathYear
    min_x_column = int(df[x_column].min())
    max_y_column = int(df[y_column].max())
    New_x_column = range(min_x_column, max_y_column + 1)
    
    # Initialize arrays to store results
    alive_counts = []
    avg_new_z_column = []

    # Create a DataFrame for all years
    for new_x_value in New_x_column:
        # Count alive directors: those whose x_column is less than or equal to the new_x_value
        # and either y_column is greater than or equal to the new_x_value OR y_column is -1
        alive_condition = (df[x_column] <= new_x_value) & ((df[y_column] >= new_x_value) | (df[y_column] == -1))
        count_alive = alive_condition.sum()  # Count how many are alive
        alive_counts.append({'Year': new_x_value, 'count': count_alive})

        if z_column is not None:
            # Calculate average z_column for those in count
            avg_rating = df.loc[alive_condition, z_column].mean()
            avg_new_z_column.append(avg_rating)
        else:
            avg_new_z_column.append(0)  # Append 0 if z_column is None

    # Create the resulting DataFrame
    alive_df = pd.DataFrame(alive_counts)
    if z_column is not None:
        alive_df['avg_' + z_column] = avg_new_z_column  # Add average NewRating

    # Count NaN values in x_column and count -1 in y_column in a vectorized manner
    nan_birth = df[x_column].isna().sum()
    nan_death = (df[y_column] == -1).sum()
    
    # Count alive directors with y_column as -1 and valid x_column
    alive_count_na = ((df[y_column] == -1) & (df[x_column].notna())).sum()

    # Print results for diagnostics
    print("nan_birth= ", nan_birth)
    print("nan_death= ", nan_death)
    print("alive_count_na= ", alive_count_na)

    return alive_df, 'Year', 'count', z_column


"""#=============================================================================
   #=============================================================================
   #============================================================================="""

