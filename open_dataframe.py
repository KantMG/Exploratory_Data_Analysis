#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 16:06:54 2024

@author: quentin
"""


"""#=============================================================================
   #=============================================================================
   #=============================================================================

    Dictionnary of functions to open/merge the dataframes

#=============================================================================
   #=============================================================================
   #============================================================================="""

import time
import pandas as pd
import dask.dataframe as dd
import re
import os
from termcolor import colored

"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def read_and_rename(filepath, usecols=None, dtype_mapping=None, rename_map=None, large_file=True):
    
    """
    Goal: 
    - Read and rename the DataFrame.
    
    Parameters:
    - filepath: Path of the tsv file.
    - usecols: List of columns present in the DataFrame.
    - dtype_mapping: Mapping for the dtype of the columns in the dataframe.
    - rename_map: List of columns to rename.
    - large_file: Estimate if the file is too large to be open with panda and use dask instead.
    
    Returns:
    - df: DataFrame
    """    
    
    if large_file:
        df = dd.read_csv(
            filepath,
            # sep='\t',
            usecols=usecols,
            encoding='utf-8',
            na_values=['\\N'],
            on_bad_lines='skip',
            quotechar='"',
            dtype=dtype_mapping
        )
    else:
        df = pd.read_csv(
            filepath,
            # sep='\t',
            usecols=usecols,
            encoding='utf-8',
            on_bad_lines='skip',
            quotechar='"'
        )

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def update_dataframe(df, col, val, n_val):
    
    """
    Goal: 
    - Update the DataFrame.
    
    Parameters:
    - df: DataFrame to update.
    - col: List of columns to update in the DataFrame.
    - val: The value which already exist in the dataframe.
    - n_val: The value to add in the dataframe in each cell which doesn't contain val.
    
    Returns:
    - df: DataFrame updated.
    """        

    # Iterate through each specified column
    for column in col:
        # Check if the column exists in the DataFrame
        if column in df.columns:
            # Update cells based on specified conditions
            df[column] = df[column].apply(lambda x: 
                x + ','+n_val if isinstance(x, str) and x and val not in x else x)
    
    return df


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def update_dataframe_remove_element_from_cell(df, col, val):
    
    """
    Goal: 
    - Update the DataFrame.
    
    Parameters:
    - df: DataFrame to update.
    - col: List of columns to update in the DataFrame.
    - val: The value which already exist in the dataframe.
    - n_val: The value to add in the dataframe in each cell which doesn't contain val.
    
    Returns:
    - df: DataFrame updated.
    """        

    # Iterate through each specified column
    for column in col:
        # Check if the column exists in the DataFrame
        if column in df.columns:
            # Update cells based on specified conditions
            df[column] = df[column].apply(lambda x: 
                ', '.join([item.strip() for item in x.split(',') if item.strip() != val]) 
                if isinstance(x, str) and x else x)
    
    return df


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def apply_filter(df, filters):
    
    """
    Goal: 
    - Apply the given filters to the DataFrame.
    
    Parameters:
    - df: DataFrame to filter
    - filters: Dictionary where keys are columns and values are filter conditions
    
    Returns:
    - df: Filtered DataFrame
    """    

    print("Apply filter.")
    
    print(df)
    
    if not filters:
        return df
    
    for col, filter_value in filters.items():
        
        if filter_value is not None:
            print("Apply on the column ", col, "the filter", filter_value)
        
        if filter_value is None or filter_value == 'All':
            continue  # Skip if filter_value is None or 'All'
        
        if isinstance(filter_value, bool):
            df = df[df[col] == filter_value]
        
        elif isinstance(filter_value, list):
            # Use regex pattern to match any of the values in the list
            pattern = '|'.join(map(re.escape, filter_value))
            df = df[df[col].str.contains(pattern, na=False)]

        else:
            # Check for interval filtering like "<10" or "<=10" and ">10" or ">=10"
            if '>=' in filter_value or '>' in filter_value or '<=' in filter_value or '<' in filter_value:
                if '>=' in filter_value:
                    threshold = float(filter_value.split('>=')[1])
                    df = df[df[col] >= threshold]
                elif '>' in filter_value:
                    threshold = float(filter_value.split('>')[1])
                    df = df[df[col] > threshold]
                elif '<=' in filter_value:
                    threshold = float(filter_value.split('<=')[1])
                    df = df[df[col] <= threshold]
                elif '<' in filter_value:
                    threshold = float(filter_value.split('<')[1])
                    df = df[df[col] < threshold]
            elif '!=' in filter_value:
                threshold = float(filter_value.split('!=')[1])
                df = df[df[col] != threshold]  # Apply not equal condition
            elif '=' in filter_value:
                threshold = float(filter_value.split('=')[1])
                df = df[df[col] == threshold]
            elif '-' in filter_value:
                bounds = filter_value.split('-')
                lower_bound = float(bounds[0])
                upper_bound = float(bounds[1]) if len(bounds) > 1 else None
                if upper_bound is not None:
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                else:
                    df = df[df[col] >= lower_bound]

            elif filter_value.endswith('*'):
                exact_value = filter_value[:-1]
                df = df[df[col] == exact_value]
                
            else:
                df = df[df[col] == filter_value]
                
            print(f"Filtered df for {col}:")
            print(f"{df[col]}")
    
        if df.empty:
            print("Filtered DataFrame is empty. Returning empty DataFrame.")

    return df
    

"""#=============================================================================
   #=============================================================================
   #============================================================================="""


