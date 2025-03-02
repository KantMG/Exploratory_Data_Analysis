#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:51:53 2024

@author: quentin
"""


"""#=============================================================================
   #=============================================================================
   #=============================================================================

    Dictionnary of functions to know the memory dispatch.

#=============================================================================
   #=============================================================================
   #============================================================================="""


import os
import psutil


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def infos_on_sys():

    """
    Goal: 
    - Get the system memory repartition.
    
    Parameters:
    - None.
    
    Returns:
    - None
    """
        
    # Get the total available memory (RAM) in bytes
    total_memory = psutil.virtual_memory().total
    # Convert to MB or GB
    total_memory_mb = total_memory / (1024 * 1024)
    total_memory_gb = total_memory / (1024 * 1024 * 1024)
    print(f"Total memory: {total_memory_mb:.2f} MB ({total_memory_gb:.2f} GB)")


    # Get the number of physical cores
    physical_cores = psutil.cpu_count(logical=False)
    # Get the number of logical cores (threads)
    logical_cores = psutil.cpu_count(logical=True)
    print(f"Physical cores: {physical_cores}")
    print(f"Logical cores (threads): {logical_cores}")

    
"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def infos_on_data(df):

    """
    Goal: 
    - Get the space memory of the dataframe.
    
    Parameters:
    - df: dataframe.
    
    Returns:
    - None
    """
    
    df_size = df.memory_usage(deep=True).compute().sum()  # In bytes
    df_size_mb = df_size / (1024 * 1024)
    df_size_gb = df_size / (1024 * 1024 * 1024)
    print(f"Dask DataFrame size: {df_size_mb:.2f} MB ({df_size_gb:.2f} GB)")
    
    # Assuming df is your Dask DataFrame
    num_rows = df.shape[0].compute()
    # Now compute to get the actual values
    num_cols = len(df.columns)
    print(f"Number of rows: {num_rows}, Number of columns: {num_cols}")