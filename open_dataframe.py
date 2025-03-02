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


def open_dataframe(Project_path, Large_file_memory, requested_columns, requested_filters):

    """
    Goal: 
    - Read and rename the DataFrame.
    
    Parameters:
    - Project_path: Path of the tsv file.
    - Large_file_memory: Estimate if the file is too large to be open with panda and use dask instead.
    - requested_columns: List of columns to extract from the DataFrame located in several file.
    - requested_filters: List of filter to apply on each column.
    
    Returns:
    - df: DataFrame
    """
    
    print()
    print(colored("**************** Open dataframe ****************", "yellow"))
    print()
    
    start_time = time.time()  
        
    # Define the mapping of files to their columns and their types
    file_columns_mapping_dtype = file_columns_dtype()
    file_columns_mapping = {k: v for k, v in file_columns_mapping_dtype.items() if k != 'name.basics.tsv'}
    
    # Create a dictionary to map each requested column to its filter
    column_filter_mapping = dict(zip(requested_columns, requested_filters))
    
    # Determine which files need to be opened
    files_to_open = []
    columns_in_files = {}
    
    # Iterate through each file and check if it has any of the requested columns
    for file, info in file_columns_mapping.items():
        columns = info["columns"]
        types = info["types"]
        
        # Find the intersection of requested columns and columns in the current file
        common_columns = set(requested_columns).intersection(columns)
        if common_columns:  # Only consider files that have at least one requested column
            files_to_open.append(file)

            # Handle renaming if it's the 'title.akas.tsv' file
            if file == 'title.akas.tsv' and "titleId" in common_columns:
                common_columns.discard("titleId")  # Remove 'titleId'
                common_columns.add("tconst")  # Add 'tconst' instead

            # Track the columns that should be used from this file, always including 'tconst' if present
            if "tconst" in columns or (file == 'title.akas.tsv' and "titleId" in columns):
                common_columns.add("tconst")

            columns_in_files[file] = {
                "columns": common_columns,
                "types": {("tconst" if col == "titleId" else col): types[col] for col in common_columns if col in types},
                "filters": {("tconst" if col == "titleId" else col): column_filter_mapping[col] for col in common_columns if col in column_filter_mapping},
                "rename": info.get("rename", {})
            }
    
    # Identify common columns between files to be opened
    if len(files_to_open) > 1:
        # Find common columns among all files using set intersection
        common_columns_all_files = set.intersection(*(columns_in_files[file]["columns"] for file in files_to_open))
    else:
        common_columns_all_files = set(columns_in_files[files_to_open[0]]["columns"])
    
    # Ensure 'tconst' is added as a common column if at least two files are being opened and 'tconst' exists in those files
    tconst_in_files = all("tconst" in file_columns_mapping[file]["columns"] for file in files_to_open)
    if len(files_to_open) > 1 and tconst_in_files:
        common_columns_all_files.add("tconst")
    
    print("Files to open:", files_to_open)
    print("Common columns across all selected files:", common_columns_all_files)
    
    print("Columns, filters, and types in each selected file:")
    for file, info in columns_in_files.items():
        print(f"{file}:")
        print("  Columns:", info["columns"])
        print("  Filters:", info["filters"])
        print("  Types:", info["types"])

    # Create DataFrames based on the files, columns, and filters
    dataframes = []
    for file, info in columns_in_files.items():
        # Define the columns to read from the file
        usecols = list(info["columns"])
        # Ensure 'titleId' is used instead of 'tconst' in the akas file
        if 'tconst' in usecols and file == 'title.akas.tsv':
            usecols.remove('tconst')
            usecols.append('titleId')

        # Create a dictionary to define the dtypes for the DataFrame
        dtype_mapping = {col: info["types"][col] for col in usecols if col in info["types"]}       
                
        # Read the file into a DataFrame
        filepath = f"{Project_path}/{file}"
        # Log the time taken for each file reading
        file_start_time = time.time()
        df = read_and_rename(
            filepath,
            usecols,
            dtype_mapping,
            rename_map=info.get("rename"),
            large_file=Large_file_memory
        )
       # Convert columns to the specified types
        for col, expected_type in info["types"].items():
            print(col, expected_type)
            if expected_type in [float, "float64"]:
                if Large_file_memory:
                    df[col] = dd.to_numeric(df[col], errors='coerce')
                    # Handle NA values
                    # df[col] = df[col].fillna(-1)  # Fill with -1 or another value as necessary  
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Handle NA values
                    # df[col] = df[col].fillna(-1)  # Fill with -1 or another value as necessary     
                   
            if expected_type == str:
                df[col] = df[col].fillna('')  # Fill NaN with empty string for string columns
        
        # Get the infos on the DataFrame
        dis.infos_on_data(df) if Get_file_sys_mem==True else None
                
        # # Log the time taken to apply filters
        log_performance(f"Read {file}", file_start_time)
        print(df)
        
        # Add the DataFrame to the list
        dataframes.append(df)
            
    print("Time taken to load all dataframe: {:.2f} seconds".format(time.time() - start_time))        
    print()

    # Log the time taken to merge DataFrames
    merge_start_time = time.time()    
    # Merge Dask DataFrames on 'tconst' to create a single unified DataFrame
    if len(dataframes) > 1:
        i = 0
        if Large_file_memory:
            merged_df = dataframes[i].compute()
        else:
            merged_df = dataframes[i]    
        print()
        print("Time taken to compute dataframe "+str(i)+": {:.2f} seconds".format(time.time() - start_time))
        print()
        for df in dataframes[1:]:
            i+=1

            if Large_file_memory:        
                df = df.compute()
            
            if "category" in df.columns:
                # Group by 'tconst' and 'nconst' and join the 'category' values
                df = df.groupby(['tconst', 'nconst'], as_index=False).agg({
                    'category': ', '.join,  # Combine categories
                    'characters': 'first'   # Keep the first non-empty value from characters (if any)
                })
            
            if Large_file_memory:
                merged_df = dd.merge(merged_df, df, on='tconst', how='inner')
            else:
                if "parentTconst" not in df.columns:
                    merged_df = pd.merge(merged_df, df, on='tconst', how='outer')
                else:
                    merged_df = pd.merge(merged_df, df, on='tconst', how='outer')
                
            print("Time taken to merge dataframe "+str(i)+": {:.2f} seconds".format(time.time() - start_time))
            print()
        # merged_df = dd.from_pandas(merged_df, npartitions=2)
        
    else:
        merged_df = dataframes[0]
    log_performance("Merging DataFrames", merge_start_time)
    
    # Print the final merged DataFrame (head only, to avoid loading too much data)
    print("\nFinal Merged DataFrame:")
    print(merged_df)
    print()
    print("Time taken to merge all dataframe: {:.2f} seconds".format(time.time() - start_time))
    print()
    print(colored("*************** Dataframe is open **************", "yellow"))
    print()
    log_performance("Complete open_data", start_time)
    
    return merged_df


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


def open_data_name(requested_columns, requested_filters, Project_path, Large_file_memory, Get_file_sys_mem):

    """
    Goal: 
    - Read and rename the DataFrame.
    
    Parameters:
    - requested_columns: List of columns to extract from the DataFrame located in several file.
    - requested_filters: List of filter to apply on each column.
    - Project_path: Path of the tsv file.
    - Large_file_memory: Estimate if the file is too large to be open with panda and use dask instead.
    - Get_file_sys_mem: Estimate the memory consuming by the files.
    
    Returns:
    - df: DataFrame
    """
    
    # Define the mapping of files to their columns and their types
    file_columns_mapping = {
        'name.basics.tsv': {
            "columns": ["nconst", "primaryName", "birthYear", "deathYear", "primaryProfession", "knownForTitles"],
            "types": {
                "nconst": str,
                "primaryName": str,
                "birthYear": float,
                "deathYear": float,
                "primaryProfession": str,
                "knownForTitles": str
            }
        }
    }

    print()
    print(colored("**************** Open dataframe ****************", "yellow"))
    print()
    
    # Create a dictionary to map each requested column to its filter
    column_filter_mapping = dict(zip(requested_columns, requested_filters))
    
    # Determine which files need to be opened
    files_to_open = []
    columns_in_files = {}
    
    # Iterate through each file and check if it has any of the requested columns
    for file, info in file_columns_mapping.items():
        columns = info["columns"]
        types = info["types"]
        
        # Find the intersection of requested columns and columns in the current file
        common_columns = set(requested_columns).intersection(columns)
        if common_columns:  # Only consider files that have at least one requested column
            files_to_open.append(file)
            # Track the columns that should be used from this file, always including 'tconst' if present
            if "tconst" in columns:
                common_columns.add("tconst")
            columns_in_files[file] = {
                "columns": common_columns,
                "types": {col: types[col] for col in common_columns if col in types},  # Map each column to its type
                "filters": {col: column_filter_mapping[col] for col in common_columns if col in column_filter_mapping}
            }
    
    # Identify common columns between files to be opened
    if len(files_to_open) > 1:
        # Find common columns among all files using set intersection
        common_columns_all_files = set.intersection(*(columns_in_files[file]["columns"] for file in files_to_open))
    else:
        common_columns_all_files = set(columns_in_files[files_to_open[0]]["columns"])
    
    # Ensure 'tconst' is added as a common column if at least two files are being opened and 'tconst' exists in those files
    tconst_in_files = all("tconst" in file_columns_mapping[file]["columns"] for file in files_to_open)
    if len(files_to_open) > 1 and tconst_in_files:
        common_columns_all_files.add("tconst")
    
    # Print the results
    print("Files to open:", files_to_open)
    print("Common columns across all selected files:", common_columns_all_files)
    
    print("Columns, filters, and types in each selected file:")
    for file, info in columns_in_files.items():
        print(f"{file}:")
        print("  Columns:", info["columns"])
        print("  Filters:", info["filters"])
        print("  Types:", info["types"])

    # Create DataFrames based on the files, columns, and filters
    for file, info in columns_in_files.items():
        # Define the columns to read from the file
        usecols = list(info["columns"])

        # Create a dictionary to define the dtypes for the DataFrame
        dtype_mapping = {col: info["types"][col] for col in usecols if col in info["types"]}       
        
        file_start_time = time.time()    
        # Read the file into a DataFrame
        filepath = f"{Project_path}/{file}"
        # Log the time taken for each file reading
        df = read_and_rename(
            filepath,
            usecols,
            dtype_mapping,
            rename_map=None,
            large_file=Large_file_memory
        )
             
       # Convert columns to the specified types
        for col, expected_type in info["types"].items():
            if expected_type == float:
                if Large_file_memory:
                    df[col] = dd.to_numeric(df[col], errors='coerce')
                    # Handle NA values
                    df[col] = df[col].fillna(-1)  # Fill with -1 or another value as necessary                    
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Handle NA values
                    df[col] = df[col].fillna(-1)  # Fill with -1 or another value as necessary   
            
            elif expected_type == str:
                df[col] = df[col].fillna('')  # Fill NaN with empty string for string columns


        # df=df.repartition(npartitions=desired_number_of_partitions)
        
        # Get the infos on the DataFrame
        dis.infos_on_data(df) if Get_file_sys_mem==True else None

        # Apply the filter to the DataFrame
        df = df[df['birthYear'] != -1]
        
        log_performance(f"Reading {file}", file_start_time)    
        
    if Large_file_memory:
        df = df.compute()

    print()
    print(colored("*************** Dataframe is open **************", "yellow"))
    print()

    return df


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def open_made_dataframe(Project_path, df1_col_groupby, Large_file_memory, Get_file_sys_mem):

    if os.path.exists(Project_path+'Made_data/groupby_'+df1_col_groupby):
        df1 = read_and_rename(
            Project_path+'Made_data/groupby_'+df1_col_groupby,
            rename_map=None,
            large_file=Large_file_memory
        )
        List_col_tab2 = df1.columns
            
    else:
        selected_columns = ["startYear", "runtimeMinutes", "genres", "titleType", "isAdult", "averageRating", "numVotes", df1_col_groupby] #, "directors", "writers", "region", "language", "isOriginalTitle" , "parentTconst", "seasonNumber", "episodeNumber"
        selected_filter  = [None for i in selected_columns]
                
        df1 = open_dataframe(selected_columns, selected_filter, Project_path, Large_file_memory, Get_file_sys_mem)
        
        if "isOriginalTitle" in df1.columns:
            df1 = df1.loc[df1["isOriginalTitle"] == 1]
            df1.reset_index(drop=True, inplace=True)
        
        if "titleType" in df1.columns:
            exclude_type = ["tvEpisode", "video", "videoGame", "tvPilot", "tvSpecial"]
            df1 = df1[~df1["titleType"].isin(exclude_type)]
                
        # Remove the value "Short" in the "genres" column since it is a value in "titleType"
        df1 = update_dataframe_remove_element_from_cell(df1, ["genres"], "Short")
        
        # Lists of columns that are relevants regarding the tab where where we are.
        List_col_tab2 = ["startYear", "runtimeMinutes", "genres", "titleType", "isAdult", "averageRating", "numVotes"] #, "region", "language" , "parentTconst", "seasonNumber", "episodeNumber"
        List_col_exclude_tab2 = ["tconst"] #, "isOriginalTitle"

        List_col = ["nconst", "primaryName", "birthYear", "deathYear"]
        List_filter = [None, None, None, None]
        df_name = open_data_name(List_col, List_filter, Project_path, Large_file_memory, Get_file_sys_mem)

        df1, List_col_tab2 = dc.create_data_specific(df1, df1_col_groupby, df_name, ["primaryName"])
        # Check if the directory does not exist and create it
        if not os.path.exists(Project_path+'Made_data'):
            os.makedirs(Project_path+'Made_data')
            print(f"Directory Made_data created.")
        else:
            print(f"Directory Made_data already exists.")
        df1.to_csv(
            Project_path+'Made_data/groupby_'+df1_col_groupby,
            sep='\t',
            index=False,
            encoding='utf-8',
            quotechar='"'
        )
    
    # List_col_exclude_tab2 = [] #, "isOriginalTitle"
    # Function to convert 'directors' column to int
    def convert_directors(director):
        if director == '\\N':
            return None  # Handling the \N value explicitly
        elif director.startswith('nm'):
            return int(director[2:])  # Convert to int by slicing off 'nm'
        else:
            return int(director)  # In case it's a number
    
    df1[df1_col_groupby] = df1[df1_col_groupby].apply(convert_directors)
    print(df1)
    print(df1.dtypes)
    
    return df1, List_col_tab2
    

