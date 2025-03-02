#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 18:27:38 2024

@author: quentin
"""



"""#=============================================================================
   #=============================================================================
   #=============================================================================

    Dictionnary of functions to generate dataframes

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


def create_data_specific(df1, df1_col_groupby, df_name, exclude_col):

    """
    Goal: Create a new dataset by combining two existing datasets and rearranging their configuration.
    
    Parameters:
    - df1: The first dataset.
    - df1_col_groupby: The column in the first dataset used for grouping other columns.
    - df_name: The second dataset.
    - exclude_col: A list of columns to be excluded from the second dataset.
    
    Returns:
    - df_test: A new DataFrame.
    """   

    df1 = df1.copy()
    
    df1 = df1[df1[df1_col_groupby].notnull() & (df1[df1_col_groupby] != '')]
    
    # Step 1: Split `directors` into multiple rows
    df1[df1_col_groupby] = df1[df1_col_groupby].str.split(',')
    df1_exploded = df1.explode(df1_col_groupby)

    # Step 2: Merge to get birthYear and deathYear
    merged_df = df1_exploded.merge(df_name, left_on=df1_col_groupby, right_on='nconst', how='left')
    
    exclude_col = ["primaryName"]
    merged_df = merged_df.drop(columns=exclude_col)

    print(df_name["birthYear"].isna().sum())
    print(df_name["deathYear"].isna().sum())
    
    # Step 3: Group by 'directors'
    group_columns = merged_df.columns.difference([df1_col_groupby, 'birthYear', 'deathYear', 'nconst'])
    final_df = merged_df.groupby(df1_col_groupby).agg({
        **{col: lambda x: ','.join(map(str, x)) for col in group_columns},  # Dynamically concatenate other columns
        'birthYear': 'first',  # Take the first birthYear
        'deathYear': 'first',   # Take the first deathYear
    }).reset_index()
        
    # df_test = final_df.loc[final_df['directors'] == 'nm0005690']
    
    df_test = final_df    

    # Apply the calculation to each row
    df_test['NewRating'] = df_test.apply(lambda row: calculate_weighted_average(row['averageRating'], row['numVotes']), axis=1)
    
    # Step 1: Replace `averageRating`, `numVotes`, `runtimeMinutes`, `isAdult` with their averages
    df_test.loc[:, 'averageRating'] = df_test['averageRating'].apply(average_of_string_values)
    df_test.loc[:, 'numVotes'] = df_test['numVotes'].apply(average_of_string_values)
    df_test.loc[:, 'runtimeMinutes'] = df_test['runtimeMinutes'].apply(average_of_string_values)
    df_test.loc[:, 'isAdult'] = df_test['isAdult'].apply(average_of_string_values)
    
    df_test.loc[:, 'numGenres'] = df_test['genres'].apply(lambda x: len(set([genre for genre in x.split(',') if genre])))
    df_test.loc[:, 'numtitleType'] = df_test['titleType'].apply(lambda x: len(set([genre for genre in x.split(',') if genre])))
    
    # Calculate first year (minimum) and last year (maximum) from startYear
    df_test.loc[:, 'firstYear'] = df_test['startYear'].apply(
        lambda x: min(
            (float(v) for v in x.split(',') if v not in ['nan', '', '\\N']),
            default=None  # Use default to avoid min() with empty sequence
        ) if x else None
    )
    
    df_test.loc[:, 'lastYear'] = df_test['startYear'].apply(
        lambda x: max(
            (float(v) for v in x.split(',') if v not in ['nan', '', '\\N']),
            default=None  # Use default to avoid max() with empty sequence
        ) if x else None
    )

    # Step 4: Replace tconst by the number of values in tconst
    # df_test.loc[:, 'tconst'] = df_test['tconst'].apply(lambda x: len(x.split(',')))
    df_test['numProductions'] = df_test['tconst'].apply(lambda x: len(x.split(',')))
    
    # Step 5: Drop startYear
    df_test.drop('genres', axis=1, inplace=True)
    df_test.drop('titleType', axis=1, inplace=True)
    df_test.drop('startYear', axis=1, inplace=True)
    df_test.drop('tconst', axis=1, inplace=True)


    # Convert specific columns to float64
    float_columns = ['averageRating', 'numVotes', 'runtimeMinutes', 'isAdult',
                     'NewRating', 'numGenres', 'numtitleType', 'firstYear', 
                     'lastYear', 'numProductions']
    for col in float_columns:
        df_test[col] = pd.to_numeric(df_test[col], errors='coerce')  # Ensure conversion to float
    
    
    print(df_test[[df1_col_groupby, "birthYear", "deathYear"]])
    print(df_test["birthYear"].isna().sum())
    print(df_test["deathYear"].isna().sum())
    
    print(df_test.dtypes)
        
    return df_test, df_test.columns.tolist()
    

"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def average_of_string_values(s):

    """
    Goal: Function to average numerical values in a string of comma-separated numbers.
    
    Parameters:
    - s: String values.
    
    Returns:
    - The average numerical value of s.
    """   

    # Split the string and filter out 'nan', empty values, '\\N', and non-convertible strings
    values = []
    for x in s.split(','):
        if x not in ['nan', '', '\\N']:  # Filter out 'nan', empty strings, and '\\N'
            try:
                values.append(float(x))  # Attempt to convert to float
            except ValueError:
                continue  # Skip non-convertible values

    if values:  # Check if there are any valid values
        return sum(values) / len(values)  # Calculate and return average
    else:
        return 0  # Return 0 if no valid values


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def calculate_weighted_average(ratings_str, votes_str):

    """
    Goal: Function to calculate the weighted average of numerical values in a comma-separated string. 
    Weights are provided by a second comma-separated string.
    
    Parameters:
    - ratings_str: String values.
    - votes_str: String values.
    
    Returns:
    - The weighted average of the ratings in ratings_str, using the values in votes_str as weights.
    """   

    # Convert the comma-separated strings into lists of floats
    # Filter out 'nan', empty values, and '\\N' in both ratings and votes
    ratings = []
    votes = []

    for x in ratings_str.split(','):
        if x not in ['nan', '', '\\N']:  # Filter out 'nan', empty strings, and '\\N'
            try:
                ratings.append(float(x))  # Attempt to convert to float
            except ValueError:
                continue  # Skip non-convertible values

    for x in votes_str.split(','):
        if x not in ['nan', '', '\\N']:  # Filter out 'nan', empty strings, and '\\N'
            try:
                votes.append(float(x))  # Attempt to convert to float
            except ValueError:
                continue  # Skip non-convertible values

    # Ensuring that both ratings and votes lists are non-empty and have the same length
    if len(ratings) != len(votes) or not votes:
        return 0  # Return 0 if there are mismatched lengths or no valid votes

    total_weighted_sum = sum(rating * vote for rating, vote in zip(ratings, votes))
    total_votes = sum(votes)

    return total_weighted_sum / total_votes if total_votes > 0 else 0


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def test_data_creation(Project_path, test_directory, Files=None, Rows_to_keep=None, Large_file_memory=True):

    """
    Goal: Generate a small test dataset from an existing DataFrame.
    
    Parameters:
    - Project_path: Path to the original datasets.
    - test_directory: Name of the new test dataset directory.
    - Files: List of files to recreate in the test version.
    - Rows_to_keep: Number of rows to retain in the test dataset.
    - Large_file_memory: Indicator for using Dask if the file is too large for Pandas.
    
    Returns:
    - New test files are created.
    """       

    # Check if the directory does not exist and create it
    if not os.path.exists(Project_path+test_directory):
        os.makedirs(Project_path+test_directory)
        print(f"Directory '{test_directory}' created.")
    else:
        print(f"Directory '{test_directory}' already exists.")
    
    file_mapping = file_columns_dtype()
        
    for data in Files:

        # Fetch the appropriate column and dtype mapping for each file
        columns_info = file_mapping.get(data, {})
        usecols = columns_info.get("columns")
        dtype_mapping = columns_info.get("types")
        rename_map = columns_info.get("rename", None)
       
        print(data)
        #Create class 'pandas.core.frame.DataFrame'
        df = read_and_rename(
            Project_path + data,
            usecols=usecols,
            dtype_mapping=dtype_mapping,
            rename_map=rename_map,
            large_file=Large_file_memory,
        )
        print(df)
        print()

        # Replace specific problematic strings with NaN
        for column, dtype in dtype_mapping.items():
            if dtype == 'float64':  # Only apply to columns expecting float
                # Example: replacing non-convertible 'Talk-Show' with NaN
                df[column] = pd.to_numeric(df[column], errors='coerce')

        
        # Keep only the first "Rows_to_keep" rows
        # df_cut = df.head(Rows_to_keep)
        df_cut = df.tail(Rows_to_keep)
        print(df_cut)
        print()   
        
        # Save the new data set into the Test_data directory
        # df_cut.to_csv(Project_path+'Test_data/'+data, index=False, sep='\t', encoding='utf-8', quotechar='"')

        df_cut.to_csv(
            Project_path+test_directory+'/'+data,
            sep='\t',
            index=False,
            encoding='utf-8',
            quotechar='"'
        )