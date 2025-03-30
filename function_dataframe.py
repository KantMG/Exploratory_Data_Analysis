#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 16:22:46 2024

@author: quentin
"""



"""#=============================================================================
   #=============================================================================
   #=============================================================================

    Dictionnary of functions on the dataframe

#=============================================================================
   #=============================================================================
   #============================================================================="""

import pandas as pd
import pylab as pl
import Levenshtein
import imageio
import os

"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def df_empty(columns, dtypes, index=None):
    
    """
    Goal: 
    - Create an empty dataframe.

    Parameters:
    - columns: List of column to create in the dataframe.
    - dtypes: List of type which corresponding to the columns list.

    Returns:
    - Dataframe which has been created.
    """
    assert len(columns)==len(dtypes)
    
    df = pd.DataFrame(index=index)
    
    for c,d in zip(columns, dtypes):
        df[c] = pd.Series(dtype=d)
        
    return df


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def explode_dataframe(df, Para):
    
    """
    Goal: 
    - Count individual elements when multiple elements are stored in a single cell
    - Explode the Dataframe where cells with muliple elements are counted multiple time.

    Parameters:
    - df: Dataframe
    - Para: List of column in the df for which the table should explode the cells with multiple elements.

    Returns:
    - Dataframe which have been explode and the new counts of each elements.
    
    Warning:
    - Can create very large array if many cells contain many elements.
    """
    
    df_temp = df.copy()
       
    # Step 1: Split the elements into lists of elements
    df_temp[Para+'_split'] = df_temp[Para].str.split(',')
    
    # Step 2: Explode the list of elements into individual rows
    df_temp = df_temp.explode(Para+'_split')
        
    # Step 3: Clean up the split elements by stripping whitespace
    df_temp[Para + '_split'] = df_temp[Para + '_split'].str.strip()    
    
    # Step 4: Replace empty cells with 'Unknown' 
    df_temp[Para + '_split'].replace({'': 'Unknown', r'\\N': 'Unknown'}, regex=True, inplace=True)

    # Step 5: Fill NaN values with 'Unknown'
    df_temp[Para + '_split'].fillna('Unknown', inplace=True)

    # Step 6: Count the occurrences of each element
    element_counts = df_temp[Para + '_split'].value_counts()   
    
    # Display the result
    print("Dataframe have been explode base on parameter "+Para)
    print("The new counts of each elements is:")
    print(element_counts)
    print()
    
    return df_temp, element_counts


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def reverse_explode_dataframe(df_exploded, Para):
    
    """
    Goal: 
    - Revert the exploded data

    Parameters:
    - df_exploded: Dataframe
    - Para: List of column in the df for which the table should explode the cells with multiple elements.

    Returns:
    - Dataframe which have been explode and the new counts of each elements.
    
    Warning:
    - 
    """

    # Group by the original ID and aggregate back to the original format
    df_reverted = df_exploded.groupby('tconst')[Para + '_split'].agg(lambda x: ', '.join(x.str.strip())).reset_index()
    
    # Rename the aggregated column back to the original name
    df_reverted.rename(columns={Para + '_split': Para}, inplace=True)
    
    # Display the reverted DataFrame
    print("Reverted DataFrame to original:")
    print(df_reverted)
    
    return df_reverted


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def Pivot_table(csvFile,Para,remove_unknown_colmun, Large_file_memory=False):
    
    """
    Goal: Get the pivot of the Count table of the dataframe.
    From a table of dimension x with n indexes to a table of dimension x+1 with n-1 index

    Parameters:
    - csvFile: dataframe
    - Para: List of column in the dataframe
    - remove_unknown_value: Bolean (True or False)
    - Large_file_memory: Estimate if the file is too large to be open with panda and use dask instead.

    Returns:
    - Dataframe which have been pivoted.
    """
    
    df = csvFile[Para]
        
    # Get the Count table of the dataframe  
    y=df.value_counts(dropna=False).reset_index(name='Count') #dropna=False to count nan value    
    
    # Pivot the Count table 
    pivot_table = y.pivot(index=Para[0], 
                          columns=Para[1] if len(Para) == 2 else (Para[1], Para[2]), 
                          values='Count').fillna(0)
        
    #Add last column for the sum of each rows named Total
    s = sum ( [pivot_table[i] for i in  pivot_table.columns])
    pivot_table2 = pivot_table.assign(Total=s).sort_values(by=['Total'], ascending=False)

    print("Dataframe of parameters "+' and '.join([str(i) for i in Para])+" have been pivoted.")
    print(pivot_table2)
    print()
    
    return pivot_table2

"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def avg_column_value_index(Pivot_table):
    
    """
    Goal: Creates in the table a new column which is th avg value of all the other column times the column name.

    Parameters:
    - Pivot_table: dataframe which have been pivoted.

    Returns:
    - Table y: new avg_col column of the dataframe 
    """        
    
    #Get the sum of each rows, where each column element is multiplied by the column's name
    # s = sum([Pivot_table[i] * int(i) for i in Pivot_table.columns if isinstance(i, str) and i.isdigit()])
    # s = Pivot_table.apply(lambda row: sum([row[i] * float(i) for i in Pivot_table.columns if isinstance(i, float)]), axis=1)
    s = Pivot_table.apply(lambda row: sum([row[i] * int(i) for i in Pivot_table.columns[:-1]]), axis=1)
    
    print(s)
    
    #Add avg_col as the last column of the dataframe and sort the dataframe
    pivot_table2 = Pivot_table.assign(avg_col=s).sort_values(by=['avg_col'], ascending=False)
    
    print(pivot_table2)
    
    #Correct the avg_col by dividing the values with the total value
    pivot_table2['avg_col']=pivot_table2['avg_col']/pivot_table2['Total']
    
    return pivot_table2['avg_col']   


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def group_small_values(data, col, count_column, n, col_ref=None):
    
    """
    Goal: Group the values which are the less present in the dataframe other the same name "Other".

    Parameters:
    - data: Dataframe.
    - col: Column in the dataframe that must be grouped.
    - count_column: Column in the dataframe (usally count) which will give the total amount of the Other.
    - n: Integer that will define which value of col are counted in the "Other" value. All values of col which are not in the n first count.
    - col_ref: Column in the dataframe that will be use as a reference to regroup the values of col.

    Returns:
    - The updated Dataframe.
    """
    
    # Group by col value and sum the count_column
    grouped_data = data.groupby(col)[count_column].sum().reset_index()
    
    # Get the top n col value based on summed of count_column
    top_n_genres = grouped_data.nlargest(n, count_column)
    
    # Extract the col value
    top_n = top_n_genres[col].unique()
    
    # Replace values not in top_n with "Other"
    data[col] = data[col].where(data[col].isin(top_n), 'Other')
    
    result = aggregate_value(data, col, count_column, col_ref)
        
    return result


"""#=============================================================================
   #=============================================================================
   #============================================================================="""

def make_movie(plotly_fig):

    """
    Goal: Create a movie from an animated Plotly figure.
    
    Parameters:
    - plotly_fig: The Plotly figure to be animated.
    
    Returns:
    - The generated movie.
    """    

    # Set the output path of your images
    image_paths = []
    
    # Loop through animation frames if they're defined
    for frame in plotly_fig.frames:
        print(frame)
        plotly_fig.update(frames=[frame])
        image_path = f"frame_{frame.name}.png"
        plotly_fig.write_image(image_path)
        image_paths.append(image_path)    


    # Create a video from images
    with imageio.get_writer('output_video.mp4', fps=10) as writer:
        for image_path in image_paths:
            image = imageio.imread(image_path)
            writer.append_data(image)
    
    print("Video created successfully: output_video.mp4")