#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 18:13:48 2024

@author: quentin
"""

"""#=============================================================================
   #=============================================================================
   #=============================================================================

    Dictionnary of functions for table dropdown creation.

#=============================================================================
   #=============================================================================
   #============================================================================="""


import dash
from dash import dcc, html, Input, Output, dash_table, callback, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
from collections import OrderedDict
import plotly.express as px
import webbrowser


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def get_max_width(col_data, col_name):

    """
    Goal: Calculate the associated dropdown for each table column.

    Parameters:
    - col_data: The dataframe column.
    - col_name: The name of the dataframe column.

    Returns:
    - The dropdown dimension.
    """    

    max_length = max(col_data.apply(lambda x: len(str(x))))
    print(max_length,col_name)
    # Set a higher max width for 'title' column
    if col_name == 'title':
        return max(150, min(max_length * 10, 600))  # Minimum 150px, maximum 400px for 'title'
    return max(80, min(max_length * 8, 300))  # Ensure minimum 80px and maximum 300px width for others
    

"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def get_column_width(col_name):

    """
    Goal: Calculate the associated dropdown for each table column.

    Parameters:
    - col_name: The name of the dataframe column.

    Returns:
    - The dropdown dimension.
    """    

    All_columns = ["title", "startYear", "runtimeMinutes", "genres", "isAdult", "averageRating", "numVotes", "directors", "writers", "nconst", "category", "characters", "isOriginalTitle"]
    
    All_width_columns = [400, 80, 80, 150, 80, 80, 80, 200, 200, 80, 200, 200, 80]

    # Create a dictionary mapping columns to their respective widths
    width_map = {All_columns[i]: All_width_columns[i] for i in range(len(All_columns))}
        
    return width_map.get(col_name, None)  # Returns None if the column isn't found
    

"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def dropdown_table(df, id_table, tab, dark_dropdown_style, uniform_style, need_dropdown):

    """
    Goal: Create the table and the associated dropdown.

    Parameters:
    - df: dataframe.
    - id_table: id of the table.
    - dark_dropdown_style: Color style of the dropdown.
    - uniform_style: Color style of the dropdown.
    - need_dropdown: Bool to decide if the table has some dropdowns or not.

    Returns:
    - dropdowns_with_labels: The table dropdowns. 
    - data_table: The data tables. 
    """    
    
    
    print(df.columns)
    
    columns = df.columns

    # Calculate widths, ensuring 'title' is handled specifically
    column_widths = {col: get_column_width(col) for col in columns}
    
    if need_dropdown == True:
        # Create dropdowns using calculated widths
        dropdowns_with_labels = []
        for col in columns:
            dtype = df[col].dtype
        
            # Container for each input/dropdown
            container_style = {'display': 'flex', 'flexDirection': 'column', 'width': '100%'}  # Flex for vertical stacking
            element = None

            dropdown_style = {'width': f'{column_widths[col]}px', 'height': '40px', 'boxSizing': 'border-box'}
    
            if dtype == "float64":
                element = dcc.Input(
                    id=f'{col}-dropdown-table-' + tab,
                    type='text',
                    debounce=True,
                    className='dash-input dynamic-width',
                    style=dropdown_style
                )
            else:
                # Collect all unique values, splitting them by commas and ensuring uniqueness
                all_roles = set()
                for value in df[col].dropna().unique():
                    roles = [role.strip() for role in str(value).split(',')]
                    all_roles.update(roles)
        
                unique_values = sorted(all_roles)
        
                element = dcc.Dropdown(
                    id=f'{col}-dropdown-table-' + tab,
                    options=[{'label': val, 'value': val} for val in unique_values],
                    className='dash-dropdown dynamic-width',
                    style=dropdown_style,
                    multi=True,
                    clearable=True
                ) 

            # Append each element wrapped in a container
            dropdowns_with_labels.append(html.Div(
                children=[element],
                style=container_style  # Use flexbox styling for the container
            ))

    else:
        dropdowns_with_labels = None
    


    data_table = dash_table.DataTable(
        id=id_table,
        data=df.to_dict('records'),
        columns=[{'id': c, 'name': c} for c in columns],
        fixed_rows={'headers': True},
        style_table={
            # 'className': 'table-container',  # Apply table container style
            'minWidth': str(int(len(columns) * 170)) + 'px',
            'overflowX': 'auto',
            'paddingLeft': '2px',
            'paddingRight': '20px',
            'marginLeft': '8px'
        },
        style_header={
            # 'className': 'table-header',    # Apply header style
            'backgroundColor': '#343a40',
            'color': 'white',
            'whiteSpace': 'nowrap',
            'textAlign': 'center',
        },
        style_cell={
            # 'className': 'table-cell',       # Apply cell style
            'backgroundColor': '#1e1e1e',
            'color': '#f8f9fa',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'whiteSpace': 'nowrap',
            'textAlign': 'center',
        },
        style_data={
            'whiteSpace': 'nowrap',
            'textAlign': 'center',
        },
        style_data_conditional=[
            {
                'if': {'column_id': col},
                'width': f'{column_widths[col]}px'
            } for col in columns
        ]
    )

    return dropdowns_with_labels, data_table
