#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 18:23:15 2024

@author: quentin
"""

"""#=============================================================================
   #=============================================================================
   #=============================================================================

    Dictionnary of functions for figure dropdown creation.

#=============================================================================
   #=============================================================================
   #============================================================================="""


import dash
from dash import dcc, html, Input, Output, dash_table, callback, callback_context
import dash_bootstrap_components as dbc
import pandas as pd

from termcolor import colored


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
    
    if col_data.empty:
        return 0  # Return 0 or a default width if there are no values
    
    max_length = max(col_data.apply(lambda x: len(str(x))))
    # Set a higher max width for 'title' column
    if col_name == 'title':
        return max(150, min(max_length * 10, 600))  # Minimum 150px, maximum 400px for 'title'
    return max(80, min(max_length * 8, 300))  # Ensure minimum 80px and maximum 300px width for others


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def dropdown_figure(df, id_graph, tab, dark_dropdown_style, uniform_style, Large_file_memory):

    """
    Goal: Create the dropdown associated to a figure.

    Parameters:
    - df: dataframe.
    - id_graph: id of the graphic.
    - dark_dropdown_style: Color style of the dropdown.
    - uniform_style: Color style of the dropdown.
    - Large_file_memory: Estimate if the file is too large to be open with panda.

    Returns:
    - dropdowns_with_labels: The finalized dropdowns figure. 
    """

    # Get column names
    columns = df.columns
    
    # Get the list of y function
    function_on_y = ["Avg", "Avg on the ordinate", "Weight on y"]
    
    # Get the type of graph
    graph_type = ["Histogram", "Curve", "Scatter", "Boxes", "Colormesh"]

    # Get the graph dimension
    dim_type = ["1D", "2D", "3D"]
    
    # Get the list of axis and graph function
    axis = ["x", "y", "z", "Func on y", "Func on z", "Graph", "Dim"]


    # Define a consistent style for both input and dropdown elements
    uniform_style = {
        'width': '160px',  # Set a consistent width
        'height': '40px',  # Set a consistent width
        'borderRadius': '5px',  # Optional: Add rounded corners
        # 'backgroundColor': '#1e1e1e',
        # 'color': '#white'
    }
    
    dropdown_container_style = {'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'}
    
    # Create the dropdowns for each column
    dropdowns_with_labels = []
    for axi in axis:
        if axi == 'Dim':
            # Get unique values and sort them
            dropdown_with_label = html.Div(
                style=dropdown_container_style,
                children=[
                    html.Label(f'Select {axi}'),  # Label for the dropdown
                    dcc.Dropdown(
                        id=f'{axi}-dropdown-'+tab,
                        options=[{'label': val, 'value': val} for val in dim_type],
                        value='1D',  # Set default to "All", meaning no filtering
                        style=uniform_style,
                        # className='dash-dropdown',
                        clearable=True
                    )
                ]
            )
        elif axi == 'Graph':
            dropdown_with_label = html.Div(
                style=dropdown_container_style,
                children=[
                    html.Label(f'Select {axi}'),  # Label for the dropdown
                    dcc.Dropdown(
                        id=f'{axi}-dropdown-'+tab,
                        options=[{'label': val, 'value': val} for val in graph_type],
                        value='Histogram',  # Set default to "All", meaning no filtering
                        style=uniform_style,
                        className='dash-dropdown',
                        clearable=True
                    )
                ]
            )
        elif axi == 'Func on z':
            dropdown_with_label = html.Div(
                style=dropdown_container_style,
                children=[
                    html.Label(f'Select {axi}'),  # Label for the dropdown
                    dcc.Dropdown(
                        id=f'{axi}-dropdown-'+tab,
                        options=[{'label': val, 'value': val} for val in function_on_y],
                        value="Avg",
                        style=uniform_style,
                        className='dash-dropdown'
                        # clearable=True
                    )
                ]
            )
        elif axi == 'Func on y':
            dropdown_with_label = html.Div(
                style=dropdown_container_style,
                children=[
                    html.Label(f'Select {axi}'),  # Label for the dropdown
                    dcc.Dropdown(
                        id=f'{axi}-dropdown-'+tab,
                        options=[{'label': val, 'value': val} for val in function_on_y],
                        # value=None,
                        style=uniform_style,
                        className='dash-dropdown',
                        clearable=True
                    )
                ]
            )
        elif axi== 'z':
            dropdown_with_label = html.Div(
                style=dropdown_container_style,
                children=[
                    html.Label(f'Select {axi}'),  # Label for the dropdown
                    dcc.Dropdown(
                        id=f'{axi}-dropdown-'+tab,
                        options=[{'label': val, 'value': val} for val in columns],
                        # value=None,
                        style=uniform_style,
                        className='dash-dropdown',
                        clearable=True
                    )
                ]
            )
        elif axi== 'y':
            dropdown_with_label = html.Div(
                style=dropdown_container_style,
                children=[
                    html.Label(f'Select {axi}'),  # Label for the dropdown
                    dcc.Dropdown(
                        id=f'{axi}-dropdown-'+tab,
                        options=[{'label': val, 'value': val} for val in columns],
                        # value=None,
                        style=uniform_style,
                        className='dash-dropdown',
                        clearable=True
                    )
                ]
            )
        else:
            dropdown_with_label = html.Div(
                style=dropdown_container_style,
                children=[
                    html.Label(f'Select {axi}'),  # Label for the dropdown
                    dcc.Dropdown(
                        id=f'{axi}-dropdown-'+tab,
                        options=[{'label': val, 'value': val} for val in columns],
                        # value=None,
                        style=uniform_style,
                        className='dash-dropdown',
                        clearable=True
                    )
                ]
            )

        dropdowns_with_labels.append(dropdown_with_label)

    return dropdowns_with_labels


"""#=============================================================================
   #=============================================================================
   #============================================================================="""

def dropdown_figure_filter(df, id_graph, tab, dark_dropdown_style, uniform_style):

    """
    Goal: Create the dropdown associated to a figure.
    These dropdowns are extra filters and the axis are not necessary shown on the graphic.

    Parameters:
    - df: dataframe.
    - id_graph: id of the graphic.
    - tab: name of the tab where the figure is located.
    - dark_dropdown_style: Color style of the dropdown.
    - uniform_style: Color style of the dropdown.

    Returns:
    - dropdowns_inputs_list: The list of filter dropdowns/Inputs for the figure. 
    """    

    columns = df.columns
    
    # Calculate widths, ensuring 'title' is handled specifically
    column_widths = {col: get_max_width(df[col], col) for col in columns}
    
    # Create dropdowns using calculated widths
    dropdowns_inputs_list = []
    for col in columns:
        dtype = df[col].dtype
        # dropdown_style = {**dark_dropdown_style, **uniform_style}  #, 'width': f'{column_widths[col]}px'
        dropdown_style = {'width': f'160px', 'height': '40px', 'boxSizing': 'border-box', 'backgroundColor': '#1e1e1e', 'color': '#f8f9fa'}
        
        dropdown_container_style = {'display': 'flex', 'flex-direction': 'column', 'margin': '2px 0'}  # Vertical alignment and spacing
        
        if dtype == "float64" or dtype == "int64":
            dropdown_with_label = html.Div(
                style=dropdown_container_style,
                children=[
                    html.Label(f'{col}:'),
                    dcc.Input(
                        id=f'fig-dropdown-{col}-'+tab,
                        type='text',
                        debounce=True,
                        style=dropdown_style,  # Adding margin for spacing
                        className='dash-input dynamic-width'
                    )
                ]
            )
        else:
            # Collect all unique values, splitting them by commas and ensuring uniqueness
            all_roles = set()
            for value in df[col].dropna().unique():
                # Split the value by comma and strip any extra spaces
                roles = [role.strip() for role in str(value).split(',')]
                all_roles.update(roles)
            
            # Convert to a sorted list
            unique_values = sorted(all_roles)
            
            dropdown_with_label = html.Div(
                style=dropdown_container_style,
                children=[
                    html.Label(f'{col}:'),
                    dcc.Dropdown(
                        id=f'fig-dropdown-{col}-'+tab,
                        options=[{'label': val, 'value': val} for val in unique_values],
                        style=dropdown_style,
                        className='dash-dropdown',
                        multi=True,
                        clearable=True
                    )
                ]
            )
    
        dropdowns_inputs_list.append(dropdown_with_label)
    
    return dropdowns_inputs_list





"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def button_modal_dropdowns_inputs(id_subname, text_button, df, id_graph, tab,
                               text_modal, dark_dropdown_style, uniform_style):

    """
    Goal: Create a button which give access to a modal.
    The modal contains a dropdown and an input with a submit button.

    Parameters:
    - id_subname: Part of all the id name associated with this button modal.
    - text_button: Text on the button.
    - df: dataframe.
    - id_graph: id of the graphic.
    - tab: name of the tab where the figure is located.
    - text_modal: Text at the Head of the modal.
    - dark_dropdown_style: Color style of the dropdown.
    - uniform_style: Color style of the dropdown.

    Returns:
    - The finalized dash button with its modal content. 
    - Creation of all the id:
        - "open-modal-"+id_subname: id of the button.
        - "fig-dropdown-{col}-tab: id of the dropdowns/inputs inside the modal.
        - "submit-button-"+id_subname: id of the submit button inside the modal.
        - "modal-"+id_subname: id of the modal.
        - "output-div-"+id_subname: id of the dash output.
        
    """       

    dropdown_style = {'width': f'200px', 'height': '40px', 'boxSizing': 'border-box'}

        
    return html.Div([
    dbc.Button(text_button, id="open-modal-"+id_subname, n_clicks=0, className='button'),
    dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(text_modal)),
            dbc.ModalBody(
                [   
                    
                    html.Div(
                        dropdown_figure_filter(df, id_graph, tab, dark_dropdown_style, uniform_style),
                        style={
                            'display': 'flex',
                            'flexWrap': 'wrap',  # Allow wrapping to new lines
                            'justify-content': 'flex-start',
                            'gap': '10px',  # Add spacing between dropdowns
                        }
                    )
                ]
            ),
            html.Span("", style={'margin': '0 10px'}),
            dbc.ModalFooter(
                dbc.Button("Submit", id="submit-button-"+id_subname, n_clicks=0, className='button')
            ),
        ],
        id="modal-"+id_subname,
        is_open=False,  # Initially closed
        className='top-modal',  # Apply the custom class here
        centered=True,
        size="lg",
    ),
    html.Div(id="output-div-"+id_subname) 
    ])


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def button_modal_double_input(id_subname, text_button, placeholder_input_1, placeholder_input_2,
                              text_modal, dark_dropdown_style, uniform_style):

    """
    Goal: Create a button which give access to a modal.
    The modal contains two inputs with a submit button.

    Parameters:
    - id_subname: Part of all the id name associated with this button modal.
    - text_button: Text on the button.
    - placeholder_input_1: Text inside the input 1 without content.
    - placeholder_input_2: Text inside the input 2 without content.
    - text_modal: Text at the Head of the modal.
    - dark_dropdown_style: Color style of the dropdown.
    - uniform_style: Color style of the dropdown.

    Returns:
    - The finalized dash button with its modal content.
    - Creation of all the id:
        - "open-modal-"+id_subname: id of the button.
        - "input_1-"+id_subname: id of the first input inside the modal.
        - "input_2-"+id_subname: id of the second input inside the modal.
        - "submit-button-"+id_subname: id of the submit button inside the modal.
        - "modal-"+id_subname: id of the modal.
        - "output-div-"+id_subname: id of the dash output.
        
    """   
    
    dropdown_style = {'width': f'200px', 'height': '40px', 'boxSizing': 'border-box'}
        
    return html.Div([
    dbc.Button(text_button, id="open-modal-"+id_subname, n_clicks=0, className='button'),
    dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(text_modal)),
            dbc.ModalBody(
                [
                    dcc.Input(id="input_1-"+id_subname, type="text", style=dropdown_style, className='dash-input dynamic-width', placeholder=placeholder_input_1),
                    html.Span(":", style={'margin': '0 10px'}),
                    dcc.Input(id="input_2-"+id_subname, type="text", style=dropdown_style, className='dash-input dynamic-width', placeholder=placeholder_input_2),
                ]
            ),
            html.Span("", style={'margin': '0 10px'}),
            dbc.ModalFooter(
                dbc.Button("Submit", id="submit-button-"+id_subname, n_clicks=0, className='button')
            ),
        ],
        id="modal-"+id_subname,
        is_open=False,  # Initially closed
        className='top-modal',  # Apply the custom class here
        centered=True,
        size="lg",
    ),
    html.Div(id="output-div-"+id_subname)
    ])


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def button_modal_dropdown_input(id_subname, text_button, option_dropdown, placeholder_input,
                               text_modal, dark_dropdown_style, uniform_style):

    """
    Goal: Create a button which give access to a modal.
    The modal contains a dropdown and an input with a submit button.

    Parameters:
    - id_subname: Part of all the id name associated with this button modal.
    - text_button: Text on the button.
    - option_dropdown: options of the dropdown inside the modal.
    - placeholder_input: Text inside the input without content.
    - text_modal: Text at the Head of the modal.
    - dark_dropdown_style: Color style of the dropdown.
    - uniform_style: Color style of the dropdown.

    Returns:
    - The finalized dash button with its modal content. 
    - Creation of all the id:
        - "open-modal-"+id_subname: id of the button.
        - "dropdown-"+id_subname: id of the dropdown inside the modal.
        - "input-"+id_subname: id of the input inside the modal.
        - "submit-button-"+id_subname: id of the submit button inside the modal.
        - "modal-"+id_subname: id of the modal.
        - "output-div-"+id_subname: id of the dash output.
        
    """       

    dropdown_style = {'width': f'200px', 'height': '40px', 'boxSizing': 'border-box'}

        
    return html.Div([
    dbc.Button(text_button, id="open-modal-"+id_subname, n_clicks=0, className='button'),
    dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(text_modal)),
            dbc.ModalBody(
                [   
                    dcc.Dropdown(
                        id="dropdown-"+id_subname,
                        options=option_dropdown,
                        # value='Decision Tree',
                        clearable=True,
                        style=dropdown_style,
                        className='dash-dropdown'
                    ),
                    html.Span(":", style={'margin': '0 10px'}),
                    dcc.Input(id="input-"+id_subname, type="number", style=dropdown_style, className='dash-input dynamic-width', placeholder=placeholder_input),
                ]
            ),
            html.Span("", style={'margin': '0 10px'}),
            dbc.ModalFooter(
                dbc.Button("Submit", id="submit-button-"+id_subname, n_clicks=0, className='button')
            ),
        ],
        id="modal-"+id_subname,
        is_open=False,  # Initially closed
        className='top-modal',  # Apply the custom class here
        centered=True,
        size="lg",
    ),
    html.Div(id="output-div-"+id_subname) 
    ])


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def button_modal_dropdown_and_double_input(id_subname, text_button, option_dropdown, placeholder_input_1,
                               placeholder_input_2, text_modal, dark_dropdown_style, uniform_style):

    """
    Goal: Create a button which give access to a modal.
    The modal contains a dropdown and an input with a submit button.

    Parameters:
    - id_subname: Part of all the id name associated with this button modal.
    - text_button: Text on the button.
    - option_dropdown: options of the dropdown inside the modal.
    - placeholder_input_1: Text inside the first input without content.
    - placeholder_input_2: Text inside the second input without content.
    - text_modal: Text at the Head of the modal.
    - dark_dropdown_style: Color style of the dropdown.
    - uniform_style: Color style of the dropdown.

    Returns:
    - The finalized dash button with its modal content. 
    - Creation of all the id:
        - "open-modal-"+id_subname: id of the button.
        - "dropdown-"+id_subname: id of the dropdown inside the modal.
        - "input-"+id_subname: id of the input inside the modal.
        - "submit-button-"+id_subname: id of the submit button inside the modal.
        - "modal-"+id_subname: id of the modal.
        - "output-div-"+id_subname: id of the dash output.
        
    """       

    dropdown_style = {'width': f'200px', 'height': '40px', 'boxSizing': 'border-box'}

        
    return html.Div([
    dbc.Button(text_button, id="open-modal-"+id_subname, n_clicks=0, className='button'),
    dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(text_modal)),
            dbc.ModalBody(
                [   
                    dcc.Dropdown(
                        id="dropdown-"+id_subname,
                        options=option_dropdown,
                        # value='Decision Tree',
                        clearable=True,
                        style=dropdown_style,
                        className='dash-dropdown'
                    ),
                    html.Span(":", style={'margin': '0 10px'}),
                    dcc.Input(id="input_1-"+id_subname, type="number", style=dropdown_style, className='dash-input dynamic-width', placeholder=placeholder_input_1),
                    html.Span(":", style={'margin': '0 10px'}),
                    dcc.Input(id="input_2-"+id_subname, type="number", style=dropdown_style, className='dash-input dynamic-width', placeholder=placeholder_input_2),
                ]
            ),
            html.Span("", style={'margin': '0 10px'}),
            dbc.ModalFooter(
                dbc.Button("Submit", id="submit-button-"+id_subname, n_clicks=0, className='button')
            ),
        ],
        id="modal-"+id_subname,
        is_open=False,  # Initially closed
        className='top-modal',  # Apply the custom class here
        centered=True,
        size="lg",
    ),
    html.Div(id="output-div-"+id_subname) 
    ])


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def button_modal_subplot_creation(id_subname, text_button, placeholder_input_1, placeholder_input_2, placeholder_input_3,
                               text_modal, dark_dropdown_style, uniform_style):

    """
    Goal: Create a button which give access to a modal.
    The modal contains a dropdown and an input with a submit button.

    Parameters:
    - id_subname: Part of all the id name associated with this button modal.
    - text_button: Text on the button.
    - placeholder_input_1: Text inside the input without content.
    - placeholder_input_2: Text inside the input without content.
    - placeholder_input_3: Text inside the input without content.
    - text_modal: Text at the Head of the modal.
    - dark_dropdown_style: Color style of the dropdown.
    - uniform_style: Color style of the dropdown.

    Returns:
    - The finalized dash button with its modal content. 
    - Creation of all the id:
        - "open-modal-"+id_subname: id of the button.
        - "input_1-"+id_subname: id of the input inside the modal.
        - "input_2-"+id_subname: id of the input inside the modal.
        - "input_3-"+id_subname: id of the input inside the modal.
        - "submit-button-"+id_subname: id of the submit button inside the modal.
        - "modal-"+id_subname: id of the modal.
        - "output-div-"+id_subname: id of the dash output.
        
    """       

    dropdown_style = {'width': f'200px', 'height': '40px', 'boxSizing': 'border-box'}

        
    return html.Div([
    dbc.Button(text_button, id="open-modal-"+id_subname, n_clicks=0, className='button'),
    dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(text_modal)),

            dbc.ModalBody(
                [   
                    dcc.Input(id="input_1-"+id_subname, type="number", style=dropdown_style, className='dash-input dynamic-width', placeholder=placeholder_input_1),
                    html.Span(":", style={'margin': '0 10px'}),
                    dcc.Input(id="input_2-"+id_subname, type="number", style=dropdown_style, className='dash-input dynamic-width', placeholder=placeholder_input_2),
                    html.Span(":", style={'margin': '0 10px'}),
                    dcc.Input(id="input_3-"+id_subname, type="number", style=dropdown_style, className='dash-input dynamic-width', placeholder=placeholder_input_3),

                ]
            ),
            html.Span("", style={'margin': '0 10px'}),
            dbc.ButtonGroup([
            dbc.ModalFooter(
                dbc.Button("Submit", id="submit-button-"+id_subname, n_clicks=0, className='button')
            ),
            dbc.ModalFooter(
                dbc.Button("Reset", id="submit-reset-button-"+id_subname, n_clicks=0, className='button mx-2', style = { 'backgroundColor': '#c0392b', 'color': '#1e1e1e'})
            ),])


        ],
        id="modal-"+id_subname,
        is_open=False,  # Initially closed
        className='top-modal',  # Apply the custom class here
        centered=True,
        size="lg",
    ),
    html.Div(id="output-div-"+id_subname) 
    ])


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def buttons_subplots(id_subname, text_button, nb_buttons, nb_buttons_row, nb_buttons_column,
                               dark_dropdown_style, uniform_style):

    """
    Goal: Create as many button as asked.

    Parameters:
    - id_subname: Part of all the id name associated with this button modal.
    - text_buttons: Text on the button.
    - nb_buttons: Number of buttons.
    - nb_buttons_row: Number of buttons on a row.
    - nb_buttons_column: Number of buttons on a column.
    - dark_dropdown_style: Color style of the dropdown.
    - uniform_style: Color style of the dropdown.

    Returns:
    - The finalized dash button with its modal content. 
    - Creation of all the id:
        - "open-modal-"+id_subname: id of the button.
        - "dropdown-"+id_subname: id of the dropdown inside the modal.
        - "input-"+id_subname: id of the input inside the modal.
        - "submit-button-"+id_subname: id of the submit button inside the modal.
        - "modal-"+id_subname: id of the modal.
        - "output-div-"+id_subname: id of the dash output.
        
    """       

    dropdown_style = {'width': f'200px', 'height': '40px', 'boxSizing': 'border-box'}
    
    button_list = []
        
    for nb_button in range(1, nb_buttons+1):
                
        button = html.Div(dbc.Button(text_button+str(nb_button), 
                                     # id=id_subname+str(nb_button), 
                                     id={'type': 'subplot-button', 'index': str(nb_button)},
                                     n_clicks=0, className='dash-input dynamic-width'))

        button_list.append(button)


    # Create rows
    rows = []
    for i in range(0, len(button_list), nb_buttons_row):
        row_buttons = button_list[i:i + nb_buttons_row]
        rows.append(html.Div(row_buttons, style={
            'display': 'flex',
            'justify-content': 'flex-start',
            'gap': '5px',
            'margin-bottom': '20px'  # Space below each row of buttons
        }))
    
    return html.Div(children=[
        html.Span("", style={'margin': '0 10px'}),
        *rows   # Unpack the rows list to include each row of buttons
    ])


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def figure_position_dash(tab, idgraph, dropdowns_with_labels_for_fig, 
                         dropdowns_with_labels_for_fig_filter, button_dropdown_function, 
                         button_dropdown_regression, button_dropdown_smoothing, button_subplot):

    """
    Goal: Create the dropdown associated to a figure.
    These dropdowns are extra filters and the axis are not necessary shown on the graphic.
    Furthermore, A checkbox is on located on the left of all Input and dropdown.

    Parameters:
    - tab: Name of the tab.
    - id_graph: id of the graphic.
    - dropdowns_with_labels_for_fig: The figue dropdowns.
    - dropdowns_with_labels_for_fig_filter: The figue dropdowns for extra filters (with or without checkbox).
    - button_dropdown_function: The button that open the modal for function creation.
    - button_dropdown_regression: The button that open the modal for regresison creation.
    - button_dropdown_smoothing: The button that open the modal for smoothing.
    - button_subplot: The button that open the modal for subplot creation.
    
    Returns:
    - The finalized figure with all the dropdowns and checkboxes on dash. 
    """   
    
    return html.Div(
        style={'display': 'flex', 'flex-direction': 'column', 'margin-top': '10px'},  # Use column direction for vertical stacking
        children=[
            # Dropdowns for the graph filters (above the graph)
            html.Div(
                dropdowns_with_labels_for_fig,
                style={
                    'display': 'flex',
                    'margin-left': '200px',
                    'justify-content': 'flex-start',
                    'gap': '5px',
                    'margin-bottom': '20px'  # Add space below the dropdowns
                }
            ),
            # Graph and dropdowns on the right (below the first set of dropdowns)
            html.Div(
                style={'display': 'flex'}, 
                children=[
                    # Graph on the left
                    html.Div(
                        [dcc.Graph(id=idgraph, style={'width': '100%', 'height': '600px'}),
                         dcc.Store(id='figure-store-'+tab, data={})], 
                        style={'margin-left': '20px', 'width': '70%'}
                    ),
                    # Dropdowns and heading in a vertical column on the right
                    html.Div(
                        style={'margin-left': '20px', 'width': '30%'},  # Container for the heading and dropdowns
                        children=[
                            # Heading above dropdowns
                            # html.H2(
                            #     'Select filters on the dataframe.',
                            #     style={'margin-bottom': '10px'},  # Add some space below the heading
                            #     className="text-light"
                            # ),
                            # Dropdowns in a vertical column
                            html.Div(
                                dropdowns_with_labels_for_fig_filter,
                                style={
                                    'display': 'flex',
                                    'justify-content': 'flex-start',
                                    'gap': '10px',  # Add spacing between dropdowns
                                }
                            ),
                            
                            html.Span("", style={'margin': '0 10px'}),
                            
                            html.Div(
                                button_dropdown_function,
                                style={
                                    'display': 'flex',
                                    'justify-content': 'flex-start',
                                    'gap': '10px',  # Add spacing between dropdowns
                                }
                            ),       
                            
                            html.Span("", style={'margin': '0 10px'}),
                            
                            html.Div(
                                button_dropdown_regression,
                                style={
                                    'display': 'flex',
                                    'justify-content': 'flex-start',
                                    'gap': '10px',  # Add spacing between dropdowns
                                }
                            ),  
                            
                            html.Span("", style={'margin': '0 10px'}),
                            
                            html.Div(
                                html.Button("Hide Dropdowns on figure", id='hide-dropdowns-'+tab, n_clicks=0, className='button'),
                                style={
                                    'display': 'flex',
                                    'justify-content': 'flex-start',
                                    'gap': '10px',  # Add spacing between dropdowns
                                }
                            ),  

                            html.Span("", style={'margin': '0 10px'}),
                            
                            html.Div(
                                button_dropdown_smoothing,
                                style={
                                    'display': 'flex',
                                    'justify-content': 'flex-start',
                                    'gap': '10px',  # Add spacing between dropdowns
                                }
                            ),  

                            html.Span("", style={'margin': '0 10px'}),
                            
                            html.Div(
                                button_subplot,
                                style={
                                    'display': 'flex',
                                    'justify-content': 'flex-start',
                                    'gap': '10px',  # Add spacing between dropdowns
                                }
                            ), 
                        ]
                    )
                ]
            )
        ]
    )


"""#=============================================================================
   #=============================================================================
   #============================================================================="""

