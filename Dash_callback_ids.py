#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:58:40 2024

@author: quentin
"""


"""#=============================================================================
   #=============================================================================
   #=============================================================================

    Dictionnary of functions for callback.

#=============================================================================
   #=============================================================================
   #============================================================================="""


import re
import ast

import plotly.graph_objects as go
import plotly.io as pio
from collections import defaultdict


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def get_component_ids(layout):

    """
    Goal: 
    - Get all the current ids in dash.
    
    Parameters:
    - layout: Dash layout.
    
    Returns:
    - ids: List of the current ids in dash.
    """

    ids = []
    
    # Handle layout as a component
    if hasattr(layout, 'id'):
        if layout.id:  # Check if id is not None or empty string
            ids.append(layout.id)

    # Handle layout as a list or tuple (children)
    if isinstance(layout, (list, tuple)):
        for item in layout:
            ids.extend(get_component_ids(item))
    
    # Handle layout as a component's children
    if hasattr(layout, 'children'):
        # Recursively get IDs from children
        ids.extend(get_component_ids(layout.children))

    return ids


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def create_flowchart_from_dash_app(file_path, target_ids=None):

    """
    Goal: 
    - Create a fig which is a flowchart of a dash app.
    
    Parameters:
    - file_path: Path of the python program.
    - target_ids: List of targets id to start the flowchart.
    
    Returns:
    - The flowchart figure.
    """    

    # Read the Python file
    with open(file_path, 'r') as file:
        code = file.read()
    
    # Regular expression to find imports with aliases
    import_pattern = r'import\s+(\w+)\s+as\s+(\w+)'
    imports = dict(re.findall(import_pattern, code))

    # Regular expression to find callbacks and their bodies
    callback_pattern = r'@app\.callback\s*\(([\s\S]*?)\)\s*def\s*(\w+)\s*\(.*?\):([\s\S]*?)(?=\n\s*def|\Z)'
    callback_matches = re.findall(callback_pattern, code)

    if not callback_matches:
        print("No callbacks found.")
        return

    # Store callback dependencies (outputs and inputs) for connection
    callback_dependencies = {}

    # Create nodes
    nodes = [{'name': 'Start', 'x': 0.5, 'y': 1.0}]
    edges = []
    communication_edges = []
    chain_edges = []
    space_between_nodes = 0.15  # Base space between nodes
    current_y = 1.0

    for i, (callback_decorators, callback_name, callback_body) in enumerate(callback_matches):
        function_calls = []
        outputs = []
        inputs = []

        # Capture outputs and inputs from callback decorators
        output_pattern = r'Output\(([^)]+)\)'
        input_pattern = r'Input\(([^)]+)\)'
        outputs.extend(re.findall(output_pattern, callback_decorators))
        inputs.extend(re.findall(input_pattern, callback_decorators))

        # Map callback dependencies
        callback_dependencies[callback_name] = {'outputs': outputs, 'inputs': inputs}

        # Detect function calls in the callback body
        for alias in imports.values():
            function_pattern = rf'{alias}\.(\w+)\('
            called_functions = re.findall(function_pattern, callback_body)
            function_calls.extend([f'{alias}.{func}' for func in called_functions])

        # Create a functions summary for hover info
        if function_calls:
            functions_summary = "Functions:<br>" + "<br>".join(function_calls)
        else:
            functions_summary = "Functions: None"
        
        # Adjust space based on the node existence
        node_gap = space_between_nodes
        current_y -= node_gap

        # Add the node with just the callback name and store the functions summary in hoverinfo and hovertext
        nodes.append({
            'name': callback_name,
            'x': 0.5,
            'y': current_y,
            'hoverinfo': functions_summary  # This will be accessed as hovertext
        })
        edges.append((0, i + 1))  # Connect each callback to the Start node

    # Identify callback communication and chain edges
    for start_callback, start_deps in callback_dependencies.items():
        for output in start_deps['outputs']:
            for end_callback, end_deps in callback_dependencies.items():
                if output in end_deps['inputs']:
                    start_index = next(i for i, node in enumerate(nodes) if node['name'] == start_callback)
                    end_index = next(i for i, node in enumerate(nodes) if node['name'] == end_callback)
                    communication_edges.append((start_index, end_index))  # Store the edge for red arrows
                    
    # Check for chains starting from each target ID
    if target_ids:
        # Extract component IDs from output strings and compare against target_ids
        target_callbacks = set()
        # Loop through each callback and check if any of its outputs match the target_ids
        for cb_name, deps in callback_dependencies.items():
            for output_str in deps['outputs']:
                # Extract the component ID from the output string, e.g., "'tabs-content', 'children'" -> "tabs-content"
                output_id = re.search(r"'([^']+)'", output_str)  # Extracts the first quoted text as component ID
                if output_id and output_id.group(1) in target_ids:
                    target_callbacks.add(cb_name)
        
        print("Target Callbacks:", target_callbacks)
        
        for start_callback in target_callbacks:
            to_visit = [start_callback]
            visited = set()

            # Traverse callbacks triggered by the starting callback
            while to_visit:
                current_callback = to_visit.pop()
                visited.add(current_callback)
                
                print(visited)
                # Find the outputs that are used as inputs in other callbacks
                for output in callback_dependencies[current_callback]['outputs']:
                    for next_callback, deps in callback_dependencies.items():
                        if output in deps['inputs'] and next_callback not in visited:
                            start_index = next(i for i, node in enumerate(nodes) if node['name'] == current_callback)
                            end_index = next(i for i, node in enumerate(nodes) if node['name'] == next_callback)
                            chain_edges.append((start_index, end_index))  # Chain edges for selected target IDs
                            to_visit.append(next_callback)
                            print(start_index, end_index)
                            
    print("to_visit=",to_visit)
    print("communication_edges=",communication_edges)
    print()
    # Create the figure
    fig = go.Figure()

    # Add nodes to the figure
    for node in nodes:
        # Adding hovertext from functions summary defined earlier
        hover_text = node.get('hoverinfo', 'No functions associated')
        fig.add_trace(go.Scatter(
            x=[node['x']],
            y=[node['y']],
            text=[node['name']],
            mode='text+markers',
            marker=dict(size=40, color='darkolivegreen'),
            textfont=dict(size=14),
            showlegend=False,
            hoverinfo='text',
            hovertext=hover_text  # Set hover info for the marker
        ))

    # Add default edges with arrows at the end
    for start, end in edges:
        x_coords = [nodes[start]['x'], nodes[end]['x']]
        y_coords = [nodes[start]['y'], nodes[end]['y']]

        # Draw lines
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines',
            line=dict(color='DarkSlateBlue', width=2),
            showlegend=False,
            hoverinfo='none'
        ))

        # Add arrow annotations at end of line
        fig.add_annotation(
            x=x_coords[1],
            y=y_coords[1],
            ax=x_coords[0],
            ay=y_coords[0],
            showarrow=True,
            arrowsize=1,
            arrowhead=3,
            arrowcolor='DarkSlateBlue',
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)',
        )

    # Add red arrows for communication edges
    for start, end in communication_edges:
        x_coords = [nodes[start]['x'], nodes[end]['x']]
        y_coords = [nodes[start]['y'], nodes[end]['y']]

        # Draw red lines for communication edges
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),  # Dashed red line for communication
            showlegend=False,
            hoverinfo='none'
        ))

        # Add red arrow annotations at end of line
        fig.add_annotation(
            x=x_coords[1],
            y=y_coords[1],
            ax=x_coords[0],
            ay=y_coords[0],
            showarrow=True,
            arrowsize=1,
            arrowhead=3,
            arrowcolor='red',
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)',
        )

    # Add blue arrows for chain edges starting from target IDs
    for start, end in chain_edges:
        x_coords = [nodes[start]['x'], nodes[end]['x']]
        y_coords = [nodes[start]['y'], nodes[end]['y']]

        # Draw blue lines for chain edges
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines',
            line=dict(color='blue', width=2),  # Solid blue line for chains
            showlegend=False,
            hoverinfo='none'
        ))

        # Add blue arrow annotations at end of line
        fig.add_annotation(
            x=x_coords[1],
            y=y_coords[1],
            ax=x_coords[0],
            ay=y_coords[0],
            showarrow=True,
            arrowsize=1,
            arrowhead=3,
            arrowcolor='blue',
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)',
        )

    # Update layout for dark mode and titles
    figname = 'Flowchart of Dash Application Callback Chains for Target IDs'
    fig.update_layout(
        plot_bgcolor='#343a40',
        paper_bgcolor='#343a40',
        font=dict(color='white'),
        title=figname,
        title_font=dict(size=20, color='white'),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        height=1000,
    )

    return fig


"""#=============================================================================
   #=============================================================================
   #============================================================================="""

def create_flowchart_from_dash_app(file_path, target_ids=None):

    """
    Goal: 
    - Create a fig which is a flowchart of a dash app.
    
    Parameters:
    - file_path: Path of the python program.
    - target_ids: List of targets id to start the flowchart.
    
    Returns:
    - The flowchart figure.
    """    
    
    target_ids = ["tabs-1", "tabs-2", "tabs-3"]
    
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read())

    # Extract variable values, particularly lists like List_col_tab2
    variable_values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            if isinstance(node.value, ast.List):  # Only handle lists
                variable_values[var_name] = [elt.s for elt in node.value.elts if isinstance(elt, ast.Str)]

    callbacks = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            callback_decorator = next((dec for dec in node.decorator_list 
                                       if isinstance(dec, ast.Call) and getattr(dec.func, 'attr', None) == 'callback'), None)
            if callback_decorator:
                function_name = node.name
                output_ids = []
                input_ids = []

                for arg in callback_decorator.args:
                    if isinstance(arg, ast.Call) and arg.func.id == 'Output':
                        output_ids.extend(extract_io_ids(arg, variable_values))
                    elif isinstance(arg, (ast.List, ast.Call, ast.ListComp, ast.BinOp)):
                        input_ids.extend(extract_io_ids(arg, variable_values))

                callbacks.append({
                    'function_name': function_name,
                    'outputs': [(comp_id, prop) for comp_type, comp_id, prop in output_ids if comp_type == 'Output'],
                    'inputs': [(comp_id, prop) for comp_type, comp_id, prop in input_ids if comp_type == 'Input'],
                })
        
    # Map each component ID to callbacks that use it as an input
    input_to_callback = defaultdict(list)    
    for cb in callbacks:
        for comp_id, prop in cb['inputs']:
            input_to_callback[comp_id].append(cb)
            
            
    # Build flowchart structure by target IDs
    def build_branch(comp_id):
        if comp_id not in input_to_callback:
            return {'id': comp_id, 'branches': []}
        
        print(f"Processing component ID: {comp_id}")
        branches = []
        
        for cb in input_to_callback[comp_id]:
            print(f"Callback for {comp_id}: {cb}")
            
            # Include outputs as branches
            for output_id, _ in cb['outputs']:
                print(f"Found output: {output_id}")
                branches.append(build_branch(output_id))
            
        # If there are no branches but this component has inputs, include it
        if not branches:
            branches.append({'id': comp_id, 'branches': []})
            print(f"No branches found for {comp_id}. Adding as input.")
        
        return {'id': comp_id, 'branches': branches}

    
    # Create branches for each target_id if provided
    flowchart_structure = []
    if target_ids:
        for target_id in target_ids:
            flowchart_structure.append(build_branch(target_id))
    else:
        for comp_id in input_to_callback:
            flowchart_structure.append(build_branch(comp_id))

    # Display the flowchart structure
    def print_flowchart(branch, level=0):
        print("    " * level + f"- {branch['id']}")
        for sub_branch in branch['branches']:
            print_flowchart(sub_branch, level + 1)

    print("Flowchart Structure:")
    for branch in flowchart_structure:
        print_flowchart(branch)
        print()


"""#=============================================================================
   #=============================================================================
   #============================================================================="""

def build_hierarchy(flowchart_info):
    
    """
    Goal: Build a hierarchy by consolidating identical dropdown chains into a single notation.

    Parameters:
    - flowchart_info: The original nested dictionary structure.

    Returns:
    - final_hierarchy: The final version of the hierarchy.
    """
    
    # Initialize the hierarchy dictionary
    hierarchy = {
        'tabs-1': {'inputs': {}, 'outputs': {}},
        'tabs-2': {'inputs': {}, 'outputs': {}},
        'tabs-3': {'inputs': {}, 'outputs': {}},
    }

    # Mapping from function name to its outputs
    function_outputs = {}

    # First pass: Build input to functions and function to outputs mappings
    for item in flowchart_info:
        function_name = item['function_name']
        outputs = item['outputs']
        inputs = item['inputs']
        
        # Determine which tab the function should belong to based on inputs
        current_tab = None
        for input_id in inputs:
            component_id = input_id[0]
            if component_id in ['tabs-1', 'tabs-2', 'tabs-3']:
                current_tab = component_id
            elif 'tab-2' in component_id and component_id != 'tabs-2':
                current_tab = 'tabs-2'
            elif 'tab-3' in component_id and component_id != 'tabs-3':
                current_tab = 'tabs-3'
            if current_tab:
                break

        if current_tab:
            # Record inputs with corresponding functions
            for input_id in inputs:
                if input_id not in hierarchy[current_tab]['inputs']:
                    hierarchy[current_tab]['inputs'][input_id] = []
                if function_name not in hierarchy[current_tab]['inputs'][input_id]:
                    hierarchy[current_tab]['inputs'][input_id].append(function_name)

            # Record function outputs
            function_outputs[function_name] = outputs
            
            # Record outputs for the current tab
            for output_id in outputs:
                if output_id not in hierarchy[current_tab]['outputs']:
                    hierarchy[current_tab]['outputs'][output_id] = []
                if function_name not in hierarchy[current_tab]['outputs'][output_id]:
                    hierarchy[current_tab]['outputs'][output_id].append(function_name)

    # Build nested hierarchy for each input
    def build_nested_hierarchy(tab, input_id):
        # Create a nested structure for the given input_id
        nested_structure = {}
        functions = hierarchy[tab]['inputs'].get(input_id, [])
        
        for function in functions:
            # Get outputs for this function
            outputs = function_outputs.get(function, [])
            # Initialize the function's output in the nested structure
            nested_structure[function] = {}
            for output_id in outputs:
                # Add output to the structure
                nested_structure[function][output_id] = build_nested_hierarchy(tab, output_id)

        return nested_structure
    
    # print()
    # Final output hierarchy with nested structures for inputs
    final_hierarchy = {}
    for tab in hierarchy.keys():
        final_hierarchy[tab] = {}
        for input_id in hierarchy[tab]['inputs']:
            final_hierarchy[tab][input_id] = build_nested_hierarchy(tab, input_id)
            # print(input_id)
            # print(final_hierarchy[tab][input_id])
            # print()

    return final_hierarchy


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def simplify_hierarchy(hierarchy):
    
    """
    Goal: Simplifies the hierarchy by consolidating identical dropdown chains into a single notation.

    Parameters:
    - hierarchy (dict): The original nested dictionary structure.

    Returns:
    - dict: A simplified version of the hierarchy with repeated structures consolidated.
    """
    
    simplified_hierarchy = {}

    for key, value in hierarchy.items():
        # Check if the key matches the fig-dropdown-* pattern and follows the same structure
        if isinstance(key, tuple) and key[0].startswith("fig-dropdown-") and "-tab-2" in key[0]:
            standardized_key = "List of fig-dropdown-*-tab-2 (value)"
            if standardized_key not in simplified_hierarchy:
                simplified_hierarchy[standardized_key] = value
        else:
            # For non-repetitive nodes, add them as is
            if isinstance(value, dict):
                simplified_hierarchy[key] = simplify_hierarchy(value)
            else:
                simplified_hierarchy[key] = value

    return simplified_hierarchy



def print_hierarchy(hierarchy):
    for tab_name in hierarchy.keys():
        print(f"{tab_name}")
        for input_id, functions in hierarchy[tab_name].items():
            # Print each input and its corresponding functions/outputs
            input_label = f"{input_id[0]} ({input_id[1]})"
            print(f"├── {input_label}")
            print_functions(functions, "│   ")

def print_functions(functions, indent):
    printed_outputs = set()
    function_count = len(functions)
    for i, (function, outputs) in enumerate(functions.items()):
        # Use "└──" for the last function at this level to end correctly
        func_prefix = "└──" if i == function_count - 1 else "├──"
        
        for output_id, sub_outputs in outputs.items():
            output_label = f"{output_id[0]} ({output_id[1]})"
            # Ensure each output only appears once per level
            if output_label not in printed_outputs:
                print(f"{indent}{func_prefix} {output_label}")
                printed_outputs.add(output_label)
                # If sub_outputs exist, recursively print them
                if sub_outputs:
                    new_indent = indent + ("    " if func_prefix == "└──" else "│   ")
                    print_functions(sub_outputs, new_indent)




"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def create_hierarchy_figure(hierarchy):
    
    """
    Goal: Create a Sunburst visualization of the hierarchical structure from a dictionary hierarchy.
    
    Parameters:
    - hierarchy (dict): The nested dictionary representing the hierarchy structure.
    
    Returns:
    - fig (go.Figure): A Plotly figure representing the hierarchy as a Sunburst chart.
    """
    
    labels = []
    parents = []
    
    def traverse_hierarchy(hierarchy, parent_label=""):
        """
        Recursive function to traverse the hierarchy and populate labels and parents for the Sunburst chart.
        
        Parameters:
        - hierarchy (dict): The current level of the hierarchy to process.
        - parent_label (str): The label of the parent node at the current level.
        """
        for node, children in hierarchy.items():
            # Add the current node and its parent
            labels.append(node)
            parents.append(parent_label)
            
            # Recurse to process children if they exist
            if isinstance(children, dict) and children:
                traverse_hierarchy(children, node)

    # Start traversal from the top level of the hierarchy
    traverse_hierarchy(hierarchy)

    # Create the Sunburst figure using the collected labels and parents
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        branchvalues="total",
        marker=dict(
            colorscale="Viridis"  # Optional: Color scheme to visually distinguish levels
        )
    ))

    # Update layout for dark mode and titles
    figname = 'Hierarchical Visualization of Callback Dependencies'
    fig.update_layout(
        plot_bgcolor='#343a40',
        paper_bgcolor='#343a40',
        font=dict(color='white'),
        title=figname,
        title_font=dict(size=20, color='white'),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        height=1000,
    )

    return fig


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def create_detailed_flowchart(file_path, target_ids=None):

    with open(file_path, 'r') as file:
        tree = ast.parse(file.read())

    # Extract variable values, particularly lists like List_col_tab2
    variable_values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            if isinstance(node.value, ast.List):  # Only handle lists
                variable_values[var_name] = [elt.s for elt in node.value.elts if isinstance(elt, ast.Str)]

    flowchart_info = []
    function_names = []
    node_pos = {}
    
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            callback_decorator = next((dec for dec in node.decorator_list 
                                       if isinstance(dec, ast.Call) and getattr(dec.func, 'attr', None) == 'callback'), None)
            if callback_decorator:
                function_name = node.name
                output_ids = []
                input_ids = []

                # Process outputs first to handle lists
                for arg in callback_decorator.args:
                    if isinstance(arg, ast.List):
                        output_ids.extend(extract_io_ids(arg, variable_values))
                    elif isinstance(arg, ast.Call) and arg.func.id == 'Output':
                        output_ids.append((arg.func.id, arg.args[0].s, arg.args[1].s))
                    elif isinstance(arg, (ast.ListComp, ast.BinOp)):
                        output_ids.extend(extract_io_ids(arg, variable_values))

                # Process inputs
                for arg in callback_decorator.args[1:]:  # Skip the first element since it's outputs
                    input_ids.extend(extract_io_ids(arg, variable_values))

                # Organize function details for each callback
                flowchart_info.append({
                    'function_name': function_name,
                    'outputs': [(comp_id, prop) for comp_type, comp_id, prop in output_ids if comp_type == 'Output'],
                    'inputs': [(comp_id, prop) for comp_type, comp_id, prop in input_ids if comp_type == 'Input'],
                })
                function_names.append(function_name)

    # Define node positions (x, y) for visualization
    num_functions = len(function_names)
    for idx, function in enumerate(function_names):
        node_pos[function] = (idx, 1)  # Place nodes horizontally
    

    return flowchart_info,function_names


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def extract_io_ids(arg, variable_values):
    """
    Goal: Extracts Input and Output component IDs and properties, handling lists, comprehensions, and concatenations.

    Parameters:
    - arg: 
    - variable_values: 
    
    Returns:
    - ids: List of the ids
    """

    ids = []
    if isinstance(arg, ast.List):  # Regular list of Inputs or Outputs
        for item in arg.elts:
            ids.extend(extract_io_ids(item, variable_values))
    elif isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name):
        if arg.func.id in {'Input', 'Output'}:
            # Handle individual Input/Output definitions
            ids.append((arg.func.id, arg.args[0].s, arg.args[1].s))
    elif isinstance(arg, ast.ListComp):  # List comprehension
        # Call extract_comprehension_ids to handle list comprehensions
        ids.extend(extract_comprehension_ids(arg, variable_values))
    elif isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Add):  # Handle concatenations
        # Recursively process the left and right parts of the concatenation
        ids.extend(extract_io_ids(arg.left, variable_values))
        ids.extend(extract_io_ids(arg.right, variable_values))
        
    return ids


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def extract_comprehension_ids(arg, variable_values):
    
    """
    Goal: Extracts IDs from comprehensions by substituting values in variables like List_col_tab2.
    
    Parameters:
    - arg: 
    - variable_values: 
    
    Returns:
    - ids: List of the ids
    """
    
    ids = []
    if isinstance(arg, ast.ListComp):  # List comprehension
        if isinstance(arg.elt, ast.Call) and isinstance(arg.elt.func, ast.Name):
            comp_type = arg.elt.func.id  # Either 'Input' or 'Output'
            component_template = arg.elt.args[0]  # e.g., f'checkbox-{col}-tab-2'
            property = arg.elt.args[1].s  # e.g., 'value'
            
            if isinstance(component_template, ast.JoinedStr):  # Handle f-strings
                for comp in variable_values.get(arg.generators[0].iter.id, []):  # Substitute list values
                    formatted_id = ''.join([
                        part.s if isinstance(part, ast.Str) else str(comp) for part in component_template.values
                    ])
                    ids.append((comp_type, formatted_id, property))
            else:
                # Handle regular string construction
                ids.append((comp_type, component_template.s, property))
    return ids

