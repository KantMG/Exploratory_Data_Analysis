#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:00:53 2024

@author: quentin
"""


"""#=============================================================================
   #=============================================================================
   #=============================================================================

    Dictionnary of functions to get information on the dataset via the html page.

#=============================================================================
   #=============================================================================
   #============================================================================="""


import requests
from bs4 import BeautifulSoup
from termcolor import colored


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def info_source(url,files=None, detail_file=None):

    """
    Goal: 
    - Acces to the data set documentation by extracting the relevant informations on the html web page.
    
    Parameters:
    - url: url web page.
    - detail_file: Bool to get a reading with more details.
    
    Returns:
    - None
    """
    
    print()
    print(colored("""################### Infos on sources ###################""", "red"))
    
    # URL of the HTML page you want to read
    
    # Fetch the content of the URL (requests.get(url): Sends a GET request to the specified URL.)
    response = requests.get(url)

    
    # Check if the request was successful 
    if response.status_code == 200:
        # Parse the HTML content with BeautifulSoup, allowing you to navigate and manipulate it.
        # response.content: Retrieves the raw HTML content of the page.
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Print the parsed HTML (or you can access specific elements)
        # Formats the HTML content in a readable manner.
        HTML_readable=soup.prettify()


        if files==None:
            # Extract all headings and paragraphs intelligently
            for section in soup.find_all(['h1', 'h2', 'p']):
                print()
                print(section.get_text(strip=True))
                
            h3_tags = soup.find_all('h3')
            # Extract all <li> items that are inside or directly related to <h3> sections 
            print("\nList Items Inside <h3> Sections:")
            for h3 in h3_tags:
                # Find the next sibling that is a <ul> or <ol> containing <li> items
                next_sibling = h3.find_next_sibling(['ul', 'ol'])
                if next_sibling: 
                    print(f"\nSection: {h3.get_text()}") 
                    list_items = next_sibling.find_all('li') 
                    for li in list_items: 
                        print(f"- {li.get_text().strip()}")
    
            # Extract all links to the datasets
            for link in soup.find_all('a', href=True):
                href = link['href']
                if "datasets" in href:
                    print(f"Dataset Link: {href}")
        else:
            
            files = [ file+'.tsv.gz' for file in files ]
            
            h3_tags = soup.find_all('h3')
            # Extract all <li> items that are inside or directly related to <h3> sections 
            for h3 in h3_tags:
                # Find the next sibling that is a <ul> or <ol> containing <li> items
                next_sibling = h3.find_next_sibling(['ul', 'ol'])
                if next_sibling: 
                    
                    if h3.get_text() in files:
                        if detail_file=='Yes':
                            print(f"\nSection: {h3.get_text()}") 
                        list_items = next_sibling.find_all('li') 
                        items_colum=[]
                        for li in list_items: 
                            if detail_file=='Yes':
                                print(f"- {li.get_text().strip()}")
                            items_colum.append(li.get_text().strip().split(' ')[0])
                        print(f"\n"+colored(h3.get_text(), "red",attrs=["bold"])+" is an array of dimension "+str(len(list_items)))
                        for i in range(len(items_colum)) : print( colored(items_colum[i], "green",attrs=["bold"]))
                

    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")    
    
    print(colored("""########################################################""", "red"))
    print()
          
