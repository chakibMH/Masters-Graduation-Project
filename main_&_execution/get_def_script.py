# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 20:45:09 2022

@author: HP
"""
import requests
import wikipediaapi

def wikidata_then_wikipedia(query):

    url = "https://www.wikidata.org/w/api.php"

    params = {
            "action" : "wbsearchentities",
            "language" : "en",
            "format" : "json",
            "search" : query
        }

    try:
        data = requests.get(url,params=params)
        
        info = data.json()["search"][0]["description"] 
        info = info.translate({ord(i): None for i in '(,"):;!?'})
        query = info.lower()
    except:
        wiki_wiki = wikipediaapi.Wikipedia('en')
        page_py = wiki_wiki.page(query)
        
        if page_py.exists() == False:
            print("Invalid Input try again!{query : ",query," }")
        else:
            wiki_wiki = wikipediaapi.Wikipedia('en')
            
            summary = page_py.summary
            summary = summary.lower()
            query = summary.replace(query.lower(),"",1)
            query = remove_extra_spaces(query)
             

    
    return query.lower()


def wikipedia(query):
    
    wiki_wiki = wikipediaapi.Wikipedia('en')
    page_py = wiki_wiki.page(query)
    
    
    if page_py.exists() == False:
        print("Invalid Input try again!{query : ",query," }")
    else:
        wiki_wiki = wikipediaapi.Wikipedia('en')
        
        summary = page_py.summary
        summary = summary.lower()
        query = summary.replace(query.lower(),"",1)
        query = remove_extra_spaces(query)
         
    return query.lower()


def wikidata(query):
    
    url = "https://www.wikidata.org/w/api.php"

    params = {
            "action" : "wbsearchentities",
            "language" : "en",
            "format" : "json",
            "search" : query
        }

    try:
        data = requests.get(url,params=params)
        
        info = data.json()["search"][0]["description"] 
        info = info.translate({ord(i): None for i in '(,"):;!?'})
        query = info.lower()
    except:
        print("Invalid Input try again!{query : ",query," }")
    
    return query.lower()

def remove_extra_spaces(txt):
    
    return " ".join(txt.split())



