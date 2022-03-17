# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 20:45:09 2022

@author: HP
"""
import requests

url = "https://www.wikidata.org/w/api.php"

query = input("Enter name : ")

params = {
        "action" : "wbsearchentities",
        "language" : "en",
        "format" : "json",
        "search" : query
    }

try:
    data = requests.get(url,params=params)
    
    print(query," : ",data.json()["search"][0]["description"])
except:
    print("Invalid Input try again!")