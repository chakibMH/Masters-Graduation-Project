# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 11:01:44 2022

@author: HP
"""

import requests
from bs4 import BeautifulSoup
import csv
from itertools import zip_longest
import pandas as pd
import re
import ast
import unicodedata
from scraping_utility import match_name
import time
import pickle

def get_data(name):
    
    def collect_n_pub(a):
        #get id author
        

        id = a[1]
            
        link="https://dl.acm.org"+id
        #print("id  : ",id)
        page_num = 0
        
        req = requests.Session()
        
        
        result = req.get(link)
        
        src = result.content
        
        #print(src)
        
        
        soup = BeautifulSoup(src, "lxml")
        
        txt = soup.find_all("div", {"class","bibliometrics__block"})
        txt = str(txt)
        

        try:
            txt = re.search('Publication counts</div><div class="bibliometrics__count"><span>(.+?)</span></div></div>, <div class="bibliometrics__block">', txt).group(1)

        except AttributeError:
            pass
        
        #list_final=[a,x.abstract,a]
        
        return txt
        

    authors_list = []
    abstracts = []
    links = []
    papers_titles = []
    links_papers = []
    
    
    req = requests.Session()
    
    name_url = name.replace(" ", "+")
    
    result = req.get("https://dl.acm.org/action/doSearch?AllField="+name_url)
    
    src = result.content
    
    #print(src)
    
    
    soup = BeautifulSoup(src, "lxml")
    
    txt = soup.find_all("div", {"class","issue-item__content-right"})
    txt = str(txt)
    # pos = txt.find(name)
    
    # name_2 = strip_accents(name)
    # pos_2 = txt.find(name_2)
    authors_list_ = []
    authors_list_ = get_list_authors(txt)
    
    find_author(authors_list,name)
    a=find_author(authors_list_,name)
    
    if a != False :
        
        return collect_n_pub(a)
    else: 
        a  = find_author(authors_list_,strip_accents(name))
        if a != False :
            return collect_n_pub(a)
        else :
            return "0"
        

def find_author(authors_list,name):
    
    for a in authors_list:
        if  match_name(a[0],name) :
            return a
            break
    
    return False

def get_list_authors(txt):

    authors_list = []
    
    txt_len = len(txt)
    pos_deb = txt.find("href=\"/profile") 
    
    
    pos_fin = txt.find("><img alt=") 
    
    
    
    while pos_deb != -1 and pos_fin != -1 :
        
        text=txt[pos_deb:pos_fin]
    
        
        pos_deb_1 = text.find("/profile") 
    
    
        pos_fin_1 = text.find("\" title") 
    
    
        url=text[pos_deb_1:pos_fin_1]
    
        
        text_len = len(text) 
     
        pos_deb_1 = text.find("title") + 7
    
        name=text[pos_deb_1:(text_len-1)]
    
        authors_list.append([name,url])
        #*********************************************************#
        txt = txt[(pos_deb+text_len+6):(txt_len-1)]
    
        txt_len  = len(txt)
        
        pos_deb = txt.find("href=\"/profile") 
    
        pos_fin = txt.find("><img alt=") 
    
    
    return authors_list

def get_papers_of_author(auth_id, auth):
    
    auth_row = auth.loc[auth.id == auth_id,['pubs']]
    
    # list of dict 
    auth_papers = ast.literal_eval(auth_row.iloc[0,0])
    
    #list of papers id
    p_ids = [int(d['i']) for d in auth_papers]
    
    return p_ids
    

def get_authors_name(authors):
    
    return list(authors.name.values)
    

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')


def construct_csv(list_authors):
    
    i=1
    for a in list_authors:
        print("author : ",i)
        i=i+1
        try:
            n_pub = get_data(a)
                    
            list_final=[a,n_pub]
            with open(r'n_pub_ACM.csv', 'a', newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(list_final)

        except:
          print("An exception occurred")
        

#*******************************************/
#******************************************/

#/************************************************************************/

#   run this part of the code only the first time you create the csv file 

#/***********************************************************************/

fields=['author_name', 'n_pub']
with open(r'n_pub_ACM.csv', 'a', newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(fields)
    
#/************************************************************************/

#             Select your list of authors by modifying the indexes

#/***********************************************************************/

with open('list_all_authors.pkl', 'rb') as f:
    list_all_authors = pickle.load(f)

list_authors=list_all_authors

#/************************************************************************/

#                                   Run 

#/***********************************************************************/

construct_csv(list_authors)