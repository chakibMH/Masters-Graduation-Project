# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 20:41:28 2022
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



def get_data(name):
    
    def collect_abstracts(a):
        #get id author
        

        id = a[1]
            
        link="https://dl.acm.org"+id
        #print("id  : ",id)
        page_num = 0
        while True :
            link_all_papers="https://dl.acm.org"+id+"/publications?Role=author&pageSize=50&startPage="+str(page_num)
            
            #get all papers
            
            result = requests.get(link_all_papers)
            src = result.content
            soup = BeautifulSoup(src, "lxml")
            
            num = soup.find_all("span",{"class":"result__count"})
            
            num =str(num[0])
            
            left="<span class=\"result__count\">"
            right=" Results</span>"
            
            page_limit = int( num[num.index(left)+len(left):num.index(right)])
            #print(page_limit)
            
            if (page_num > page_limit // 50):
                #print("pages ended")
                break
            
            papers_title = soup.find_all("h5", {"class":"issue-item__title"})
        
            #print(papers_title)
            
            for i in range(len(papers_title)):
                papers_titles.append(papers_title[i].text)
                
                links_papers.append("https://dl.acm.org"+papers_title[i].find("a").attrs["href"])
                
                #dfObj = pd.DataFrame(columns=['paper_title', 'url', 'Action'])
            
            #print(len(papers_titles))  
            #print(len(links_papers))
            
            #get abstracts
            df = pd.DataFrame(columns=['title', 'abstract'])
            for i in range(len(links_papers)):
                result = req.get(links_papers[i])
        
                src = result.content
        
                #print(src)
        
                soup = BeautifulSoup(src, "lxml")
        
                abstract = soup.find_all("div", {"class":"abstractSection abstractInFull"})
                title = soup.find_all("h1",{"class":"citation__title"})
                for i in range(len(abstract)):
                    if i==0:
                        abs_ = abstract[i].text
                        abs_ = abs_.replace("\n", "")
                        
                        df2 = {'title': title[i].text, 'abstract': abs_}
                        df = df.append(df2, ignore_index = True)
            #abstracts.pop()
            
            page_num +=1
            #print("page switched")
        return df
        

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
        
        return collect_abstracts(a)
    else: 
        a  = find_author(authors_list_,strip_accents(name))
        if a != False :
            return collect_abstracts(a)
        else :
            print("there is no such an author !")
            dfObj = pd.DataFrame(columns=['title', 'abstract'])
            return dfObj
        
def construct_csv(list_authors):
    df_final = pd.DataFrame(columns=['author', 'papers'])
    i=1
    for a in list_authors:
        print("author : ",i)
        i=i+1
        df = get_data(a)
        #print(df)
        list_papers = []
        if df.empty == False :
            
            for x in df.itertuples():
                list_papers.append([x.title,x.abstract])
            
            if len(list_papers)==0:
                list_papers.append("No data available")
            
        df_ = {'author': a, 'papers': list_papers}
        df_final = df_final.append(df_, ignore_index = True)
        
    #print(df_final)
    df_final.to_csv("authors_data_ACM.csv", encoding='utf-8',index=False)
    return df_final

def get_real_npubs(authors, papers):
    
    auths = authors.id.values
    
    all_p_ids = papers.id.values
    
    real_number = []
    i=0
    for auth_id in auths:
        i+=1
        #print(i)
        
        p_ids = get_papers_of_author(auth_id, authors)
        
        cpt = 0
        
        for p in p_ids:
            
            if p in all_p_ids:
                cpt += 1
        
        real_number.append(cpt)
        
    
    df_final = pd.DataFrame({'author':auths, 'papers': real_number})
    
    return df_final
        
        
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
#exemple
# #["Janez Brank", "Hoda Heidari","Eden Chlamtáč"]
# list_authors = ["Janez Brank", "Hoda Heidari","A.S.L.O Campanharo"]
# df = construct_csv(list_authors)

df_all_authors = pd.read_csv ('authors.csv')
all_names = df_all_authors.name
list_authors = all_names.head(20)
df = construct_csv(list_authors)

