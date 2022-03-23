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


def get_data(name):

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
    
    authors  = soup.find("div",{"class","issue-item__content-right"}).ul
    respon_text = ""
    for li in authors.find_all("li"):
        respon_text += li.text
    
    authors_list.append(respon_text)
    
    #authors= soup.find_all("div", {"class":"issue-item__content-right"})
    
    #print(authors_list[0])
    if authors_list[0].find(name) != -1 :
        
        #get id author
        
        links  = soup.find_all("div",{"class","issue-item__content-right"})
        text=str(links[0]).strip()
        left = '<a href=\"/profile'
        right = '\" title=\"'+name+'\"'
    
        id = text[text.index(left)+len(left):text.index(right)]
        link="https://dl.acm.org/profile"+id
        page_num = 0
        while True :
            link_all_papers="https://dl.acm.org/profile"+id+"/publications?Role=author&pageSize=50&startPage="+str(page_num)
            #print("id  : ",id)
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
                print("pages ended")
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
        print(df)
        return df
    else: 
        print("there is no such an author !")
        
        
    
    
get_data("Janez Brank")
