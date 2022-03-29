# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 20:58:39 2022

@author: HP
"""
from bs4 import BeautifulSoup
import requests

papers_name = []

papers_title = []
papers_abstract = []
all_papers = []

headers = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.1.1 Safari/605.1.15'}

def get_data(name):
    #rearch for author
    name_url = name.replace(" ", "+")
    url ='https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q='+name_url+'&btnG='
    
    response = requests.get(url,headers=headers)
    
    soup=BeautifulSoup(response.content,'lxml')
    
    #print(soup)
    #get url author (id)
    
    author_name = soup.find_all("h4", {"class":"gs_rt2"})
    
    text = str(author_name)
    try:
        # pos = text.find(name)
        # #print(pos)
        
        # pos_deb=pos-70
        # pos_fin=pos+70
        # text=text[pos_deb:pos_fin]
        
        #print(text)
        
        left = 'href=\"'
        right = '\"><b>'
        
        
        url_ = text[text.index(left)+len(left):text.index(right)]
        
        #print(url_)
        
        url_ = "https://scholar.google.com/"+url_
        
        #print(url_)
        
        #get papers
        
        url = url_
        
        response = requests.get(url,headers=headers)
        
        soup=BeautifulSoup(response.content,'lxml')
        
        #print(soup)
        #get url author (id)
        
        papers = soup.find_all("a", {"class":"gsc_a_at"})
        
        
        
        for i in range(len(papers)):
            left = 'href=\"'
            right = '\">'
            text = str(papers[i])
            url_ = text[text.index(left)+len(left):text.index(right)]
            url_ = url_.replace("amp;","")
            url_ = "https://scholar.google.com"+url_
            papers_name.append(url_)
            
            
            
        #print(papers_name)
        
        for p in papers_name:
            response = requests.get(p,headers=headers)
        
            soup=BeautifulSoup(response.content,'lxml')
        
        
            title = soup.find_all("a", {"class":"gsc_oci_title_link"})
            abstract = soup.find_all("div", {"class":"gsh_small"})
        
        
            if (len(title) != 0) and (len(abstract) != 0):
                for i in range(len(title)):
                    
                    all_papers.append([title[i].text,abstract[i].text])
        
        return all_papers
    except Exception as e:
        print("there is no such an author !")
        

all_papers=get_data("Zhang Liu")
print(all_papers)
