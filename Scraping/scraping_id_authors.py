# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 18:06:36 2022

@author: HP
"""

import requests
from bs4 import BeautifulSoup
import re
import time
from webScrapingACM_old import find_author,get_list_authors




# output: /profile/81100154382
def get_id(name):

    
    req = requests.Session()
    
    name_url = name.replace(" ", "+")
    
    result = req.get("https://dl.acm.org/action/doSearch?AllField="+name_url)
    
    src = result.content
    
    #print(src)
    
    
    soup = BeautifulSoup(src, "lxml")
    
    txt = soup.find_all("div", {"class","issue-item__content-right"})
    txt = str(txt)
    # pos = txt.find(name)
    txt2=soup.find_all("div", {"class","col-sm-6 table__cell-view vertical-center"})
    test=[]
    for i in range(len(txt2)):
        test.append(txt2[i].text)
    
    
    # name_2 = strip_accents(name)
    # pos_2 = txt.find(name_2)
    authors_list_ = []
    authors_list_ = get_list_authors(txt)
    
    a=find_author(authors_list_,name)

    return a
    


# output : [[id, name]]
def get_all_authors_ids(list_authors):
    
    list_author_id = []
    i=0
    for a in list_authors:
        print(" Author : ",i+1)
        i=i+1
        list_author_id.append(get_id(a))
        
    return list_author_id

# output : [c1,c2,...cn]

def get_colleagues(id_au, n=5):

    req = requests.Session()
    
    result = req.get("https://dl.acm.org"+id_au+"/colleagues")
    
    src = result.content
    
    soup = BeautifulSoup(src, "lxml")
    
    try:
    
        authors  = soup.find("ul",{"rlist results-list"})
        respon_text = ""
        for li in authors.find_all("li"):
            respon_text += li.text
        
        respon_text=" ".join(respon_text.split())
        
        txt_names = re.sub('\d', '', respon_text)
         
        txt_names = txt_names.replace("  Papercounts ", ",")
        
        txt_names = txt_names.replace("  Papercounts", "") 
        
        l_names = txt_names.split(",")
    except:
        authors  = soup.find("div",{"col-lg-9 col-sm-8 sticko__side-content"})
        txt=authors.text
        txt=" ".join(txt.split())
        print(txt)
        return []
    
    size = len(l_names)
    
    if size >= 5 :
        l_names = l_names[:5]
    
    return l_names



#  output : [[name,id,[c1,c2,...cn]]]

def get_all_authors_colleagues(list_authors,n=5):
    
    print("/*************  get ids ***************/")
    list_author_id = get_all_authors_ids(list_authors)
    print("/*************  get colleagues ***************/")
    data = []
    i=0
    for a in list_author_id :
        print("Author : ",i+1)
        data.append([a[0],a[1], get_colleagues(str(a[1]),5)])
        i=i+1
        
    return data
    


list_authors=["Laura Dietz","Kristina A Keesom"]

def execut(list_authors,n):
    start = time.time()
    ll=get_all_authors_colleagues(list_authors,n)
    end = time.time()
    print("time: ",(end - start)/60," min")
    
    return ll

list_authors = list_top_500

ll=execut(list_authors,5)




import pickle
with open('top_500_data.pkl', 'wb') as f:
   pickle.dump(data, f)

with open('top_500_data.pkl', 'rb') as f:
   data_ = pickle.load(f)