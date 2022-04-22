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
import time
import pickle
import random
import time


def get_data(name):
    
    def collect_abstracts(a):
        #get id author
        

        id = a[1]
            
        link_="https://dl.acm.org"+id
        
        #name = "Janez Brank"

        req1 = requests.Session()

        #name_url = name.replace(" ", "+")

        link = req1.get(link_)

        src1 = link.content

        #print(src)


        soup1 = BeautifulSoup(src1, "lxml")

        infos  = soup1.find_all("div",{"class","bibliometrics__count"})
        infos_liste= []

        for i in range(len(infos)):
            infos_liste.append(infos[i].text)

        infos_liste=infos_liste[:5]   

        #*********************************/



        infos = soup1.find_all("div",{"class","tag-cloud"})

        links=str(infos)


        # s = links
        # u=re.findall(r',"label":"(.*?)\","count":', s)

        s = links
        u1=re.findall(r'<div class="tag-cloud(.*?)\</div>,', s)
        subjects=[]
        if u1 != []:
            
            subjects=re.findall(r',"label":"(.*?)\","count":', u1[0])

        u2=re.findall(r'</div>, <div class(.*?)\></div>', s)
        keywords=[]
        if u2 != []:
            keywords=re.findall(r',"label":"(.*?)\","count":', u2[0])
            
        #///////////////////////////////////**************************/////////////////

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
            df = pd.DataFrame(columns=['id_paper','title', 'abstract', 'paper_citation','revue','index_terms'])
            for i in range(len(links_papers)):
                bbb=links_papers[i]
                result = req.get(bbb)
                id_paper=bbb.split("/doi/",1)[1]
        
                src = result.content
        
                #print(src)
        
                soup = BeautifulSoup(src, "lxml")
        
                abstract = soup.find_all("div", {"class":"abstractSection abstractInFull"})
                
                title = soup.find_all("h1",{"class":"citation__title"})
                
                infos_cit  = soup.find_all("span",{"class","citation"})
                
                
                if infos_cit != []:
                    num=infos_cit[0].text
                    try:
                        
                        left=""
                        right="citation"
                        cit_num = str( num[num.index(left)+len(left):num.index(right)])
                    except:
                        cit_num=num
                else:

                    infos_cit  = soup.find_all("div",{"class","bibliometrics__count"})
                    
                    if infos_cit != []:
                        num=infos_cit[0].text
                        cit_num=str(num)
                    else:
                        cit_num="0"
                    
                infos_cit  = soup.find_all("div",{"class","citation article__section article__index-terms"})
                index_terms=[]

                s=str(infos_cit)

                index_terms=re.findall(r'expand=all">(.*?)\</a></p>', s)
                
                revue  = soup.find_all("span",{"class","epub-section__title"})
                revue_list=[]

                for i in range(len(revue)):
                    revue_list.append(revue[i].text)
                    
                revue  = soup.find_all("span",{"class","comma-separator"})

                for i in range(len(revue)):
                    revue_list.append(revue[i].text)
                    
                revue  = soup.find_all("span",{"class","dot-separator"})

                for i in range(len(revue)):
                    revue_list.append(revue[i].text)
                
                if len(title) != 0:
                    for i in range(len(abstract)):
                        if i==0:
                            abs_ = abstract[i].text
                            abs_ = abs_.replace("\n", "")
                            
                            df2 = {'id_paper':id_paper,'title': title[i].text, 'abstract': abs_, 'paper_citation': cit_num,'revue':revue_list,'index_terms':index_terms }
                            df = df.append(df2, ignore_index = True)
            #abstracts.pop()
            
            page_num +=1
            #print("page switched")
        return df,infos_liste,subjects,keywords
        

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
            return dfObj,[],[],[]
        


#*********************************/
def construct_csv(list_authors,txt_indices):
    
    i=1
    k=0
    j=0
    list_exeptions=[]
    for a in list_authors:
        
        # Sleep
        t=random.uniform(0.5, 1)
        time.sleep(t)
        
        print("author : ",i)
        i=i+1
        try:
            df,infos_liste,subjects,keywords = get_data(a)
            #print(df)
            if df.empty == False :
                k=k+1
                
                for x in df.itertuples():
                    
                    list_final=[x.id_paper,x.title,x.abstract,x.paper_citation,x.revue,x.index_terms,a,infos_liste[0],infos_liste[1],infos_liste[2],infos_liste[3],infos_liste[4],subjects,keywords]
                    with open(r'Data_Base\papers_ACM_'+txt_indices+'.csv', 'a', newline='', encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(list_final)
            else:
                j=j+1
                        
        except:
          list_exeptions.append([i-2,a])
          print("An exception occurred")
    
    print("Number of authors found is : ",k," / ",i-1)
    
    return list_exeptions    
        


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






#/************************************************************************/

#   run this part of the code only the first time you create the csv file 

#/***********************************************************************/

fields=['id_paper','title', 'abstract','paper_citation','revue','index_terms','author_name','author_average_citation_per_article','author_citation_count','author_publication_counts','author_publication_years','papers_available_for_download','author_subject_areas','author_keywords']
txt_indices="01"
with open(r'Data_Base\papers_ACM_'+txt_indices+'.csv', 'a', newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(fields)

#/************************************************************************/

#             Select your list of authors by modifying the indexes

#/***********************************************************************/

with open('list_all_authors.pkl', 'rb') as f:
    list_all_authors = pickle.load(f)

list_authors=list_all_authors[503:600]


# list_authors=['Gerhard Lakemeyer']



#/************************************************************************/

#                                   Run 

#/***********************************************************************/

start = time.time()
txt_indices="01"
list_exeptions = construct_csv(list_authors,txt_indices)
end = time.time()
print("time: ",(end - start)/60," min")

 


# txt_indices1="2400_2499"
# list_authors1=list_all_authors[2400:2500]
# txt_indices2="2500_2599"
# list_authors2=list_all_authors[2500:2600]
# txt_indices3="2600_2699"
# list_authors3=list_all_authors[2600:2700]


# import multiprocessing


# start = time.time()
# # creating processes
# p1 = multiprocessing.Process(target=construct_csv, args=(list_authors1,txt_indices1,))
# p2 = multiprocessing.Process(target=construct_csv, args=(list_authors2,txt_indices2,))
# p3 = multiprocessing.Process(target=construct_csv, args=(list_authors3,txt_indices3,))
# # starting process 1
# p1.start()
# # starting process 2
# p2.start()
# # starting process 3
# p3.start()
  
# # wait until process 1 is finished
# p1.join()
# # wait until process 2 is finished
# p2.join()
# # wait until process 3 is finished
# p3.join()
  
# # both processes finished
# print("Done!")
# end = time.time()
# print("time: ",end - start)



# import threading
# start = time.time()
# # creating thread

# t1 = threading.Thread(target=construct_csv, args=(list_authors1,txt_indices1,))
# t2 = threading.Thread(target=construct_csv, args=(list_authors2,txt_indices2,))
# # t3 = threading.Thread(target=construct_csv, args=(list_authors3,txt_indices3,))
  
# # starting thread 1
# t1.start()
# # starting thread 2
# t2.start()
# # starting thread 3
# # t3.start()
  
# # wait until thread 1 is completely executed
# t1.join()
# # wait until thread 2 is completely executed
# t2.join()
# # wait until thread 3 is completely executed
# # t3.join()
  
# # both threads completely executed
# print("Done!")
# end = time.time()
# print("time: ",end - start)