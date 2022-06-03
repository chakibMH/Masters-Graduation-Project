# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 21:53:35 2022

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
import traceback


def get_data(id_a,ex_id_papers):
    
    def collect_abstracts(id_a,ex_id_papers):
        #get id author
        
            
        link_="https://dl.acm.org"+id_a
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
            link_all_papers="https://dl.acm.org"+id_a+"/publications?Role=author&pageSize=50&startPage="+str(page_num)
            
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
                
                jki=links_papers[i].replace("https://dl.acm.org/doi/", "")
                
                if jki in ex_id_papers:
                    df2 = {'id_paper':jki,'title': "Exists", 'abstract': "Exists", 'paper_citation': "Exists",'revue':"Exists",'index_terms':"Exists" }
                    df = df.append(df2, ignore_index = True)
                else:
                    # Sleep
                    t=random.uniform(0.5, 1)
                    time.sleep(t)
                    
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
    
    
    result = req.get("https://dl.acm.org/action/doSearch?AllField=jon")
    
    src = result.content
    
    #print(src)
    
    
    soup = BeautifulSoup(src, "lxml")
 
    txt2=soup.find_all("div", {"class","col-sm-6 table__cell-view vertical-center"})
    test=[]
    for i in range(len(txt2)):
        test.append(txt2[i].text)
    

    

    if test == ['\n\n\ue924\n\n', '\n\nYour IP Address has been blocked\nPlease contact \n    \n[email\xa0protected]\n\n\n\n']:
        print("@ IP blocked !")
        dfObj = pd.DataFrame(columns=['title', 'abstract'])
        return dfObj,["@ IP blocked !"],[],[]
    else:
        return collect_abstracts(id_a,ex_id_papers)
        


#*********************************/
def construct_csv(list_authors,txt_indices,ex_id_papers):
    
    i=1
    k=0
    j=0
    list_exeptions=[]
    for a in list_authors:
        
        # Sleep
        # t=random.uniform(0.5, 1)
        # time.sleep(t)
        
        print("author : ",i)
        i=i+1
        try:
            df,infos_liste,subjects,keywords = get_data(a[1],ex_id_papers)
            #print(df)
            
            if infos_liste !=[]:
                if infos_liste[0]=="@ IP blocked !" :
                    s=i-2
                    list_exeptions=list_exeptions+list_authors[s:]
                    break
                
            if df.empty == False :
                k=k+1
                
                for x in df.itertuples():
                    
                    list_final=[x.id_paper,x.title,x.abstract,x.paper_citation,x.revue,x.index_terms,a[0],infos_liste[0],infos_liste[1],infos_liste[2],infos_liste[3],infos_liste[4],subjects,keywords]
                    with open(r'Dada_Base\papers_ACM_'+txt_indices+'.csv', 'a', newline='', encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(list_final)
            else:
                j=j+1
            
                            
        except Exception as ex :
          list_exeptions.append(a)
          print("An exception occurred")
          print(traceback.format_exc())
    
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



def get_index(txt_indices,list_all_authors):
    df = pd.read_csv('Dada_Base\papers_ACM_'+txt_indices+'.csv')
    l=df.author_name
    l=l.tolist()
    b=len(l)
    return list_all_authors.index(l[b-1])




def execute(ind_start, ind_end, num_process):
    
    #/************************************************************************/

    #             Select your list of authors by modifying the indexes

    #/***********************************************************************/
    
    with open('list_1844_new_author.pkl', 'rb') as f:
        list_all_authors = pickle.load(f)
        
    with open('ex_id_papers.pkl', 'rb') as f:
        ex_id_papers = pickle.load(f)
    
        
    list_authors=list_all_authors[ind_start:ind_end]
    
    #/************************************************************************/

    #                                   Run 

    #/***********************************************************************/

    start = time.time()
    list_exeptions = construct_csv(list_authors,num_process,ex_id_papers)
    end = time.time()
    print("time: ",(end - start)/60," min")
    
    return list_exeptions


def re_execute(list_exeptions,num_process):
    
    with open('ex_id_papers.pkl', 'rb') as f:
        ex_id_papers = pickle.load(f)
    
    start = time.time()
    list_exeptions = construct_csv(list_exeptions,num_process,ex_id_papers)
    end = time.time()
    print("time: ",(end - start)/60," min")
    
    return list_exeptions


#/************************************************************************/

#   run this part of the code only the first time you create the csv file 

#/***********************************************************************/

def first_time(num_process):
    fields=['id_paper','title', 'abstract','paper_citation','revue','index_terms','author_name','author_average_citation_per_article','author_citation_count','author_publication_counts','author_publication_years','papers_available_for_download','author_subject_areas','author_keywords']
    txt_indices="02"
    with open(r'Dada_Base\papers_ACM_'+num_process+'.csv', 'a', newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)

# first_time("01")
list_exeptions=execute(0, 100, "01")

list_exeptions=execute(100, 200, "02")

list_exeptions=execute(200, 300, "03")

list_exeptions=execute(300, 400, "04")


list_exeptions=re_execute(list_exeptions,"01")