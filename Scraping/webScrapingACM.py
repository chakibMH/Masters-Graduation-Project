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
import traceback


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
    
    find_author(authors_list,name)
    a=find_author(authors_list_,name)

    
    if a != False :
        
        return collect_abstracts(a)
    else: 
        a  = find_author(authors_list_,strip_accents(name))
        if a != False :
            return collect_abstracts(a)
        else :
            if test == ['\n\n\ue924\n\n', '\n\nYour IP Address has been blocked\nPlease contact \n    \n[email\xa0protected]\n\n\n\n']:
                print("@ IP blocked !")
                dfObj = pd.DataFrame(columns=['title', 'abstract'])
                return dfObj,["@ IP blocked !"],[],[]
            else:
                print("There is no such an author !")
                dfObj = pd.DataFrame(columns=['title', 'abstract'])
                return dfObj,[],[],[]
        


#*********************************/
def construct_csv(list_authors,num_process, from_recovery = False):
    """
    

    Parameters
    ----------
    list_authors : TYPE
        DESCRIPTION.
    num_process : TYPE
        DESCRIPTION.
    from_recovery : Boolean, optional
    if True then it is for recovery.
        if it is from recovery then we do not push it again. The default is False.

    Returns
    -------
    list_exeptions : TYPE
        DESCRIPTION.

    """
    
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
            
            if infos_liste !=[]:
                if infos_liste[0]=="@ IP blocked !" :
                    s=i-2
                    list_exeptions=list_exeptions+list_authors[s:]
                    sz=list_authors[s:]
                    # if it is the 2nd time that an error occured with that name we do not push it
                    for w in sz:
                        push(num_process, [w])
                    break
                
            if df.empty == False :
                k=k+1
                
                for x in df.itertuples():
                    
                    list_final=[x.id_paper,x.title,x.abstract,x.paper_citation,x.revue,x.index_terms,a,infos_liste[0],infos_liste[1],infos_liste[2],infos_liste[3],infos_liste[4],subjects,keywords]
                    with open(r'Data_Base\papers_ACM_'+num_process+'.csv', 'a', newline='', encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(list_final)
            else:
                j=j+1
            
                            
        except Exception as ex :
          list_exeptions.append(a)
          # if it is the 2nd time that an error occured with that name we do not push it
          if from_recovery == False:
              push(num_process, [a])
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



def get_index(num_process,list_all_authors):
    df = pd.read_csv('Data_Base\papers_ACM_'+num_process+'.csv')
    l=df.author_name
    l=l.tolist()
    b=len(l)
    return list_all_authors.index(l[b-1])

def execute(ind_start, ind_end, num_process, 
            file_list_author='list_all_authors.pkl'):
    """
    

    Parameters
    ----------
    ind_start : TYPE
        DESCRIPTION.
    ind_end : TYPE
        DESCRIPTION.
    num_process : TYPE
        DESCRIPTION.
    file_list_author : TYPE, optional
        DESCRIPTION. The default is 'list_all_authors.pkl'.

    Returns
    -------
    list_exeptions : TYPE
        DESCRIPTION.

    """
    #/************************************************************************/

    #             Select your list of authors by modifying the indexes
    
    #/***********************************************************************/
    
    
    #check if recovery from exception
    

    with open(file_list_author, 'rb') as f:
        list_all_authors = pickle.load(f)
    
    list_authors=list_all_authors[ind_start:ind_end]
    
    #/************************************************************************/
    
    #                                   Run 
    
    #/***********************************************************************/
    
    start = time.time()
    list_exeptions = construct_csv(list_authors,num_process)
    end = time.time()
    print("time: ",(end - start)/60," min")
    
    return list_exeptions


def recovery_from_exception(num_process, file_list_exception = "Data_Base\Exception_"):
    
    
    file_name = file_list_exception+num_process+".txt"
    
    name = pop(file_name)
    
    l = construct_csv([name], num_process, from_recovery=True)
    
    return l
    
# def pop_from_csv(file_DF):
#     """
    

#     Parameters
#     ----------
#     exp_DF : DataFrame
#         Exception file_DF (csv)..

#     Returns
#     -------
#     None.

#     """
    
#     #get fisrt element
#     first_name = file_DF.iloc[0,0]
    
#     #delete it
#     file_DF.drop(0,inplace=True)
    
#     #reset index
#     file_DF.reset_index(drop=True,inplace=True)
    
#     #save the new file again
    
#     return first_name,file_DF

def pop(file_name):
    """
    

    Parameters
    ----------
    file_name : str
        exception file.

    Returns
    -------
    first_name : str
        name of first author.
    
    -1 si vide

    """
    
    with open(file_name) as fin:
        data = [line.rstrip() for line in fin]
        
    
    #si la liste n'est vide
    if data != []:
        first_name = data[0]
        
        # save again
        
        with open(file_name, 'w', encoding="utf-8") as fout:
            fout.writelines(data[1:])
            
        return first_name
    else:
        print("pas d'exception")
        return -1



    
def push(num_process, list_names, file_list_exception = "Data_Base\Exception_"):
    """
    

    Parameters
    ----------
    num_process : str
        process (console) num.
    list_names : list
        list of names to put in exception file.
    file_list_exception : str, optional
         The default is "Data_Base\exception".

    Returns
    -------
    Boolean
    
    1 is sucess.
    
    -1 if an error occured.

    """
    
    try:
        
        # read the file
        file_name = file_list_exception+num_process+".txt"
            
        with open(file_name, encoding="utf-8") as fin:
            data = [line.rstrip() for line in fin]
            
        # add new names
        data += list_names
        
        data =[e+"\n" for e in data]
        
        print(data)
        
        # save again
        
        with open(file_name, 'w',newline='\n', encoding="utf-8") as fout:
            for n in data:
                fout.write(n)
            
        return 1
    
    except :
        
        print("Error while pushing : ")
        
        return -1
    

def setup(nb_process=4, num_first_process=1):
    """
    This function will create new files to store data in.
    
    Execute this function only when you run the program for the first time, or you want to
    create new process.
    

    Parameters
    ----------
    nb_process : int, optional
        Number of process we want to create. The default is 4.
    num_first_process : int, optional
        Num of the first process. The default is 1.

    Returns
    -------
    None.

    """
    
    fields=['id_paper','title', 'abstract','paper_citation','revue','index_terms','author_name','author_average_citation_per_article','author_citation_count','author_publication_counts','author_publication_years','papers_available_for_download','author_subject_areas','author_keywords']

    
    for i in range(nb_process):

        # num of the process to str
        num_process="0"+str(num_first_process+i)
        
        # file to store the results of scraping
        with open(r'Data_Base\papers_ACM_'+num_process+'.csv', 'a', newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            
        # file to store the results of exceptions (IP blocked, crashes)
        with open('Data_Base\Exception_'+num_process+'.txt','a', newline='', encoding="utf-8") as f:
            pass





#/************************************************************************/

#   run this part of the code only the first time you create the csv file_DF 

#/***********************************************************************/

# fields=['id_paper','title', 'abstract','paper_citation','revue','index_terms','author_name','author_average_citation_per_article','author_citation_count','author_publication_counts','author_publication_years','papers_available_for_download','author_subject_areas','author_keywords']
# num_process="01"

# with open(r'Data_Base\papers_ACM_'+num_process+'.csv', 'a', newline='', encoding="utf-8") as f:
#     writer = csv.writer(f)
#     writer.writerow(fields)
    
    
# # with open(r'Data_Base\exception'+num_process+".csv", 'a', newline='',encoding="utf-8") as f:
# #         writer = csv.writer(f)
# #       #  writer.writerow(['author_name'])

# with open('Data_Base\Exception_'+num_process+'.txt','a', newline='', encoding="utf-8") as f:
#     pass





# list_authors=list_exeptions
# list_authors.pop(0)

# k=get_index("04",list_all_authors)


# df = pd.read_csv('Data_Base\papers_ACM_04.csv')

# l=df.author_name
# l=l.tolist()

# l= list(dict.fromkeys(l))


# m=list_all_authors[970:1200]

# n=list(set(m) - set(l))
# list_authors=n


# num_process1="2400_2499"
# list_authors1=list_all_authors[2400:2500]
# num_process2="2500_2599"
# list_authors2=list_all_authors[2500:2600]
# num_process3="2600_2699"
# list_authors3=list_all_authors[2600:2700]


# import multiprocessing


# start = time.time()
# # creating processes
# p1 = multiprocessing.Process(target=construct_csv, args=(list_authors1,num_process1,))
# p2 = multiprocessing.Process(target=construct_csv, args=(list_authors2,num_process2,))
# p3 = multiprocessing.Process(target=construct_csv, args=(list_authors3,num_process3,))
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

# t1 = threading.Thread(target=construct_csv, args=(list_authors1,num_process1,))
# t2 = threading.Thread(target=construct_csv, args=(list_authors2,num_process2,))
# # t3 = threading.Thread(target=construct_csv, args=(list_authors3,num_process3,))
  
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