# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 09:23:27 2022

@author: HP
"""




import pickle
import random
import requests
import wikipediaapi
import re
import string
from num2words import num2words


def remove_extra_spaces(txt):
    
    return " ".join(txt.split())


def remove_url(txt):
    
    regex_url = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"

    return re.sub(regex_url, "", txt)

def remove_email(txt):
    
    regex_email = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    return re.sub(regex_email, "", txt)

def remove_numbers(txt):
    
    regex_num = r"\([0-9]+\)"
    
    return re.sub(regex_num, "", txt)

def remove_punc(s):
    
    punc = string.punctuation + '—'+'–'+'−'+'“'+'∼'+'®'+'’'+'”'
    
    
    s_res = ''
    
    
    for c in s:
        if c == '°':
            s_res += ' degree '
        elif c in punc:
            s_res  += ' '        
        else:
            s_res += c
            
            
    return s_res


    


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

def clean_def(query):
    
    txt = wikipedia(query)
    
    txt = txt.lower()
    
    #remove the first 'abstract' word
    
    

    # treating numbers
    
    #abstract = remove_numbers(abstract)
    
    # reomve ,
    
    txt = txt.replace(",","")
    
    # replace numbers with words
    
    txt = re.sub(r"(\d+\.\d+)", lambda x:num2words(x.group(0)), txt)
    
    txt = re.sub(r"(\d+)", lambda x:num2words(x.group(0)), txt)
    
    
    
    # remove email
    
    txt = remove_email(txt)
    
    # remove url
    
    txt = remove_url(txt)
    
    txt = remove_punc(txt)
    
    txt = remove_extra_spaces(txt)
    
    return txt

def get_freq(q_l,dict_queries):
    
    new_q_l = []
    for q in q_l:
        new_q_l.append((q,dict_queries.get(q)))
    
    return new_q_l


def get_def_wikipedia(query):
    
    wiki_wiki = wikipediaapi.Wikipedia('en')
    page_py = wiki_wiki.page(query)
    
    if page_py.exists() == False:
        
        print("Invalid Input try again!{query : ",query," }")
        query="no data"
    else:
        wiki_wiki = wikipediaapi.Wikipedia('en')
        
        summary = page_py.summary.replace("\n", " ")
        summary = summary.replace("\t", " ")
        summary = summary.replace("  ", "")
        query = summary

    return query



def get_new_list(q_list):    

    new = []
    queries_def = []
    for t in q_list:
        
        query = clean_def(t)
        
        if query != t:
           new.append(t)
            
           query = query.replace(t, "", 1)
           
           q_qd = t+"@"+query
           
           queries_def.append(q_qd) 
    
    
    return new,queries_def



    
def get_list_query(dict_queries,n):  
    
    tags_List = list(dict_queries.keys())
    
    freq_List=list(dict_queries.values())
    
    Sum_freq = sum(freq_List)
    
    prob_List = []
    
    for fl in freq_List:
        
        prob_List.append(((fl*100)/Sum_freq))
        
    Sum_prob = sum(prob_List)
    
    q_list = random.choices(tags_List, weights=prob_List, k=n)
    
    q_list = list(dict.fromkeys(q_list))
    

    return q_list
    
def select_queries(n,dict_queries):
          
    
    tags_List = list(dict_queries.keys())
    
    q_list = get_list_query(dict_queries,n)
    
    q_list,queries_def =  get_new_list(q_list)
    
    new_q_list = get_freq(q_list,dict_queries)
    
    new_q_list = sorted(new_q_list, key=lambda tup: tup[1], reverse=True)
    
    queries_def = sort_queries_def(new_q_list,queries_def)

    return new_q_list,queries_def


def sort_queries_def(new_q_list,queries_def):
    
    lk = []
    
    for e in new_q_list :
        h = e[0]
        for b in queries_def:
            x = b.split("@")
            tag = x[0]
            if tag == h :
                lk.append(b)
    return lk

def get_queries_def(new_q_list):
    
    queries_def = []
    for q in new_q_list:
        q= q[0]
        deff = clean_def(q)
        
        deff = deff.replace(q, "", 1)
        
        q_qd = q+"@"+deff
        
        queries_def.append(q_qd)
    
    return queries_def
    

#/**************************************Execution***********************************/

with open("queries_freq_dict.pkl", "rb") as f:
    dict_queries = pickle.load(f)


k = dict_queries.keys()

dict_queries_new = {a:dict_queries[a] for a in k if dict_queries[a] > 5}


# k is the number of queries (it is likely that the number of real queries returned will be reduced by half).

k= 10

new_q_list ,queries_def = select_queries(k,dict_queries_new)

