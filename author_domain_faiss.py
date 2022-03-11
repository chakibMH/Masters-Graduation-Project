# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 14:59:01 2022

@author: chaki
"""
import pandas as pd
import ast
import faiss
import numpy as np

def load_relevant_index(type="separate_sbert"):
    index = None
    if type == "separate_sbert":
        index = faiss.read_index("faiss_indexes/separate_embeddings_faiss.index")
    elif type == "merged_sbert":
        index = faiss.read_index("faiss_indexes/merged_embeddings_faiss.index")
    elif type == "retro_merged_sbert":
        index = faiss.read_index("faiss_indexes/retro_merged_embeddings_faiss.index")
    elif type == "retro_separate_sbert":
        index = faiss.read_index("faiss_indexes/retro_separate_embeddings_faiss.index")
    elif type == "tfidf_svd":
        index = faiss.read_index("faiss_indexes/tfidf_embeddings_faiss.index")
    elif type == "pooled_bert":
        index = faiss.read_index("faiss_indexes/mean_bert_faiss.index")
    elif type == "pooled_glove":
        index = faiss.read_index("faiss_indexes/glove_faiss.index")
    return index


def get_papers_of_author(auth_id, auth):
    
    auth_row = auth.loc[auth.id == auth_id,['pubs']]
    
    # list of dict 
    auth_papers = ast.literal_eval(auth_row.iloc[0,0])
    
    #list of papers id
    p_ids = [int(d['i']) for d in auth_papers]
    
    return p_ids


def get_authors_of_paper(paper_id, papers):
    
    paper_row = papers.loc[papers.id == paper_id,['authors']]
    
    #list of authors
    authors = ast.literal_eval(paper_row.iloc[0,0])
    
    
    #list of authors id
    auths_id = [int(d['id'])  for d in authors]
    
    return auths_id



def calcul_centeroid(data):
    
    nb_cols = data.shape[1]
    
    centroid = np.zeros(nb_cols)
    
    
    for col in range(nb_cols) :
        centroid[col] = np.sum(data[:,col])
        
    centroid /= data.shape[0]
        
    return centroid