# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 14:59:01 2022

@author: chaki
"""
import pandas as pd
import ast
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, util
from embedder import create_abstract_sentence_embeddings, calculate_average_abstract_embedding

from numpy import dot
from numpy.linalg import norm

from scipy import spatial



embedder = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')

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


def get_mean_emb_TEST(paper_id, papers):
    
        #calcul du embedding de ce doc
    # try:
        print("paper id : ",paper_id)
        df_sent = papers.loc[papers.id == paper_id, ['cleaned_abstract_sentences']]
        if df_sent.index.empty:
            #cant return None
            return np.zeros(1)
        else:
            abst_sen = df_sent.iloc[0,0]
            
            df_sent = df_sent.rename(columns={'cleaned_abstract_sentences':'abstract_sentences'})
            
            flat_sentence_embeddings = create_abstract_sentence_embeddings(df_sent, embedder)
            
            mean_emb = calculate_average_abstract_embedding(flat_sentence_embeddings)
        
            return mean_emb
    # except :
    #     print("paper id ", paper_id)


def similarity_auth_with_paper(auths_id, paper_id, papers, authors, embedder):
    
    
    #list of doc of this papers
    
    p_ids = get_papers_of_author(auths_id, authors)
    
    # embedding du doc cible
    mean_emb_doc = get_mean_emb_TEST(paper_id, papers)
    
    #embedding de tous les doc de cet auteur
    list_paper_embedding = []
    nb_vides = 0
    nb_non_vides = 0
    for p in p_ids:
        #tous les articles
        i = get_mean_emb_TEST(p, papers)
        if i.any():
            list_paper_embedding.append(i)
            nb_non_vides += 1
        else:
            nb_vides += 1
    #centroid
    
    #trasnform to no.array
    
    data = np.array(list_paper_embedding)
    
    centroid = calcul_centeroid(data)
    
    #calcul de similarite
    #cos_sim = dot(centroid, mean_emb_doc)/(norm(centroid)*norm(mean_emb_doc))
    #score = util.cos_sim()
    
    result = 1 - spatial.distance.cosine(centroid, mean_emb_doc)
    
    return result, nb_non_vides, nb_non_vides+nb_vides


def authors_expertise_to_paper(paper_id, papers, authors, embedder):
    
    auths_id = get_authors_of_paper(paper_id, papers)
    
    for a in auths_id:
        
        s, nbnv, total = similarity_auth_with_paper(a,paper_id,papers, authors, embedder)
        
        print("auteur id : ",a," has sim with paper id ",paper_id," = ",s, "[",nbnv,"/",total,"]")
    
    
    
# query_emb = embedder.encode([query])[0]
# normalized_query = np.float32(normalize([query_emb])[0])

# assert type(normalized_query[0]).__name__ == 'float32'