# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 14:59:01 2022

@author: chaki
"""
import ast
import numpy as np
from Embedding_functions import  embedde_single_query, get_mean_embedding
import math
from custom_faiss_indexer import len_paper
# from distance_functions import dist2sim

def dist2sim(d):
    """
    Converts cosine distance into cosine similarity.

    Parameters:
    d (int): Cosine distance.

    Returns:
    sim (list of tuples): Cosine similarity.
    """
    return 1 - d / 2


# embedder = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')

# def load_relevant_index(type="separate_sbert"):
#     index = None
#     if type == "separate_sbert":
#         index = faiss.read_index("faiss_indexes/separate_embeddings_faiss.index")
#     elif type == "merged_sbert":
#         index = faiss.read_index("faiss_indexes/merged_embeddings_faiss.index")
#     elif type == "retro_merged_sbert":
#         index = faiss.read_index("faiss_indexes/retro_merged_embeddings_faiss.index")
#     elif type == "retro_separate_sbert":
#         index = faiss.read_index("faiss_indexes/retro_separate_embeddings_faiss.index")
#     elif type == "tfidf_svd":
#         index = faiss.read_index("faiss_indexes/tfidf_embeddings_faiss.index")
#     elif type == "pooled_bert":
#         index = faiss.read_index("faiss_indexes/mean_bert_faiss.index")
#     elif type == "pooled_glove":
#         index = faiss.read_index("faiss_indexes/glove_faiss.index")
#     return index

def load_custom_index(index_type):
    pass


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





    


def similarity_auth_with_paper(auths_id, paper_id, papers, authors, embedder):
    
    
    #list of doc of this papers
    
    p_ids = get_papers_of_author(auths_id, authors)
    
    # embedding du doc cible
    mean_emb_doc = get_mean_embedding(paper_id, papers, embedder)
    
    #embedding de tous les doc de cet auteur
    list_paper_embedding = []
    nb_vides = 0
    nb_non_vides = 0
    for p in p_ids:
        #tous les articles
        i = get_mean_embedding(p, papers, embedder)
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
    
    #result = 1 - spatial.distance.cosine(centroid, mean_emb_doc)
    
    result = np.linalg.norm(centroid-mean_emb_doc)
    
    return result, nb_non_vides, nb_non_vides+nb_vides


def authors_expertise_to_paper(paper_id, papers, authors, embedder):
    """
    first get all authors of the given paper id.
    then for each of them calculate distance between his expertise domain (center of his cluster) and,
    with the embedding of the paper.
    

    Parameters
    ----------
    paper_id : int
        paper of id.
    papers : Dataframe
        date set of all papers.
    authors : DataFrame
        data set of all authors.
    embedder : Sentence Embedder
        to embedde phrases of a paper.

    Returns
    -------
    dict_expertise : dict
        sim of athors of a paper.
        Each authors has a score of the given paper id

    """
    
    auths_id = get_authors_of_paper(paper_id, papers)
    
    dict_expertise = {}
    
    for a in auths_id:
        
        s, nbnv, total = similarity_auth_with_paper(a,paper_id,papers, authors, embedder)
        
        dict_expertise[a] = s
        
        print("auteur id : ",a," has sim with paper id ",paper_id," = ",s, "[",nbnv,"/",total,"]")
        
    return dict_expertise

def norm_fct(sen_index, papers, paper_id, sim_Q_D):
    """
    

    Parameters
    ----------
    sen_index : TYPE
        DESCRIPTION.
    papers : TYPE
        DESCRIPTION.
    paper_id : TYPE
        DESCRIPTION.
    sim_Q_D : TYPE
        DESCRIPTION.

    Returns
    -------
    sim_Q_D : TYPE
        DESCRIPTION.

    """
    
    # l = len_paper(sen_index, paper_id)
    
    l = len_paper_from_DB(papers, paper_id)
    
    sim_Q_D /= l
    
    return sim_Q_D

def len_paper_from_DB(papers, paper_id):
    
    paper_row = papers.loc[papers.id == paper_id,['cleaned_abstract_sentences']]
    
    list_abst = ast.literal_eval(paper_row.iloc[0,0])
    
    return len(list_abst)

    
def get_relevant_experts(query, sen_index, papers, authors, embedder, 
                         strategy = 'min', norm = False, k=1000):
    """
    

    Parameters
    ----------
    query : str
        query text.
    sen_index : custom faiss index
        index of all phrases.
    papers : Dataframe
        data set of all papers.
    authors : DataFrame
        data set of all authors.
    embedder : Sentence Embedder
        to embedde phrases of a paper.
    strategy : str, optional
        Define a strategy to give score for document based on its selected phrases.
        Available strategies:
            min strategy: score of paper is defined by the minimum score of all its phrases.
            mean strategy: score of paper is defined by the mean score of all its phrases.
            sum strategy: score of paper is defined by the sum score of all its phrases.
        The default is 'min'.
    norm: boolean, optional
        choose either we apply a normalization to the score
        The default is False.
        Available normalization:
            l: length of the document
    k : int, optional
        number of nearest neighbors (phrases) to the query to be returned. The default is 1000. 

    Returns
    -------
    score_authors_dict : dict
    Expert Id: score with query, ( sim ( A, Q) )
        Ranking of expert for this query, after calculation of expoCombSum.

    """
    
    # embedding thr query
    query_emb = embedde_single_query(query, embedder)
    
    print("searching...")
    df = sen_index.search(query_emb, k)
    print("relevant phrases extracted...")
    
    if strategy == 'min':
        df_res =  df.groupby(['paper_id'])['dist_phrase_with_query'].min()
        
    elif strategy == 'mean':
        df_res = df.groupby(['paper_id'])['dist_phrase_with_query'].mean()
    elif strategy == 'sum':
        df_res = df.groupby(['paper_id'])['dist_phrase_with_query'].sum()
    else:
        print("erreur pas d'autres strategies")
    
    print("relevant doc determined...")
    
    ids_of_sim_papers = list(df_res.index)
    
    
    #cle : id auteur
    # val : sum
    score_authors_dict = {}
    
    for p_id in ids_of_sim_papers:
        
        # waiting for scraping ... to introduce dist with auth's cluster
        # dict_expertise = authors_expertise_to_paper(p_id, papers,authors, embedder)
        
        # expo (  s[Q, D] * s[D, A] )
        
        # get authors of papre p_id
        
        auth_of_p_id = get_authors_of_paper(p_id, papers)
        
        for a in auth_of_p_id:
            
            # now is uniform 
            dist_Q_D = df_res.loc[p_id]
            
            # transform dist to sim
            sim_Q_D = dist2sim(dist_Q_D)
            
            # waiting for scraping ... to introduce dist with auth's cluster
            # dist_D_A = dict_expertise[a]
            
            # sim_D_A = dist2sim(dist_D_A)
            
            print("[Processing: ",query," ]","before normalization sim_Q_D = ",sim_Q_D)
            
            ## normalization
            
            
            if norm == True:
                sim_Q_D = norm_fct(sen_index, papers, p_id, sim_Q_D)
            
            print(" "*10,"after normalization sim_Q_D = ",sim_Q_D)
            # print(sim_D_A)
            
            # check if first time
            
            if a in score_authors_dict.keys():
                
                # score_authors_dict[a] += math.exp(sim_D_A * sim_Q_D)
                score_authors_dict[a] += math.exp(sim_Q_D)
                
            else:
                # score_authors_dict[a] = math.exp(sim_D_A * sim_Q_D)
                score_authors_dict[a] = math.exp(sim_Q_D)
                
    # sort the dict
    
    d = sorted(score_authors_dict.items(), key = lambda x:x[1],reverse=True)
    
    score_authors_dict = {e[0]:e[1] for e in d}
                
    return   score_authors_dict     
    
    
    
# save the dict

# import pickle

# with open("cluster_analysis_result","wb") as f:
#     p = pickle.Pickler(f)
#     p.dump(score_authors_dict)
    
    
    
# # to load

# with open("cluster_analysis_result", "rb") as f:
#           u = pickle.Unpickler(f)
#           result = u.load()
    
    
    
    
# query_emb = embedder.encode([query])[0]
# normalized_query = np.float32(normalize([query_emb])[0])

# assert type(normalized_query[0]).__name__ == 'float32'