# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 14:59:01 2022

@author: chaki
"""
import ast
import numpy as np
from Embedding_functions import  embedde_single_query, get_mean_embedding, use_def_mean
import math
from custom_faiss_indexer import len_paper,load_index
from sklearn.preprocessing import normalize
from def_cleaning import clean_def
import pandas as pd
import pickle
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
    # p_ids = [int(d['i']) for d in auth_papers]
    p_ids = [d['i'] for d in auth_papers]
    
    return p_ids


def get_authors_of_paper(paper_id, papers):
    
    paper_row = papers.loc[papers.id_paper == paper_id,['authors']]
    
    #list of authors
    authors = ast.literal_eval(paper_row.iloc[0,0])
    
    
    #list of authors id
    # auths_id = [int(d['id'])  for d in authors]
    auths_id = [d['id']  for d in authors]
    
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
    # first get all authors of the given paper id.
    auths_id = get_authors_of_paper(paper_id, papers)
    
    dict_expertise = {}
    
    for a in auths_id:
        
        d, nbnv, total = similarity_auth_with_paper(a,paper_id,papers, authors, embedder)
        
        dict_expertise[a] = d
        
        print("auteur id : ",a," has sim with paper id ",paper_id," = ",d, "[",nbnv,"/",total,"]")
        
    return dict_expertise

def norm_fct(papers, paper_id, sim_Q_D):
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
    
    paper_row = papers.loc[papers.id_paper == paper_id,['cleaned_abstract_sentences']]
    
    list_abst = ast.literal_eval(paper_row.iloc[0,0])
    
    return len(list_abst)



def get_relevant_experts(query, sen_index, papers, authors, embedder, 
                         strategy = 'min', norm = False, transform_to_score_before=True
                         ,k=1000, dist_score_cluster = False):
                         #use_definition = None, data_source = "wikidata_then_wikipedia"):

    
    papers_of_expertise = {}
    # {id_author:[p1,p2,..etc]}
    papers_of_expertise = {}
    sim_D_A = 1
    # embedding the query
    
    # if use_definition is None:
    # if the query is str, then we transform to vector
    if type(query) == str:
        query_emb = embedde_single_query(query, embedder)
    else:
        # query embedded outside fucntion. type must be dict-> str : np.array
        q_name = list(query.keys())[0]
        
        query_emb = query[q_name]
        
        query = q_name
    
    print("searching...")
    df = sen_index.search(query_emb, k)
    print("relevant phrases extracted...")
        
    # elif use_definition == 'mean':
   
    #     new_query = use_def_mean(query, embedder, data_source)
        
    #     print("searching...")
    #     df = sen_index.search(new_query, k)
    #     print("relevant phrases extracted...")
    # elif use_definition == 'hybrid':#hybrid
    
    #     # fct_hybrid
    
    #     print("searching...")
    #     # df = sen_index.search(norm_query, k)
    #     print("relevant phrases extracted...")
    # else:
    #     print("erreur")
        
    
    if strategy == 'min':
        df_res =  df.groupby(['paper_id'])['dist_phrase_with_query'].min()
        # transform dist to sim
        df_res = df_res.map(lambda x: dist2sim(x))
    elif strategy == 'mean':
        
        if transform_to_score_before :
            # transform dist to sim
            df['score'] = df.dist_phrase_with_query.map(lambda x: dist2sim(x))
            df_res = df.groupby(['paper_id'])['score'].mean()
        else: # transform after
            df_res = df.groupby(['paper_id'])['dist_phrase_with_query'].mean()
            #calculate score
            df_res = df_res.map(lambda x: dist2sim(x))
        
    elif strategy == 'sum':
        if transform_to_score_before :
            # transform dist to sim
            df['score'] = df.dist_phrase_with_query.map(lambda x: dist2sim(x))
            df_res = df.groupby(['paper_id'])['score'].sum()
        else: # transform after
            df_res = df.groupby(['paper_id'])['dist_phrase_with_query'].sum()
            # calculate score
            df_res = df_res.map(lambda x: dist2sim(x))
    else:
        print("erreur pas d'autres strategies")
    
    print("relevant doc determined...")
    
    ids_of_sim_papers = list(df_res.index)
    
    
    #cle : id auteur
    # val : sum
    score_authors_dict = {}
    
    for p_id in ids_of_sim_papers:
        
        # NEW !
        if dist_score_cluster == True:
            dict_expertise = authors_expertise_to_paper(p_id, papers,authors, embedder)
        # {id_auth:Dist_Centroid_D: pour chaque auth de p_id}
        # END NEW
        
        # expo (  s[Q, D] * s[D, A] )
        
        # get authors of papre p_id
        
        auth_of_p_id = get_authors_of_paper(p_id, papers)
        
        # now is uniform 
        #dist_Q_D = df_res.loc[p_id]
        
        sim_Q_D = df_res.loc[p_id]
        
        for a in auth_of_p_id:
            

            
            # NEW !
            if dist_score_cluster == True:
                
                dist_D_A = dict_expertise[a]
                
                sim_D_A = dist2sim(dist_D_A)
                
             # END NEW  
            
            print("[Processing: ",query," ]","before normalization sim_Q_D = ",sim_Q_D)
            
            ## normalization
            
            
            if norm == True:
                sim_Q_D = norm_fct( papers, p_id, sim_Q_D)
            
            print(" "*10,"after normalization sim_Q_D = ",sim_Q_D)
            # print(sim_D_A)
            
            # check if first time
            
            if a in score_authors_dict.keys():
                
                
                # papers used in expertise
                
                papers_of_expertise[a].append(p_id)

                score_authors_dict[a] += math.exp(sim_D_A * sim_Q_D)
                # score_authors_dict[a] += math.exp(sim_Q_D)
                
            else:
                # papers used in expertise
                papers_of_expertise[a] = [p_id]
                score_authors_dict[a] = math.exp(sim_D_A * sim_Q_D)
                # score_authors_dict[a] = math.exp(sim_Q_D)
      
    # I comment this for optisation ( sort only one time outside tis fct)
    # sort the dict
    
    #d = sorted(score_authors_dict.items(), key = lambda x:x[1],reverse=True)
    
    
    #score_authors_dict = {e[0]:e[1] for e in d}
                
    return   score_authors_dict, papers_of_expertise 
    
def update_scores(final_score_authors_dict, score_authors_dict):
    """
    Chack if the authors exist in the final dict, if so then update his score
    else create new entry in the dict

    Parameters
    ----------
    final_score_authors_dict : dict
        
    score_authors_dict : dict
        

    Returns
    -------
    dict.

    """
    
    new_author_set = score_authors_dict.keys()
    final_author_set = final_score_authors_dict.keys()
    
    for a in new_author_set:
        
        if a in final_author_set:
            # update the score
            final_score_authors_dict[a] += score_authors_dict[a]
        else:
            # add for the first time
            final_score_authors_dict[a] = score_authors_dict[a]
            
    return final_score_authors_dict

def update_papers_of_expertise(final_papers_of_expertise, papers_of_expertise):
    
    already_in = list(final_papers_of_expertise.keys())
    
    for a in papers_of_expertise.keys():
        
        if a  in already_in:
            final_papers_of_expertise[a] += papers_of_expertise[a]
        else:
            final_papers_of_expertise[a] = papers_of_expertise[a]
    
    return final_papers_of_expertise
    
def get_relevant_experts_multi_index(queries, list_index_path, papers, 
                                     authors, embedder, 
                         strategy = 'min', norm = False, 
                         transform_to_score_before=True
                         ,k=1000):
    """
    

    Parameters
    ----------
    queries : list of str or np.array
        query text.
    list_index_path : list
        names of multi index.
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
    final_score_each_query = {}
    final_papers_of_expertise = {}
    
    # init the final dict
    for q in queries:
        final_score_each_query[q] = {}
    
    for f in list_index_path:
        # read the index only one time, and process all the queries with it
        sen_index = load_index(f)
        
        for q in queries:

            final_score_authors_dict = final_score_each_query[q]
                
            score_authors_dict, papers_of_expertise = get_relevant_experts(q, sen_index, papers, 
                                        authors, embedder,strategy,norm,transform_to_score_before,k)
    
            # concat dict 
            final_score_authors_dict = update_scores(final_score_authors_dict, score_authors_dict)
            final_papers_of_expertise = update_papers_of_expertise(final_papers_of_expertise, papers_of_expertise)
          
            # update the scores for this query
            final_score_each_query[q] = final_score_authors_dict
            # delete the index
            
            del(sen_index)
            
    
    
    for q in queries:
        
        final_score_authors_dict = final_score_each_query[q]
        
        # sort the dict of author, for this query
    
        d = sorted(final_score_authors_dict.items(), key = lambda x:x[1],reverse=True)
        
        
        final_score_authors_dict = {e[0]:e[1] for e in d}
        
        final_score_each_query[q] = final_score_authors_dict
    
    return final_score_each_query, final_papers_of_expertise
        
def str_to_list(list_str):
    
    def remove_extra_spaces(txt):
    
        return " ".join(txt.split())
    
    list_str = remove_extra_spaces(list_str)
    list_str = list_str.replace(" ",",")
    if list_str[1] == ',':
        list_str = list_str[:1] + list_str[2:]
     
    list_float = ast.literal_eval(list_str)
    
    return list_float


def authors_expertise_to_paper_v2(p_id, papers, profiles, loc_embs) :
    """
    
    return 

    Parameters
    ----------
    p_id : str
        DESCRIPTION.
    papers : pd.Dataframe
        DESCRIPTION.

    Returns
    -------
    bool
        DESCRIPTION.

    """
    
    dict_expertise = {}
    #read the mean embedding of the paper
    
    # search for the rigth embedding file
    

    
    # {"pid":N}
    n = loc_embs[p_id]
    # loc determined
    emb_file = "dependencies/docs_embedding/emb"+str(n)+".csv"
    #print("he*****re")
    #print(emb_file)
    emb = pd.read_csv(emb_file)
    #print("bbbbbbbbbbbbbbbbbbbbb")
    v_doc = emb.loc[emb.id_paper == p_id].mean_embedding.values[0]

    
    del(emb)
    #print("ccccccccccccccccccccccccccccc")
    v_doc = str_to_list(v_doc)
    
    v_doc = np.array(v_doc)
    
    # normalize v_doc
    v_doc = np.float32(normalize(np.array([v_doc]))[0])
    
    # calculate the distance with all authors of this paper
    
    # determining the authors for this document
    
    authors = papers.loc[papers.id_paper == p_id].authors.values[0]
    
    authors = ast.literal_eval(authors)
    
    for da in authors:
        
        
        a_name = da['name']
        
        centroid = profiles[a_name]['centroid']
        
        # normalize v_doc
        
        centroid = np.float32(normalize(np.array([centroid]))[0])
        
        #calculate the distance in [1:3]
        
        dist = np.linalg.norm(centroid-v_doc)+1
        
        # use id author ??
        
        a_id = da['id']
        
        dict_expertise[a_id] = dist
    
    
    #print("ddddddddddddddddddddddddddddddddd")
    
    return dict_expertise


def authors_expertise_to_paper_v3(p_id, papers, profiles, df_embs) :
    """
    
    return 

    Parameters
    ----------
    p_id : str
        DESCRIPTION.
    papers : pd.Dataframe
        DESCRIPTION.

    Returns
    -------
    bool
        DESCRIPTION.

    """
    
    dict_expertise = {}
    #read the mean embedding of the paper
    
    # search for the rigth embedding file

    v_doc = df_embs.loc[df_embs.id_paper == p_id].mean_embedding.values[0]

    
    #print("ccccccccccccccccccccccccccccc")
    v_doc = str_to_list(v_doc)
    
    v_doc = np.array(v_doc)
    
    # normalize v_doc
    v_doc = np.float32(normalize(np.array([v_doc]))[0])
    #print("eeeeeeeeeeeeeeeeeeeeeeeee")
    
    # calculate the distance with all authors of this paper
    
    # determining the authors for this document
    
    authors = papers.loc[papers.id_paper == p_id].authors.values[0]
    
    authors = ast.literal_eval(authors)
    
    for da in authors:
        
        
        a_name = da['name']
        
        if a_name in profiles.keys():
        
            centroid = profiles[a_name]['centroid']
        
        # normalize v_doc
        
        centroid = np.float32(normalize(np.array([centroid]))[0])
        
        #calculate the distance in [1:3]
        
        dist = np.linalg.norm(centroid-v_doc)+1
        
        # use id author ??
        
        a_id = da['id']
        
        dict_expertise[a_id] = dist
        #print("auth expertise",dict_expertise[a_id])
    
    
    #print("ddddddddddddddddddddddddddddddddd")
    
    return dict_expertise


def query_with_deff(query, deff_type, embedder, sen_index, k, a=0.7, b=0.3):
    
    papers_of_expertise = {}
    # embedding thr query
    # print("query :", query)
    l_query = query.split('@')
    # print(l_query[1])
    
    if deff_type == 'mean':
    
        concept_emb = embedde_single_query(l_query[0], embedder, False)
        
        clean_d = clean_def(l_query[1])
        
        q_def_emb = embedde_single_query(clean_d, embedder, False)
        # q_def_emb = embedde_single_query(l_query[1], embedder, False)
        
        q_mean = (q_def_emb+ concept_emb)/2
        norm_query = np.float32(normalize(q_mean))
        
        df = sen_index.search(norm_query, k)
        
    elif deff_type == 'hybrid':
        
        concept_emb_norm = embedde_single_query(l_query[0], embedder, True)
        
        # cleaning of 
        clean_d = clean_def(l_query[1])
        
        q_def_emb_norm = embedde_single_query(clean_d, embedder, True)
        
        df_concept = sen_index.search(concept_emb_norm, k)
        
        df_deff = sen_index.search(q_def_emb_norm, k)
        
        df = hybrid_phrases_score(df_concept, df_deff, a, b, k)
    else:
        print("erreur : auncune autre strategie de deffinition")
    
    return df
    
    
def get_search_result(query, sen_index, embedder, 
                         strategy = 'min', norm = False, 
                         transform_to_score_before=True
                         ,k=1000, deff_type = None):
    
    
    
        
        # embedding the query
        
        # if use_definition is None:
        # if the query is str, then we transform to vector
    
        query_emb = embedde_single_query(query, embedder)


        print("searching...")
        if deff_type == None:
            df = sen_index.search(query_emb, k)
        else:
            df = query_with_deff(query, deff_type,embedder,sen_index,k)
        print("relevant phrases extracted...")
        # print(df.shape)
        # print(df.)
            
        if strategy == 'min':
            df_res =  df.groupby(['paper_id'])['dist_phrase_with_query'].min()
            # transform dist to sim
            # print(df_res.shape)
            df_res = df_res.map(lambda x: dist2sim(x))
        elif strategy == 'mean':
        
            if transform_to_score_before :
                # transform dist to sim
                df['score'] = df.dist_phrase_with_query.map(lambda x: dist2sim(x))
                df_res = df.groupby(['paper_id'])['score'].mean()
                # print(df_res.shape)
            else: # transform after
                df_res = df.groupby(['paper_id'])['dist_phrase_with_query'].mean()
                #calculate score
                df_res = df_res.map(lambda x: dist2sim(x))
        
        elif strategy == 'sum':
                if transform_to_score_before :
                    # transform dist to sim
                    df['score'] = df.dist_phrase_with_query.map(lambda x: dist2sim(x))
                    df_res = df.groupby(['paper_id'])['score'].sum()
                else: # transform after
                    df_res = df.groupby(['paper_id'])['dist_phrase_with_query'].sum()
                    # calculate score
                    df_res = df_res.map(lambda x: dist2sim(x))
        else:
            print("erreur pas d'autres strategies")
    
        print("relevant doc determined...")
    
        return df_res



def get_scores(df_res, papers,authors, dist_score_cluster = False,
               norm = False, profiles = None, loc_embs = None, df_embs = None):
    
    print("start deteermining score...")
    
    
    ids_of_sim_papers = list(df_res.index)
    
    global not_found
    
    #cle : id auteur
    # val : sum
    score_authors_dict = {}
    papers_of_expertise={}
    sim_D_A = 1
    
    existing_ids = papers.id_paper.values
    

    di = 1
    td = len(ids_of_sim_papers)
    for p_id in ids_of_sim_papers:
        print("doc . [{}/{}]".format(di,td))
        di+=1
        #td+=1
        
        if (p_id in existing_ids) and (p_id not in df_embs.keys()):
        
            # NEW !
            if dist_score_cluster == True:
                print("*"*50)
                
                #dict_expertise = authors_expertise_to_paper_v2(p_id, papers, profiles, loc_embs)
                dict_expertise = authors_expertise_to_paper_v3(p_id, papers, profiles, df_embs)
            # {id_auth:Dist_Centroid_D: pour chaque auth de p_id}
            # END NEW
            
            # expo (  s[Q, D] * s[D, A] )
            
            # get authors of papre p_id
            # print("paper_id = ",p_id)
            
            auth_of_p_id = get_authors_of_paper(p_id, papers)
            
            # now is uniform 
            #dist_Q_D = df_res.loc[p_id]
            
            sim_Q_D = df_res.loc[p_id]
            
            ia = 1
            ta = len(auth_of_p_id)
            
            for a in auth_of_p_id:
                print(" "*10,"auth . [{}/{}]".format(ia,ta))
                ia+=1
                #ta+=1
                
                # NEW !
                if dist_score_cluster == True:
                    
                    #test if aid is in the list of ids ??
                    
                    sim_D_A = dict_expertise[a]
                    
                    #sim_D_A = dist2sim(dist_D_A)
                    
                 # END NEW  
                            
                ## normalization
                
                
                if norm == True:
                    sim_Q_D = norm_fct( papers, p_id, sim_Q_D)
                
                #print(" "*10,"after normalization sim_Q_D = ",sim_Q_D)
                # print(sim_D_A)
                
                # check if first time
                
                if a in score_authors_dict.keys():
                    
                    
                    # papers used in expertise
                    
                    papers_of_expertise[a].append(p_id)
                    #print("+"*15," added socre for this doc before : ", math.exp(sim_Q_D))
                    #print("+"*15," added socre for this doc after : ", math.exp(sim_Q_D/sim_D_A))
    
                    #print("/"*5,"  socre for this auth before : ", score_authors_dict[a])
                    score_authors_dict[a] += math.exp(sim_Q_D / sim_D_A)
                    #print("/"*5,"  socre for this auth after : ", score_authors_dict[a])
                    # score_authors_dict[a] += math.exp(sim_Q_D)
                    
                else:
                    #print("+"*15," added socre for this doc before : ", math.exp(sim_Q_D))
                    #print("+"*15," added socre for this doc after : ", math.exp(sim_Q_D/sim_D_A))
                    # papers used in expertise
                    papers_of_expertise[a] = [p_id]
                    score_authors_dict[a] = math.exp(sim_Q_D / sim_D_A)
                    # score_authors_dict[a] = math.exp(sim_Q_D)
        else:
            not_found += 1
            
      
    # I comment this for optisation ( sort only one time outside tis fct)
    # sort the dict
    
    #d = sorted(score_authors_dict.items(), key = lambda x:x[1],reverse=True)
    
    
    #score_authors_dict = {e[0]:e[1] for e in d}
                
    return   score_authors_dict, papers_of_expertise
    
# papers = papers[['id_paper', 'authors', 'list_authors_name'] ]

def get_relevant_experts_multi_index_v2(queries, list_index_path, papers, 
                                     authors, embedder, 
                         strategy = 'min', norm = False, 
                         transform_to_score_before=True
                         ,k=1000, dist_score_cluster = False, deff_type = None):
    
    
    
    
    global not_found
    not_found = 0
    
    rel_docs_each_query = {}
    final_scores = {}
    papers_of_expertise = {}
    
    # init the final dict

        
    # get results from first index separatly
    
    index_1 = load_index(list_index_path[0])
    print("searching in first index...")
    
    for q in queries:
        
        
        result = get_search_result(q,index_1,embedder,strategy,norm,
                          transform_to_score_before,k,deff_type )
        
        
        # init each query with a pandas.Serie of results from first index
        rel_docs_each_query[q] = result
    
    
    print("ddeleting first index")
    del(index_1)
    for f in list_index_path[1:]:
        
        # read the index only one time, and process all the queries with it
        print("new index loaded")
        sen_index = load_index(f)
        
        result = pd.Series()
        
        for q in queries:
            
                result = get_search_result(q,sen_index,embedder,strategy,norm,
                  transform_to_score_before,k)
                
                print("before appand ",rel_docs_each_query[q].shape)
                
                
                rel_docs_each_query[q] = rel_docs_each_query[q].append(result)
                
                print("after appand ",rel_docs_each_query[q].shape)
                
                
         # delete this index for optimization   
        del(sen_index)
        
        
    
    # sort the result and select top k
    print("got results from every index")
    if dist_score_cluster == True:
        with open("dependencies/loc_embedding.pkl", "rb") as f:
            loc_embs = pickle.load(f)
            
            
        
        # load the profil dict
        with open("dependencies/profile_file", "rb") as f:
            profiles = pickle.load(f)
   
    else:
        loc_embs = None
        profiles = None
        
    qi = 1
    qt = len(queries)
    for q in queries:
        print("query","*"*5,"[{}/{}]".format(qi,qt))
        qi+=1
        
        
        
        #rel_docs_each_query[q].sort_values(ascending=False, inplace=True)
    
        #print("shape before : ",rel_docs_each_query[q].shape)
    
        rel_docs_each_query[q] = rel_docs_each_query[q].sort_values(ascending=False).iloc[:k]
        #print("shape before : ",rel_docs_each_query[q].shape)
        # determine the embedding of each doc
        
        df_embs = read_emb(rel_docs_each_query[q], loc_embs)
    
        final_scores[q], papers_of_expertise[q] = get_scores(rel_docs_each_query[q], papers, 
                                                             authors,dist_score_cluster, 
                                                             profiles = profiles, loc_embs = loc_embs, df_embs = df_embs)
                   
        
        
    return   final_scores, papers_of_expertise
    
    
    
def read_emb(res_query, loc_file):
    
    global s_loc
    
    print("processing query...")
    s_loc={}
    for p_id in res_query.index:
        if p_id in loc_file.keys():
            nb = loc_file[p_id]
        
            if nb in s_loc.keys():
                s_loc[nb].append(p_id)
            else:
                s_loc[nb] = [p_id]

    
    df_res = pd.DataFrame(columns=['id_paper', 'mean_embedding'])
    
    for nb in s_loc.keys():
        #print("new df: ",nb)
        fn = "dependencies/docs_embedding/emb{}.csv".format(nb)
        df = pd.read_csv(fn)
        
        df_res = df_res.append(df.loc[df.id_paper.isin(s_loc[nb])])
    
    return df_res
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
    
    
    
#save the dict

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

def sim2dist(x):
    
    return (1-x)*2


def hybrid_phrases_score(df_concept, df_deff, a, b, k):
    
    df_concept['score'] = df_concept.dist_phrase_with_query.map(lambda x: dist2sim(x))
    df_deff['score'] = df_deff.dist_phrase_with_query.map(lambda x: dist2sim(x))
    min_concept = df_concept.score.min()
    min_deff = df_deff.score.min()
    
    df=pd.concat([df_concept, df_deff])
    df.drop_duplicates(['id_ph'], inplace=True)
    
    
    ids_concept = list(set(df_concept.id_ph))
    ids_deff = list(set(df_deff.id_ph))
    
    def hyb(x):
        # global df_concept
        # global df_deff
        # global  ids_concept
        # global ids_deff
        
        if x in ids_concept and x in ids_deff:
            
            s1=df_concept.loc[df_concept.id_ph == x, ['score']].iloc[0][0]
            s2=df_deff.loc[df_deff.id_ph == x, ['score']].iloc[0][0]
            return a*s1+b*s2
        elif x in ids_concept:
            s1=df_concept.loc[df_concept.id_ph == x, ['score']].iloc[0][0]
            return a*s1+b*min_deff
        else:
            s2=df_deff.loc[df_deff.id_ph == x, ['score']].iloc[0][0]
            return a*min_concept+b*s2
        
        
    
    df['score'] = df.id_ph.map(lambda x:hyb(x))
    df = df.sort_values(by=['score'], ascending = False).iloc[:k]
    # transform to dist for simplicity
    df['dist_phrase_with_query'] = df.score.map(lambda x: sim2dist(x))
    
    return df
    
    
        

    # df[]
    
    


def get_relevant_experts_WITH_DEFF(query, sen_index, papers, authors, embedder, 
                                   deff_type = 'mean', a = 0.7, b=0.3,
                         strategy = 'min', norm = False, transform_to_score_before=True
                         ,k=1000):
                          
      
    papers_of_expertise = {}
    # embedding thr query
    # print("query :", query)
    l_query = query.split('@')
    # print(l_query[1])
    
    if deff_type == 'mean':
    
        concept_emb = embedde_single_query(l_query[0], embedder, False)
        
        clean_d = clean_def(l_query[1])
        
        q_def_emb = embedde_single_query(clean_d, embedder, False)
        # q_def_emb = embedde_single_query(l_query[1], embedder, False)
        
        q_mean = (q_def_emb+ concept_emb)/2
        norm_query = np.float32(normalize(q_mean))
        
        df = sen_index.search(norm_query, k)
        
    elif deff_type == 'hybrid':
        
        concept_emb_norm = embedde_single_query(l_query[0], embedder, True)
        
        # cleaning of 
        clean_d = clean_def(l_query[1])
        
        q_def_emb_norm = embedde_single_query(clean_d, embedder, True)
        
        df_concept = sen_index.search(concept_emb_norm, k)
        
        df_deff = sen_index.search(q_def_emb_norm, k)
        
        df = hybrid_phrases_score(df_concept, df_deff, a, b, k)
        
        
    
    else:
        print("erreur")
    
        
    
    if strategy == 'min':
        df_res =  df.groupby(['paper_id'])['dist_phrase_with_query'].min()
        # transform dist to sim
        df_res = df_res.map(lambda x: dist2sim(x))
    elif strategy == 'mean':
        
        if transform_to_score_before :
            # transform dist to sim
            df['score'] = df.dist_phrase_with_query.map(lambda x: dist2sim(x))
            df_res = df.groupby(['paper_id'])['score'].mean()
        else: # transform after
            df_res = df.groupby(['paper_id'])['dist_phrase_with_query'].mean()
            #calculate score
            df_res = df_res.map(lambda x: dist2sim(x))
        
    elif strategy == 'sum':
        if transform_to_score_before :
            # transform dist to sim
            df['score'] = df.dist_phrase_with_query.map(lambda x: dist2sim(x))
            df_res = df.groupby(['paper_id'])['score'].sum()
        else: # transform after
            df_res = df.groupby(['paper_id'])['dist_phrase_with_query'].sum()
            # calculate score
            df_res = df_res.map(lambda x: dist2sim(x))
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
        
        # now is uniform 
        #dist_Q_D = df_res.loc[p_id]
        
        sim_Q_D = df_res.loc[p_id]
        
        for a in auth_of_p_id:
            

            
            # waiting for scraping ... to introduce dist with auth's cluster
            # dist_D_A = dict_expertise[a]
            
            # sim_D_A = dist2sim(dist_D_A)
            
            print("[Processing: ",query," ]","before normalization sim_Q_D = ",sim_Q_D)
            
            ## normalization
            
            
            if norm == True:
                sim_Q_D = norm_fct( papers, p_id, sim_Q_D)
            
            print(" "*10,"after normalization sim_Q_D = ",sim_Q_D)
            # print(sim_D_A)
            
            # check if first time
            
            if a in score_authors_dict.keys():
                
                papers_of_expertise[a].append(p_id)
                
                # score_authors_dict[a] += math.exp(sim_D_A * sim_Q_D)
                score_authors_dict[a] += math.exp(sim_Q_D)
                
            else:
                papers_of_expertise[a] = [p_id]
                # score_authors_dict[a] = math.exp(sim_D_A * sim_Q_D)
                score_authors_dict[a] = math.exp(sim_Q_D)
                
    # sort the dict
    
    d = sorted(score_authors_dict.items(), key = lambda x:x[1],reverse=True)
    
    score_authors_dict = {e[0]:e[1] for e in d}
                
    return   score_authors_dict, papers_of_expertise