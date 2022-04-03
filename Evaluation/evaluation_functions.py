# -*- coding: utf-8 -*-
import pandas as pd
import ast
import scipy
from sentence_transformers import SentenceTransformer
import numpy as np




def retrieve_author_tags(authors, author_id):
    """
    

    Parameters
    ----------
    authors : Dataframe
        authors' dataset.
    author_id : int
        id of author.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    try:
        return ast.literal_eval(authors[authors.id == author_id].tags.values[0])
    except:
        return {}
    
    
def check_if_author_relevant(authors, author_id, query):
    query = query.lower()
    tags = [t['t'].lower() for t in retrieve_author_tags(authors, author_id)]
    if tags:
        if query in tags:
            return True
        else:
            return False
    else:
        return "Not in the dataset or no tags present!"
    
    
##################
##################
##################
##################    APROXIMATE
##################
##################



def check_if_author_relevant_approximate(author_id, query, embedder, similarity_threshold=0.7, tfidf=False):
    query = query.lower()
    tags = [t['t'].lower() for t in retrieve_author_tags(author_id)]
    if tfidf:
        print("tfidf")
    else:
        distances = calculate_distances_from_query_to_fos(query, tags,embedder)
    similar = [d for d in distances if d[1] > similarity_threshold]
    # can we use the length of the list ? 
    # print("Approx. similar:", similar)
    if similar:
        return True
    else:
        return False
    
    
def calculate_distances_from_query_to_fos(query, fos_tags,embedder, tfidf_classifier=None):

    if tfidf_classifier:
        fos_tag_embeddings = tfidf_classifier.transform(fos_tags)
        query_emb = tfidf_classifier.transform([query])[0]
    else:
        fos_tag_embeddings = embedder.encode(fos_tags)
        query_emb = embedder.encode([query])[0]

    distances = [ 1- scipy.spatial.distance.cdist([query_emb], [fos_tag_embedding], 'cosine')[0][0] for fos_tag_embedding in fos_tag_embeddings]

    return [(ft, d) for ft, d in zip(fos_tags, distances)]


####################################
##################
##################
##################

##################


def produce_authors_ranking_new(result):
    sortd = [(k, v) for k, v in sorted(result.items(), key=lambda item: item[1], reverse=True)]
    return sortd

def get_author_ranking_exact_v2(query1,relvents_auths_all_queries, authors, k=50,  strategy="uniform",
                                normalized=False, norm_alpha=100, extra_term=10):
    """
    Produces an author ranking given a query and adds relevancy flag to the author
    based on the exact topic evaluation criteria. Used for evaluating the system.
    
    Parameters:
    query (string): The search query
    relvents_auths_all_queries: data frame of results
    authors: dataset authors
    
    k (int): The amount of authors to retrieve
    strategy (string): The data fusion strategy used for assigning author score per paper (clustering our case)
    normalized (bool): Whether normalization should be applied to the scores, boosting less prolific
    authors and "punishing" highly prolific authors
    norm_alpha (int or float): The inverse strength of normalization (higher alpha means less normalization)
    extra_term (int): Extra normalization damping term, further reduces normalization effect
    
    Returns:
    ranking (dict): A mapping of authors to their retrieved rank and their 
    relevancy in relation to the query
    """


    

    res = relvents_auths_all_queries[query1].copy()
    # sort values
    #res.sort_values(inplace=True)
    # dict like cluster analysis' one
    dic_q = res.to_dict()
    result_top_k = produce_authors_ranking_new(dic_q)[:k]
    
    top_n = result_top_k

    relevancies = [check_if_author_relevant(authors, aid[0], query1) for aid in top_n]

    ranking = {}

    for rank, (author, relevancy) in enumerate(zip([a[0] for a in top_n], relevancies)):
        if author not in ranking.keys():
            ranking[author] = {"relevancy": relevancy, "rank": rank}
        else:
            continue

    return ranking


def get_author_ranking_approximate_v2(query1,  relvents_auths_all_queries, authors, embedder,k=50, similarity_threshold=0.7, strategy="uniform",
                                      normalized=False, norm_alpha=100, extra_term=10):
    """
    Produces an author ranking given a query and adds relevancy flag to the author
    based on the approximate topic evaluation criteria. Used for evaluating the system.
    
    Parameters:
    query (string): The search query
    index (obj): The loaded FAISS index populated by paper embeddings
    k (int): The amount of authors to retrieve
    similarity_threshold (float): The approximate topic query similarity threshold
    tfidf (bool): Whether the tf-idf embeddings are used for retrieval instead of SBERT.
    strategy (string): The data fusion strategy used for assigning author score per paper
    normalized (bool): Whether normalization should be applied to the scores, boosting less prolific
    authors and "punishing" highly prolific authors
    norm_alpha (int or float): The inverse strength of normalization (higher alpha means less normalization)
    extra_term (int): Extra normalization damping term, further reduces normalization effect
    
    Returns:
    ranking (dict): A mapping of authors to their retrieved rank and their 
    relevancy in relation to the query
    """

    

    
    
    res = relvents_auths_all_queries[query1].copy()
    # sort values
    #res.sort_values(inplace=True)
    # dict like cluster analysis' one
    dic_q = res.to_dict()
    result_top_k = produce_authors_ranking_new(dic_q)[:k]
    
    top_n = result_top_k


    relevancies = [check_if_author_relevant_approximate(aid[0], query1, embedder, similarity_threshold) for aid in top_n]

    ranking = {}

    for rank, (author, relevancy) in enumerate(zip([a[0] for a in top_n], relevancies)):
        if author not in ranking.keys():
            ranking[author] = {"relevancy": relevancy, "rank": rank}
        else:
            continue

    return ranking









############## METRICS




def mean_reciprocal_rank(results):
    partial_ranks = []
    
    for result in results:
        sortd = sorted(result.items(), key=lambda item: item[1]['rank'])

        for s in sortd:
            if s[1]['relevancy'] == True:
                # We had to do rank from 1 on instead of 0 on because of the 1 / rank formula.
                partial_ranks.append(1 / (s[1]['rank']+1))
                break
    
    mrr = np.around(np.mean(partial_ranks), decimals=3)
    
    return mrr

def mean_average_precision(results):
    
    average_precision_scores = []
    
    for result in results:
        sortd = sorted(result.items(), key=lambda item: item[1]['rank'])
        
        average_precison_partials_list = []
        current_sublist_size = 0
        relevant_found = 0
        
        for s in sortd:
            if s[1]['relevancy'] == True:
                current_sublist_size += 1
                relevant_found += 1
                average_precision_partial = relevant_found / current_sublist_size
                average_precison_partials_list.append(average_precision_partial)
            else:
                current_sublist_size += 1

        average_precision = np.sum(average_precison_partials_list) / len(sortd)
        average_precision_scores.append(average_precision)
    
    mapr = np.around(np.mean(average_precision_scores), decimals=3)
    
    return mapr

def mean_precision_at_n(results, list_n=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]):
    
    average_precision_scores = []
    
    d_n = {}
    
    for n in list_n:
    
        for result in results:
        
            sortd = sorted(result.items(), key=lambda item: item[1]['rank'])
            
            correct = 0
            
            for s in sortd[:n]:
                if s[1]['relevancy'] == True:
                    correct += 1
            
            average_precision_scores.append(correct / n)
    
        mpan = np.around(np.mean(average_precision_scores), decimals=3)
            
        d_n[n] = mpan
    
    return d_n


