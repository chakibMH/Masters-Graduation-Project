# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:26:29 2022

@author: HP
"""
import os
import time
import math
import random
import joblib
import numpy as np
import json
import pandas as pd
from sklearn.preprocessing import normalize
import faiss
from pprint import pprint
import scipy
from collections import Counter
import ast
from more_itertools import unique_everseen
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import math
from ast import literal_eval



def load_data_and_authors(data_path="papers.csv", 
                          authors_path="authors.csv"):
    data = pd.read_csv(data_path)
    authors = pd.read_csv(authors_path)
    return data, authors


def get_authors_by_id(id_):
    try:
        return data[data.id == id_].authors.values[0]
    except:
        print(id_)
        return [{"id": -999999}]

def retrieve_author_tags(author_id):
    try:
        return ast.literal_eval(authors[authors.id == author_id].tags.values[0])
    except:
        return {}

def check_if_author_relevant(author_id, query):
    query = query.lower()
    tags = [t['t'].lower() for t in retrieve_author_tags(author_id)]
    if tags:
        if query in tags:
            return True
        else:
            return False
    else:
        return "Not in the dataset or no tags present!"
 

def get_authors_by_id(id_):
    try:
        return data[data.id == id_].authors.values[0]
    except:
        print(id_)
        return [{"id": -999999}]

def calculate_distances_from_query_to_fos(query, fos_tags, tfidf_classifier=None):

    if tfidf_classifier:
        fos_tag_embeddings = tfidf_classifier.transform(fos_tags)
        query_emb = tfidf_classifier.transform([query])[0]
    else:
        fos_tag_embeddings = embedder.encode(fos_tags)
        query_emb = embedder.encode([query])[0]

    distances = [ 1- scipy.spatial.distance.cdist([query_emb], [fos_tag_embedding], 'cosine')[0][0] for fos_tag_embedding in fos_tag_embeddings]

    return [(ft, d) for ft, d in zip(fos_tags, distances)]

def check_if_author_relevant_approximate(author_id, query, similarity_threshold=0.7, tfidf=False):
    query = query.lower()
    tags = [t['t'].lower() for t in retrieve_author_tags(author_id)]
    if tfidf:
        print("tfidf")
    else:
        distances = calculate_distances_from_query_to_fos(query, tags)
    similar = [d for d in distances if d[1] > similarity_threshold]
    # print("Approx. similar:", similar)
    if similar:
        return True
    else:
        return False

def produce_authors_ranking(result):
    sortd = [(k, v) for k, v in sorted(result.items(), key=lambda item: item[1]['score'], reverse=True)]
    return sortd

def produce_authors_ranking_new(result):
    sortd = [(k, v) for k, v in sorted(result.items(), key=lambda item: item[1], reverse=True)]
    return sortd

def get_author_ranking_exact_v2(query, relvents_auths_all_queries, k=10, tfidf=False, strategy="binary",
                                normalized=False, norm_alpha=100, extra_term=10):
   
    res = relvents_auths_all_queries[query].copy()

    dic_q = res.to_dict()
    top_n = produce_authors_ranking_new(dic_q)[:k]
    
   

    relevancies = [check_if_author_relevant( aid[0], query) for aid in top_n]
    
    ranking = {}

    for rank, (author, relevancy) in enumerate(zip([a[0] for a in top_n], relevancies)):
        if author not in ranking.keys():
            ranking[author] = {"relevancy": relevancy, "rank": rank}
        else:
            continue

    return ranking



def get_author_ranking_approximate_v2(query, relvents_auths_all_queries, k=10, similarity_threshold=0.7, tfidf=False, strategy="binary",
                                      normalized=False, norm_alpha=100, extra_term=10):
    print("query : ",queries.index(query))
    
    res = relvents_auths_all_queries[query].copy()
    
    dic_q = res.to_dict()
    top_n = produce_authors_ranking_new(dic_q)[:k]
    

    relevancies = [check_if_author_relevant_approximate(aid[0], query,  similarity_threshold=0.7, tfidf=False) for aid in top_n]

    ranking = {}                                       

    for rank, (author, relevancy) in enumerate(zip([a[0] for a in top_n], relevancies)):
        if author not in ranking.keys():
            ranking[author] = {"relevancy": relevancy, "rank": rank}
        else:
            continue

    return ranking


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

def mean_precision_at_n(results, n=5):
    
    average_precision_scores = []
    
    for result in results:
        
        sortd = sorted(result.items(), key=lambda item: item[1]['rank'])
        
        correct = 0
        
        for s in sortd[:n]:
            if s[1]['relevancy'] == True:
                correct += 1
        
        average_precision_scores.append(correct / n)
    
    mpan = np.around(np.mean(average_precision_scores), decimals=3)
    
    return mpan

#*//*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*

embedder = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')


data_and_authors = load_data_and_authors()
data = data_and_authors[0]
authors = data_and_authors[1]

relvents_auths_all_queries = pd.read_csv("relvents_auths_all_queries_sum_False_new.csv",index_col=0)


          
queries = ['cluster analysis', 'Image segmentation', 'Parallel algorithm', 'Monte Carlo method',
           'Convex optimization', 'Dimensionality reduction', 'Facial recognition system', 
           'k-nearest neighbors algorithm', 'Hierarchical clustering', 'Automatic summarization',
           'Dynamic programming', 'Genetic algorithm', 'Human-computer interaction', 'Categorial grammar', 
           'Semantic Web', 'fuzzy logic', 'image restoration', 'generative model', 'search algorithm',
           'sample size determination', 'anomaly detection', 'sentiment analysis', 'semantic similarity',
           'world wide web', 'gibbs sampling', 'user interface', 'belief propagation', 'interpolation', 
           'wavelet transform', 'transfer of learning', 'topic model', 'clustering high-dimensional data', 
           'game theory', 'biometrics', 'constraint satisfaction', 'combinatorial optimization', 'speech processing',
           'multi-agent system', 'mean field theory', 'social network', 'lattice model', 'automatic image annotation',
           'computational geometry', 'Evolutionary algorithm', 'web search query', 'eye tracking', 'query optimization',
           'logic programming', 'Hyperspectral imaging', 'Bayesian statistics', 'kernel density estimation',
           'learning to rank', 'relational database', 'activity recognition', 'wearable computer', 'big data', 
           'ensemble learning', 'wordnet', 'medical imaging', 'deconvolution', 'Latent Dirichlet allocation', 
           'Euclidian distance', 'web service', 'multi-task learning', 'Linear separability', 'OWL-S',
           'Wireless sensor network', 'Semantic role labeling', 'Continuous-time Markov chain', 
           'Open Knowledge Base Connectivity', 'Propagation of uncertainty', 'Fast Fourier transform', 
           'Security token', 'Novelty detection', 'semantic grid', 'Knowledge extraction', 
           'Computational biology', 'Web 2.0', 'Network theory', 'Video denoising', 'Quantum information science',
           'Color quantization', 'social web', 'entity linking', 'information privacy', 'random forest', 
           'cloud computing', 'Knapsack problem', 'Linear algebra', 'batch processing', 'rule induction', 
           'Uncertainty quantification', 'Computer architecture', 'Best-first search', 'Gaussian random field',
           'Support vector machine', 'ontology language', 'machine translation', 'middleware', 'Newton\'s method']

start = time.time()

exact = [get_author_ranking_exact_v2(query, relvents_auths_all_queries , k=10, tfidf=False, strategy="binary", normalized=False) for query in queries]
                             

approximate = [get_author_ranking_approximate_v2(query, relvents_auths_all_queries , k=10, similarity_threshold=0.7, tfidf=False, strategy="binary", normalized=False) for query in queries]

end = time.time()
print("time: ",(end - start)/60," min")

#*******************************************************  
#                   dataframe ranking  
#*******************************************************  

df_results = pd.DataFrame(columns=["Query","Exact","Approximate"])
i=0
for q in queries:
    dict = {"Query":q,"Exact":exact[i],"Approximate":approximate[i]}
    df_results = df_results.append(dict, ignore_index = True)
    i=i+1

import pandas as pd
df_results.to_csv("new_results/Our_method/sum_False/sum_False.csv")  


#*******************************************************  
#               original_method_results.txt  
#******************************************************* 


text = "Exact binary MRR@10:"+ str(mean_reciprocal_rank(exact))+" \nApproximate binary MRR@10:"+ str(mean_reciprocal_rank(approximate))+"\nExact binary MAP@10:"+ str(mean_average_precision(exact)) +" \nApproximate binary MAP@10:"+ str(mean_average_precision(approximate))+"\nExact binary MP@5 :"+ str(mean_precision_at_n(exact, n=5))+"\nApproximate binary MP@5 :"+ str(mean_precision_at_n(approximate, n=5))+"\nExact binary MP@10 :"+ str(mean_precision_at_n(exact, n=10))+"\nApproximate binary MP@10 :"+ str(mean_precision_at_n(approximate, n=10))

with open('new_results/Our_method/sum_False/sum_False_results.txt', 'w') as f:
    f.write(text) 

#*******************************************************  
#               original_method_metrics.txt  
#*******************************************************

df_results_eval = pd.DataFrame(columns=["Query",
                                        "Exact binary MRR@10",
                                        "Approximate binary MRR@10",
                                        "Exact binary MAP@10",
                                        "Approximate binary MAP@10",
                                        "Exact binary MP@5",
                                        "Exact binary MP@10",
                                        "Approximate binary MP@5",
                                        "Approximate binary MP@10"])

i=0
for q in queries:
    l=[]
    l.append(exact[i])
    b=[]
    b.append(approximate[i])
    
    dict_ = {"Query":q,"Exact binary MRR@10":  ( 0 if math.isnan( mean_reciprocal_rank(l)) else mean_reciprocal_rank(l)),"Approximate binary MRR@10":( 0 if math.isnan(mean_reciprocal_rank(b)) else mean_reciprocal_rank(b)),"Exact binary MAP@10":( 0 if math.isnan(mean_average_precision(l)) else mean_average_precision(l)) ,"Approximate binary MAP@10":mean_average_precision(b),"Exact binary MP@5":mean_precision_at_n(l, n=5),"Exact binary MP@10":mean_precision_at_n(l, n=10),"Approximate binary MP@5":mean_precision_at_n(b, n=5),"Approximate binary MP@10":mean_precision_at_n(b, n=10)}
    df_results_eval = df_results_eval.append(dict_, ignore_index = True)
    i=i+1
    
import pandas as pd
df_results_eval.to_csv("new_results/Our_method/sum_False/sum_False_metrics.csv")  