# import os
import time
import math
# import random
# import joblib
import numpy as np
# import json
import pandas as pd
# from sklearn.preprocessing import normalize
# import faiss
# from pprint import pprint
import scipy
# from collections import Counter
import ast
# from more_itertools import unique_everseen
from sentence_transformers import SentenceTransformer
# from collections import defaultdict
# import math
# from ast import literal_eval

#


def load_data_and_authors(data_path="papers.csv", 
                          authors_path="authors.csv"):
    data = pd.read_csv(data_path)
    authors = pd.read_csv(authors_path)
    return data, authors




def retrieve_author_tags_new(au_id):
  
    #author_id= int(author_id)
    try:
        return ast.literal_eval(authors[authors.id == au_id].tags.values[0])
    except:
        return {}

   

def check_if_author_relevant_new(author_id, query):
    query = query.lower()
    tags = [t.lower() for t in retrieve_author_tags(author_id)]
    if len(tags) > 1:
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
    
    

def check_if_author_relevant_approximate_new(author_id, query, similarity_threshold=0.7, tfidf=False):
    query = query.lower()
    tags = [t.lower() for t in retrieve_author_tags(author_id)]
    if tfidf:
        print("tfidf")
    else:
        if tags[0] == "no tags" :
            tags = []
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




def get_author_ranking_exact_v2_from_csv_without_def(query, relvents_auths_all_queries, k=10, tfidf=False, strategy="binary",
                                normalized=False, norm_alpha=100, extra_term=10):
   
    res = relvents_auths_all_queries[query].copy()
    
    res = res.dropna()
    
    dic_q = res.to_dict()
    
    top_n = produce_authors_ranking_new(dic_q)[:k]
    

    relevancies = [check_if_author_relevant_new( aid[0], query) for aid in top_n]
    
    ranking = {}

    for rank, (author, relevancy) in enumerate(zip([a[0] for a in top_n], relevancies)):
        if author not in ranking.keys():
            ranking[author] = {"relevancy": relevancy, "rank": rank}
        else:
            continue

    return ranking



def get_author_ranking_approximate_v2_from_csv_without_def(query, relvents_auths_all_queries, k=10, similarity_threshold=0.7, tfidf=False, strategy="binary",
                                      normalized=False, norm_alpha=100, extra_term=10):
    print("query : ",queries2.index(query))
    
 
    res = relvents_auths_all_queries[query].copy()
    
    res = res.dropna()
    
    dic_q = res.to_dict()
    
    top_n = produce_authors_ranking_new(dic_q)[:k]
    

    relevancies = [check_if_author_relevant_approximate_new(aid[0], query,  similarity_threshold=0.7, tfidf=False) for aid in top_n]

    ranking = {}                                       

    for rank, (author, relevancy) in enumerate(zip([a[0] for a in top_n], relevancies)):
        if author not in ranking.keys():
            ranking[author] = {"relevancy": relevancy, "rank": rank}
        else:
            continue

    return ranking


#***********************************************************************************
def get_author_ranking_exact_v2_from_csv(query,deff, relvents_auths_all_queries, k=10, tfidf=False, strategy="binary",
                                normalized=False, norm_alpha=100, extra_term=10):
   
    res = relvents_auths_all_queries[deff].copy()
    
    res = res.dropna()
    
    dic_q = res.to_dict()
    
    top_n = produce_authors_ranking_new(dic_q)[:k]
    

    relevancies = [check_if_author_relevant_new( aid[0], query) for aid in top_n]
    
    ranking = {}

    for rank, (author, relevancy) in enumerate(zip([a[0] for a in top_n], relevancies)):
        if author not in ranking.keys():
            ranking[author] = {"relevancy": relevancy, "rank": rank}
        else:
            continue

    return ranking



def get_author_ranking_approximate_v2_from_csv(query,deff, relvents_auths_all_queries, k=10, similarity_threshold=0.7, tfidf=False, strategy="binary",
                                      normalized=False, norm_alpha=100, extra_term=10):
    print("query : ",queries2.index(query))
    
 
    res = relvents_auths_all_queries[deff].copy()
    
    res = res.dropna()
    
    dic_q = res.to_dict()
    
    top_n = produce_authors_ranking_new(dic_q)[:k]
    

    relevancies = [check_if_author_relevant_approximate_new(aid[0], query,  similarity_threshold=0.7, tfidf=False) for aid in top_n]

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

# embedder = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')


# data_and_authors = load_data_and_authors()
# data = data_and_authors[0]
# authors = data_and_authors[1]


# #load papers and authors
# data = pd.read_csv("papers.csv")
# authors = pd.read_csv("authors.csv")

# #load embedder
# embedder = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')


def execute(file_name):

    relvents_auths_all_queries = pd.read_csv(file_name+"_new.csv",index_col=0)
    
    
              
    
    
    
    start = time.time()
    
    exact = [get_author_ranking_exact_v2_from_csv(query,deff, relvents_auths_all_queries , k=10, tfidf=False, strategy="binary", normalized=False) for query,deff in zip(queries2,queries)]
                                 
    
    approximate = [get_author_ranking_approximate_v2_from_csv(query,deff, relvents_auths_all_queries , k=10, similarity_threshold=0.7, tfidf=False, strategy="binary", normalized=False) for query,deff in zip(queries2,queries)]
    
    end = time.time()
    print("time: ",(end - start)/60," min")
    
    return exact, approximate

###############################################################################

def execute_without_def(file_name):

    relvents_auths_all_queries = pd.read_csv(file_name+"_new.csv",index_col=0)
    
    
              
    
    
    
    start = time.time()
    
    exact = [get_author_ranking_exact_v2_from_csv_without_def(query, relvents_auths_all_queries , k=10, tfidf=False, strategy="binary", normalized=False) for query in queries2]
                                 
    
    approximate = [get_author_ranking_approximate_v2_from_csv_without_def(query, relvents_auths_all_queries , k=10, similarity_threshold=0.7, tfidf=False, strategy="binary", normalized=False) for query in queries2]
    
    end = time.time()
    print("time: ",(end - start)/60," min")
    
    return exact, approximate



# those function were in our_method_script, names were very confusing, and the execution was complicated

# def get_relevant_authors(file_name,strategy = 'sum',norm=True, deff_type="mean",a = 0.7, b=0.3, transform_to_score_before=True):

#     results_all_queries = pd.DataFrame()
    
#     d_all_query = {}
#     i=1
#     l = len(queries)
#     for q in queries:
#         print('current query: ',q,' [{}/{}'.format(i,l)) 
#         # score_authors_dict = get_relevant_experts(q, sen_index, data, 
#         #                                           authors, embedder, strategy,norm,
#         #                                           transform_to_score_before, 2000)
        
#         score_authors_dict, papers_of_expertise = get_relevant_experts_WITH_DEFF(q, sen_index, data, authors, embedder, 
#                                            deff_type,a , b,strategy, norm ,  transform_to_score_before)
        
        
#         d_all_query[q] = score_authors_dict
#         i+=1
#     df = pd.DataFrame(d_all_query)
#     all_authors = authors.id.values
#     l = list(df.index)
#     # to_drop = []
#     # for i in l:
#     #     if int(i) not in all_authors:
#     #         # to drop
#     #         to_drop.append(i)
    
#     # df.drop(to_drop, inplace=True)
#     df.to_csv(file_name+".csv")
    

    
    
#     relvents_auths = pd.read_csv(file_name+".csv")
    
#     relvents_auths = relvents_auths.rename(columns={relvents_auths.columns[0]: 'id'})
    
#     list_ids_relevant=relvents_auths["id"].tolist()
    
    
#     def retrieve_author_tags_new(authors, author_id):
      
#         author_id= int(author_id)
#         try:
#             return ast.literal_eval(authors[authors.id == author_id].tags.values[0])
#         except:
#             return {}
    
    
#     for n in list_ids_relevant:
#         tags = [t['t'].lower() for t in retrieve_author_tags_new(authors,n)]
#         #print("* tags  : ",tags,"// id : ",n)
        
#         if tags:
#             b=1
#         else:
#             print(n)
#             relvents_auths.drop(relvents_auths.index[relvents_auths['id'] == n], inplace=True)
    
#     relvents_auths.to_csv(file_name+'_new.csv', index=False)
    
    
    
########################################################


def get_relevant_authors(list_index_path,file_name,strategy = 'min',norm=False, deff_type = None, k=1000):

    results_all_queries = pd.DataFrame()
    
    d_all_query = {}
    i=1
    l = len(queries2)
    # for q in queries2:
    #     print('current query: ',q,' [{}/{}'.format(i,l)) 
    #     # score_authors_dict = get_relevant_experts(q, sen_index, data, 
    #     #                                           authors, embedder, strategy,norm,
    #     #                                           transform_to_score_before, 2000)
        
    #     # score_authors_dict = get_relevant_experts(q, sen_index, data, authors, embedder
    #     #                                           ,strategy, norm ,  transform_to_score_before)
    #     score_authors_dict, papers_of_expertise = get_relevant_experts_multi_index_v2(q, list_index_path, data, 
    #                                          authors, embedder, 
    #                              strategy = 'min', norm = False, 
    #                              transform_to_score_before=True
    #                              ,k=1000, dist_score_cluster = False)
        
    #     d_all_query[q] = score_authors_dict
    #     i+=1
        
    score_authors_dict, papers_of_expertise = get_relevant_experts_multi_index_v2(queries2, list_index_path, data, 
                                          authors, embedder, 
                              strategy = 'min', norm = False, 
                              transform_to_score_before=True
                              ,k=1000, dist_score_cluster = False, deff_type = None)
        
    # d_all_query =final_scores
    d_all_query = score_authors_dict
    df = pd.DataFrame(d_all_query)
    all_authors = authors.id.values
    l = list(df.index)
    # to_drop = []
    # for i in l:
    #     if int(i) not in all_authors:
    #         # to drop
    #         to_drop.append(i)
    
    # df.drop(to_drop, inplace=True)
    df.to_csv(file_name+".csv")
    

    
    
    relvents_auths = pd.read_csv(file_name+".csv")
    
    relvents_auths = relvents_auths.rename(columns={relvents_auths.columns[0]: 'id'})
    
    list_ids_relevant=relvents_auths["id"].tolist()
    
    
    def retrieve_author_tags_new(authors, au_id):
      
        #author_id= int(author_id)
        try:
            return ast.literal_eval(authors[authors.author_id == au_id].tags.values[0])
        except:
            return {}
    
    
    for n in list_ids_relevant:
        tags = [t.lower() for t in retrieve_author_tags_new(authors,n)]
        #print("* tags  : ",tags,"// id : ",n)
        
        if tags[0] == "no tags":
            print("id : ",n)
            relvents_auths.drop(relvents_auths.index[relvents_auths['id'] == n], inplace=True)
            
    
    relvents_auths.to_csv(file_name+'_new.csv', index=False)
    




#/*/*/*/*/*/*/*/*/

# def get_relevant_authors_without_def(file_name,strategy = 'sum',norm=True, transform_to_score_before=True):

#     results_all_queries = pd.DataFrame()
    
#     d_all_query = {}
#     i=1
#     l = len(queries2)
#     for q in queries2:
#         print('current query: ',q,' [{}/{}'.format(i,l)) 
#         # score_authors_dict = get_relevant_experts(q, sen_index, data, 
#         #                                           authors, embedder, strategy,norm,
#         #                                           transform_to_score_before, 2000)
        
#         score_authors_dict, papers_of_expertise = get_relevant_experts(q, sen_index, data, authors, embedder
#                                                   ,strategy, norm ,  transform_to_score_before)
        
        
#         d_all_query[q] = score_authors_dict
#         i+=1
#     d_all_query = final_scores
#     df = pd.DataFrame(d_all_query)
#     all_authors = list(authors.id.values)
#     l = list(df.index)
#     to_drop = []
#     for i in l:
#         if i not in all_authors:
#             # to drop
#             to_drop.append(i)
    
#     df.drop(to_drop, inplace=True)
#     df.to_csv(file_name+".csv")
    

    
    
#     relvents_auths = pd.read_csv(file_name+".csv")
    
#     relvents_auths = relvents_auths.rename(columns={relvents_auths.columns[0]: 'id'})
    
#     list_ids_relevant=relvents_auths["id"].tolist()
    
    
#     def retrieve_author_tags_new(authors, author_id):
      
#         author_id= int(author_id)
#         try:
#             return ast.literal_eval(authors[authors.id == author_id].tags.values[0])
#         except:
#             return {}
    
    
#     for n in list_ids_relevant:
#         tags = [t['t'].lower() for t in retrieve_author_tags_new(authors,n)]
#         #print("* tags  : ",tags,"// id : ",n)
        
#         if tags:
#             b=1
#         else:
#             relvents_auths.drop(relvents_auths.index[relvents_auths['id'] == n], inplace=True)
    
#     relvents_auths.to_csv(file_name+'_new.csv', index=False)





    

def save_files(file_path):

    #*******************************************************  
    #                   dataframe ranking  
    #*******************************************************  
    
    df_results = pd.DataFrame(columns=["Query","Exact","Approximate"])
    i=0
    for q in queries2:
        dict = {"Query":q,"Exact":exact[i],"Approximate":approximate[i]}
        df_results = df_results.append(dict, ignore_index = True)
        i=i+1
    

    df_results.to_csv(file_path+"_ranking.csv")  
    
    
    #*******************************************************  
    #               original_method_results.txt  
    #******************************************************* 
    
    
    text = "Exact binary MRR@10:"+ str(mean_reciprocal_rank(exact))+" \nApproximate binary MRR@10:"+ str(mean_reciprocal_rank(approximate))+"\nExact binary MAP@10:"+ str(mean_average_precision(exact)) +" \nApproximate binary MAP@10:"+ str(mean_average_precision(approximate))+"\nExact binary MP@5 :"+ str(mean_precision_at_n(exact, n=5))+"\nApproximate binary MP@5 :"+ str(mean_precision_at_n(approximate, n=5))+"\nExact binary MP@10 :"+ str(mean_precision_at_n(exact, n=10))+"\nApproximate binary MP@10 :"+ str(mean_precision_at_n(approximate, n=10))
    
    with open(file_path+'_results.txt', 'w') as f:
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
    for q in queries2:
        l=[]
        l.append(exact[i])
        b=[]
        b.append(approximate[i])
        
        dict_ = {"Query":q,"Exact binary MRR@10":  ( 0 if math.isnan( mean_reciprocal_rank(l)) else mean_reciprocal_rank(l)),"Approximate binary MRR@10":( 0 if math.isnan(mean_reciprocal_rank(b)) else mean_reciprocal_rank(b)),"Exact binary MAP@10":( 0 if math.isnan(mean_average_precision(l)) else mean_average_precision(l)) ,"Approximate binary MAP@10":mean_average_precision(b),"Exact binary MP@5":mean_precision_at_n(l, n=5),"Exact binary MP@10":mean_precision_at_n(l, n=10),"Approximate binary MP@5":mean_precision_at_n(b, n=5),"Approximate binary MP@10":mean_precision_at_n(b, n=10)}
        df_results_eval = df_results_eval.append(dict_, ignore_index = True)
        i=i+1
        

    df_results_eval.to_csv(file_path+"_metrics.csv")