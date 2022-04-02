#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 18:11:09 2022

@author: serine
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


def dist2sim(d):
    return 1 - d / 2


def load_relevant_index(type="separate_sbert"):
    index = None
    if type == "separate_sbert":
        index = faiss.read_index("Mapped_indeces/separate_embeddings_faiss.index")
    elif type == "merged_sbert":
        index = faiss.read_index("Mapped_indeces/merged_embeddings_faiss.index")
    elif type == "retro_merged_sbert":
        index = faiss.read_index("Mapped_indeces/retro_merged_embeddings_faiss.index")
    elif type == "retro_separate_sbert":
        index = faiss.read_index("Mapped_indeces/retro_separate_embeddings_faiss.index")
    elif type == "tfidf_svd":
        index = faiss.read_index("Mapped_indeces/tfidf_embeddings_faiss.index")
    elif type == "pooled_bert":
        index = faiss.read_index("Mapped_indeces/mean_bert_faiss.index")
    elif type == "pooled_glove":
        index = faiss.read_index("Mapped_indeces/glove_faiss.index")
    return index

def get_authors_by_id(id_):
    try:
        return data[data.id == id_].authors.values[0]
    except:
        print(id_)
        return [{"id": -999999}]


def get_first_author_by_id(id_):
    authors = get_authors_by_id(id_)
    return authors[0]

from ast import literal_eval 

def get_author_ranking_exact(query, index, k=10, tfidf=False):
    query = query.lower()
    results = retrieve_results(query, index, k, tfidf=tfidf)
    candidate_papers = results[0]

    # We remove duplicate authors for now, while preserving order (their highest position)
    #authors = list(unique_everseen([get_first_author_by_id(str(rid))["id"] for rid in candidate_papers]))
    
    authors = list(unique_everseen([literal_eval(get_first_author_by_id(rid))["id"] for rid in candidate_papers]))
    
    #authors = [a["id"] for a in literal_eval(get_authors_by_id(rid))]
    
    relevancies = [check_if_author_relevant(int(a), query) for a in authors]

    ranking = {}

    for rank,(author, relevancy) in enumerate(zip(authors, relevancies)):
        if author not in ranking.keys():
            ranking[author] = {"relevancy": relevancy, "rank": rank}
        else:
            continue

    return ranking

from collections import defaultdict
import math

def retrieve_pub_count_by_id(author_id):
    return authors[authors.id == int(author_id)].n_pubs.values[0]

from ast import literal_eval

def create_score_author_dict(query, retrieved_paper_ids, retrieved_distances, strategy="uniform", normalized=False,
                             average_pub_count=58,
                             normalization_alpha=1, extra_normalization_term=10):
    """
    Create a dictionary where each author gets a score in relation to the query. 
    The author ranking is assembled through a document-centric voting model process: 
    first, for each top retrieved paper, its score is assigned to each of the paper 
    authors following one of the data fusion strategies. Next, all the scores per author
    are aggregated into a mapping of authors to scores. Finally, a combination function (expCombSUM) 
    is applied to all author scores. These scores are returned per author in combination with the papers 
    that contributed to that score (for explainibility sake).
    
    Parameters:
    query (string): The search query
    retrieved_paper_ids (list): The papers that were retrieved from the FAISS index as 
    nearest neighbours for the query
    retrieved_distances (list): The distances from the query for each paper that were retrieved 
    from the FAISS index as nearest neighbours for the query
    strategy (string): The data fusion strategy used for assigning author score per paper
    normalized (bool): Whether normalization should be applied to the scores, boosting less prolific
    authors and "punishing" highly prolific authors
    average_pub_count (int): Average publication count for the authors in our dataset. Used for normalization
    normalization_alpha (int or float): The inverse strength of normalization (higher alpha means less normalization)
    extra_normalization_term (int): Extra normalization damping term, further reduces normalization effect
    
    
    Returns:
    authorship_scores (dict): A mapping between authors and their calculated score in relation to the query.
    """
    def expCombSUM(list_of_scores):
        return sum([math.exp(score) for score in list_of_scores])

    def normalize_score(score, l_pro, average_l=average_pub_count, alpha=normalization_alpha):
        normalized_score = score * math.log(1 + alpha * (average_l / (l_pro + extra_normalization_term)), 2)
        return normalized_score

    scores_per_author = defaultdict(list)
    reasons_per_author = defaultdict(list)
    for pi, score in zip(retrieved_paper_ids, retrieved_distances):
        # Prune only for author that exist in our data.
        authors = [item["id"] for item in literal_eval(get_authors_by_id(pi)) if
                   check_if_author_relevant(int(item["id"]), query) != 'Not in the dataset or no tags present!']
        if authors:
            if strategy == "uniform":
                score_per_author = score / len(authors)
                for author in authors:
                    if normalized:
                        pub_count = retrieve_pub_count_by_id(int(author))
                        normalized_score = normalize_score(score_per_author, pub_count)
                        scores_per_author[author].append(normalized_score)
                    else:
                        scores_per_author[author].append(score_per_author)
                    reasons_per_author[author].append({"paper": pi, "score": score})
            elif strategy == "binary":
                score_per_author = score
                for author in authors:
                    if normalized:
                        pub_count = retrieve_pub_count_by_id(int(author))
                        normalized_score = normalize_score(score_per_author, pub_count)
                        scores_per_author[author].append(normalized_score)
                    else:
                        scores_per_author[author].append(score_per_author)
                    reasons_per_author[author].append({"paper": pi, "score": score})
            elif strategy == "descending":
                decay_factor = 1
                for author in authors:
                    if normalized:
                        score_d = score * decay_factor
                        pub_count = retrieve_pub_count_by_id(int(author))
                        normalized_score = normalize_score(score_d, pub_count)
                        scores_per_author[author].append(normalized_score)
                        decay_factor -= 0.2
                    else:
                        scores_per_author[author].append(score * decay_factor)
                        decay_factor -= 0.2
                    reasons_per_author[author].append({"paper": pi, "score": score})
            elif strategy == "parabolic":
                #  TODO: here we did not yet write the normalization code because we do not run it for this config.
                decay_factor = 0.8
                scores_per_author[authors[0]].append(score)
                scores_per_author[authors[-1]].append(score)
                reasons_per_author[authors[0]].append({"paper": pi, "score": score})
                reasons_per_author[authors[-1]].append({"paper": pi, "score": score})
                for author in authors[1:-1]:
                    scores_per_author[author].append(score * decay_factor)
                    decay_factor -= 0.2
                    reasons_per_author[author] = {"paper": pi, "score": score}
                    reasons_per_author[author].append({"paper": pi, "score": score})
        else:
            continue

    authorship_scores = {k: {"score": expCombSUM(v),
                             "reasons": reasons_per_author[k]} for k, v in scores_per_author.items()}

    return authorship_scores

def produce_authors_ranking(authorship_scores):
    sortd = [(k, v) for k, v in sorted(authorship_scores.items(), key=lambda item: item[1]['score'], reverse=True)]
    return sortd

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

def retrieve_authorname_by_authorid(author_id):
    return authors[authors.id == int(author_id)].name.values[0]

def retrieve_pub_count_by_id(author_id):
    return authors[authors.id == int(author_id)].n_pubs.values[0]

def retrieve_cit_count_by_id(author_id):
    return authors[authors.id == int(author_id)].n_citation.values[0]


def get_information_by_author_id(aid, query, tfidf=False):
    pprint(f"Name: {retrieve_authorname_by_authorid(aid)}")
    print("===")
    pprint(f"Number of publications: {retrieve_pub_count_by_id(aid)}")
    print("===")
    pprint(f"Number of citations: {retrieve_cit_count_by_id(aid)}")
    print("===")
    pprint(f"Exactly relevant: {check_if_author_relevant(int(aid), query)}")
    print("===")
    pprint(f"Approximately relevant: {check_if_author_relevant_approximate(int(aid), query, tfidf=tfidf)}")
    

def get_title_by_id(id_):
    return data[data.id == id_].title.values[0]

def retrieve_authors(query, index, k=10, strategy="binary", normalized=False, norm_alpha=100, extra_term=10):
    """
    Produces an author ranking given a query and adds various metadata about the authors.
    This is the main retrieval method of the system. Each author's affiliation is looked up
    from their MAG entry at ma-graph.org, and additional author information is looked up
    through exact name look-up on WikiData. The final, enriched ranking of authors is 
    returned through the API endpoint.
    
    Parameters:
    query (string): The search query
    index (obj): The loaded FAISS index populated by paper embeddings
    k (int): The amount of authors to retrieve
    strategy (string): The data fusion strategy used for assigning author score per paper
    normalized (bool): Whether normalization should be applied to the scores, boosting less prolific
    authors and "punishing" highly prolific authors
    norm_alpha (int or float): The inverse strength of normalization (higher alpha means less normalization)
    extra_term (int): Extra normalization damping term, further reduces normalization effect
    
    Returns:
    enriched_top_n (list): A list of k most relevant authors to the query, where each author is contained within a dictionary
    which was enriched with various additional metadata.
    """
    
    s = time.time()
    if k*10 < 100:
        amount_papers_to_retrieve = 100
    else:
        amount_papers_to_retrieve = k*10

    print("Searching in:", amount_papers_to_retrieve, "papers")

    i, d = get_most_similar_ids(query.lower(), index, amount_papers_to_retrieve)

    # print(time.time() - s, "seconds spend in the FAISS search.")
    # s1 = time.time()
    # print("i:", i)
    # print("d", d)
    author_score_dict = create_score_author_dict(query, i, d, strategy,
                                                 normalized=normalized, normalization_alpha=norm_alpha,
                                                 extra_normalization_term=extra_term)

    # print(time.time() - s1, "seconds spend doing the voting model / data fusion and creating the author to score dict.")
    # s2 = time.time()
    # print("author_score_dict:", author_score_dict)
    top_n = produce_authors_ranking(author_score_dict)[:k]
    # print(time.time() - s2, "seconds spend sorting the author scores.")
    # s3 = time.time()

    def enrich_author_info(aid, extra_info, index):

        def parse_reasons(reasons):

            reasons = list(unique_everseen(reasons))

            return [{'paper': {"id": str(item["paper"]),
                               "title": "tirle",
                               "year": "year",
                               # "abstract": get_abstract_by_id(str(item['paper'])),
                               "tags": "fos",
                               "magPage": f"https://academic.microsoft.com/paper/{item['paper']}",
                               "semanticScholarPage": f"https://api.semanticscholar.org/MAG:{item['paper']}"}, 'score': float(item['score'])} for item in reasons]


        def parse_wikidata(wd):
            cleaned = {}
            if wd == {}:
                return cleaned
            else:
                if wd["wikidata_id"]:
                    cleaned["wikidataId"] = wd["wikidata_id"]
                if wd['google_scholar_link']:
                    cleaned["googleScholarLink"] = wd["google_scholar_link"]
                if wd['occupations'] != ['']:
                    cleaned["occupations"] = list(set(wd["occupations"]))
                if wd['personal_websites']:
                    cleaned["personalWebsites"] = wd["personal_websites"]
                if wd['employers'] != ['']:
                    cleaned["allEmployers"] = list(set(wd["employers"]))
                if wd['educated_at'] != ['']:
                    cleaned['educatedAt'] = list(set(wd["educated_at"]))
                if wd['age']:
                    cleaned["age"] = wd["age"]
                if wd['notable_works'] != ['']:
                    cleaned["notableWorks"] = list(set(wd["notable_works"]))
                if wd['awards']:
                    cleaned["receivedAwards"] = list(set(wd['awards']))
                if wd['academic_degree']:
                    cleaned['academicDegree'] = wd['academic_degree']
                if wd['alive']:
                    cleaned['alive'] = wd['alive']
            return cleaned

        name = retrieve_authorname_by_authorid(aid)
        n_pubs = retrieve_pub_count_by_id(aid)
        n_cits = retrieve_cit_count_by_id(aid)
        #affiliation_info = get_author_affiliation_infos(aid)
        #wikidata = parse_wikidata(get_author_wikidata(aid))
        #reasons = parse_reasons(extra_info['reasons'])
        #try:
        #    country = affiliation_info["country"]
        #except KeyError:
        country = "Unknown"
        mag_profile_link = f"https://academic.microsoft.com/author/{aid}"
        return {"name": name, "nPublications": int(n_pubs),
                "nCitations": int(n_cits), "country": country, "id": aid, "magProfile": mag_profile_link,
                "rank": index}

    enriched_top_n = [enrich_author_info(aid, extra_info, index) for index, (aid, extra_info) in enumerate(top_n)]

    # print(time.time() - s3, "seconds spend enriching the retrieved authors with extra info.")

    return enriched_top_n

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

def load_data_and_authors(data_path="papers.csv", 
                          authors_path="authors.csv"):
    data = pd.read_csv(data_path)
    authors = pd.read_csv(authors_path)
    return data, authors

def get_author_ranking_exact_v2(query1,query2, index, k=10, tfidf=False, strategy="uniform",
                                normalized=False, norm_alpha=100, extra_term=10):
    """
    Produces an author ranking given a query and adds relevancy flag to the author
    based on the exact topic evaluation criteria. Used for evaluating the system.
    
    Parameters:
    query (string): The search query
    index (obj): The loaded FAISS index populated by paper embeddings
    k (int): The amount of authors to retrieve
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
    if tfidf:
        print("tfidf")
    else:
        i, d = get_most_similar_ids(query2.lower(), index, 100)
        #i, d = get_most_similar_ids(query2, index, 100)

    

    res = df_read[query1].copy()
    # sort values
    res.sort_values(inplace=True)
    # dict like cluster analysis' one
    dic_q = res.to_dict()
    result_top_10 = produce_authors_ranking_new(dic_q)[:10]
    
    top_n = result_top_10

    relevancies = [check_if_author_relevant(aid[0], query1) for aid in top_n]

    ranking = {}

    for rank, (author, relevancy) in enumerate(zip([a[0] for a in top_n], relevancies)):
        if author not in ranking.keys():
            ranking[author] = {"relevancy": relevancy, "rank": rank}
        else:
            continue

    return ranking


def get_author_ranking_approximate_v2(query1,query2, index, k=10, similarity_threshold=0.7, tfidf=False, strategy="uniform",
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
    if tfidf:
        print("tfidf")
    else:
        i, d = get_most_similar_ids(query2.lower(), index, 100)
        #i, d = get_most_similar_ids(query2, index, 100
    

    
    
    res = df_read[query1].copy()
    # sort values
    res.sort_values(inplace=True)
    # dict like cluster analysis' one
    dic_q = res.to_dict()
    result_top_10 = produce_authors_ranking_new(dic_q)[:10]
    
    top_n = result_top_10


    relevancies = [check_if_author_relevant_approximate(aid[0], query1, similarity_threshold, tfidf=tfidf) for aid in top_n]

    ranking = {}

    for rank, (author, relevancy) in enumerate(zip([a[0] for a in top_n], relevancies)):
        if author not in ranking.keys():
            ranking[author] = {"relevancy": relevancy, "rank": rank}
        else:
            continue

    return ranking

def retrieve_author_by_id(author_id):
    return authors[authors.id == int(author_id)]

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
from ast import literal_eval    

def prune_results_for_authors_wo_tags(results, query, how_many=10):
    ids = results[0]
    distances = results[1]
    
    relevant_ids = []
    relevant_distances = []
    # For now, I check if the first author is not in the set, I throw the paper away, because I now
    # only look at first author for evaluation. But later if I have another strategy for retrieving author per paper
    # we can change this logic back to "all authors not in the set".
    for rid, rd in zip(ids, distances):
         
        authors = [a["id"] for a in literal_eval(get_authors_by_id(rid))]
        #print("authors : ", get_authors_by_id(rid))
        relevancy = [check_if_author_relevant(int(a), query) for a in authors]
        #print("relevancy  : ", relevancy )
        # if relevancy != ['Not in the dataset or no tags present!']*len(relevancy):
        #     relevant_ids.append(rid)
        #     relevant_distances.append(rd)
        if relevancy[0] != 'Not in the dataset or no tags present!':
            relevant_ids.append(rid)
            relevant_distances.append(rd)
    
    
    return relevant_ids[:how_many], relevant_distances[:how_many]

def get_most_similar_ids(query, index, k=10, tfidf_classifier=None):
    # First, embed the query, normalize the vector and convert to float32

    if tfidf_classifier:
        query_emb = tfidf_classifier.transform([query])[0]
        normalized_query = np.float32([query_emb])[0]
    else:
        query_emb = embedder.encode([query])[0]
        #print("query_emb : ",query_emb)
        normalized_query = np.float32(normalize([query_emb])[0])
        #print("normalized_query : ",normalized_query)

    assert type(normalized_query[0]).__name__ == 'float32'

    #Next, run the index search
    s = time.time()
    dists, idxs = index.search(np.array([normalized_query]), k)
    #print("dists : ",dists)
    # print("Search execution time:")
    # print((time.time() - s), "s.")
    # print("IDS, sorted by similarity:")
    # print(idxs[0])
    # print('Similarity scores:')
    # print(dist2sim(dists[0]))
    return idxs[0], dist2sim(dists[0])

def retrieve_results(query, index, k=10, verbose=False, tfidf=False):
    initial_retrieval = k*5
    s = time.time()
    if tfidf:
        print("tfidf")
    else:
        most_similar_raw = get_most_similar_ids(query, index, initial_retrieval)
    s1 = time.time()
    pruned = prune_results_for_authors_wo_tags(most_similar_raw, query, k)
    s2 = time.time()
    if verbose:
        print(f"Full search execution time: {time.time() - s} seconds")
        print(f"from which {s1-s} s. in the search and {s2 - s1} s. in the pruning.")
        print("===")
        print("Pruned IDS, sorted by similarity:")
        print(pruned[0])
        print('Similarity scores:')
        print(pruned[1])
    return pruned

def produce_authors_ranking_new(result):
    sortd = [(k, v) for k, v in sorted(result.items(), key=lambda item: item[1], reverse=True)]
    return sortd

#/***********************************************************************************/

embedder = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
# res = faiss.StandardGpuResources()  # use a single GPU


data_and_authors = load_data_and_authors()
data = data_and_authors[0]
authors = data_and_authors[1]

index = load_relevant_index("separate_sbert")


df_read = pd.read_csv("relvents_auths_all_queries.csv",index_col=0)

queries = df_read.columns.values

#df_results = pd.DataFrame(columns=["Query","Exact binary MRR@10","Approximate binary MRR@10","Exact binary MAP@10","Approximate binary MAP@10","Exact binary MP@10:","Approximate binary MP@10","Exact binary MP@5","Approximate binary MP@5","Exact uniform MRR@10","Approximate uniform MRR@10","Exact uniform MAP@10","Approximate uniform MAP@10","Exact uniform MP@10","Approximate uniform MP@10","Exact uniform MP@5","Approximate uniform MP@5"])


#i=1
#for q in queries:
    #print("query : ",q," / indice : ":i)
    #i=i+1
    
    #res = df_read[q].copy()
    # sort values
    #res.sort_values(inplace=True)
    # dict like cluster analysis' one
    #dic_q = res.to_dict()
    #result_top_10 = produce_authors_ranking_new(dic_q)[:10]
    
    #queries1 = [q]
    #queries2 = [q]

    

exact = [get_author_ranking_exact_v2(query1,query1, index, tfidf=False, strategy="binary", normalized=True, norm_alpha=1) for query1 in queries]

    
approximate = [get_author_ranking_approximate_v2(query1,query1, index, tfidf=False, strategy="binary", normalized=True, norm_alpha=1) for query1 in queries]


exact_uniform = [get_author_ranking_exact_v2(query1,query1, index, tfidf=False, strategy="uniform", normalized=True, norm_alpha=1) for query1 in queries]


approximate_uniform = [get_author_ranking_approximate_v2(query1,query1,index, tfidf=False, strategy="uniform", normalized=True, norm_alpha=1) for query1 in queries]

#dict = {"Query":q,"Exact binary MRR@10": mean_reciprocal_rank(exact),"Approximate binary MRR@10":mean_reciprocal_rank(approximate),"Exact binary MAP@10":mean_average_precision(exact),"Approximate binary MAP@10":mean_average_precision(approximate),"Exact binary MP@10":mean_precision_at_n(exact, 10),"Approximate binary MP@10":mean_precision_at_n(approximate, 10),"Exact binary MP@5":mean_precision_at_n(exact, 5),"Approximate binary MP@5":mean_precision_at_n(approximate, 5),"Exact uniform MRR@10":mean_reciprocal_rank(exact_uniform),"Approximate uniform MRR@10":mean_reciprocal_rank(approximate_uniform),"Exact uniform MAP@10":mean_average_precision(exact_uniform),"Approximate uniform MAP@10":mean_average_precision(approximate_uniform),"Exact uniform MP@10":mean_precision_at_n(exact_uniform, 10),"Approximate uniform MP@10":mean_precision_at_n(approximate_uniform, 10),"Exact uniform MP@5":mean_precision_at_n(exact_uniform, 5),"Approximate uniform MP@5":mean_precision_at_n(approximate_uniform, 5)}
#df_results = df_results.append(dict, ignore_index = True)

print("Exact binary MRR@10:", mean_reciprocal_rank(exact)," / Approximate binary MRR@10:", mean_reciprocal_rank(approximate)," / Exact binary MAP@10:", mean_average_precision(exact)," / Approximate binary MAP@10:", mean_average_precision(approximate)," / Exact binary MP@10:", mean_precision_at_n(exact, 10)," / Approximate binary MP@10:", mean_precision_at_n(approximate, 10)," / Exact binary MP@5:", mean_precision_at_n(exact, 5)," / Approximate binary MP@5:", mean_precision_at_n(approximate, 5)," // Exact uniform MRR@10:", mean_reciprocal_rank(exact_uniform)," / Approximate uniform MRR@10:", mean_reciprocal_rank(approximate_uniform)," / Exact uniform MAP@10:", mean_average_precision(exact_uniform)," / Approximate uniform MAP@10:", mean_average_precision(approximate_uniform)," / Exact uniform MP@10:", mean_precision_at_n(exact_uniform, 10)," / Approximate uniform MP@10:", mean_precision_at_n(approximate_uniform, 10)," / Exact uniform MP@5:", mean_precision_at_n(exact_uniform, 5)," / Approximate uniform MP@5:", mean_precision_at_n(approximate_uniform, 5))

# import pandas as pd
# df_results.to_csv("our_method_evaluation_results.csv")


# eval_1=df_results['Approximate uniform MP@5'].tolist()
# eval_1 = np.array(eval_1) 
# eval_1 = [0 if pd.isna(x) else x for x in eval_1]
# pan = np.mean(eval_1)
