#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 23:10:08 2022

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
from collections import defaultdict
import math



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


def load_data_and_authors(data_path="papers.csv", 
                          authors_path="authors.csv"):
    data = pd.read_csv(data_path)
    authors = pd.read_csv(authors_path)
    return data, authors

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
        q_list = query.split('-')
        l = []
        for q in q_list:
            e = embedder.encode([q.lower()])[0]
            l.append(e)
        
        #print(l)
        l = np.array(l)
        #print(l.shape)
        mean_query = (l[0]+l[1] ) / 2
        #mean_query = np.mean(l)
        #print(mean_query.shape)
        #print("mean_query : ",mean_query)
        #query_emb = embedder.encode([query])[0]
        #normalized_query = np.float32(normalize([query_emb])[0])
        normalized_query = np.float32(normalize(np.array([mean_query]))[0])
        

    assert type(normalized_query[0]).__name__ == 'float32'

    #Next, run the index search
    s = time.time()
    dists, idxs = index.search(np.array([normalized_query]), k)
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
        #most_similar_raw = get_most_similar_ids(query, index, initial_retrieval, tfidf_clf)
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
    
    
def get_author_ranking_exact_v2(query1,query2, index,result_top_10, k=10, tfidf=False, strategy="uniform",
                                normalized=False, norm_alpha=100, extra_term=10):
   
    if tfidf:
        print("tfidf")
    else:
        #i, d = get_most_similar_ids(query2.lower(), index, 100)
        i, d = get_most_similar_ids(query2, index, 100)

    
    top_n = result_top_10

    relevancies = [check_if_author_relevant(int(aid), query1) for aid, _ in top_n]

    ranking = {}

    for rank, (author, relevancy) in enumerate(zip([a[0] for a in top_n], relevancies)):
        if author not in ranking.keys():
            ranking[author] = {"relevancy": relevancy, "rank": rank}
        else:
            continue

    return ranking


def get_author_ranking_approximate_v2(query1,query2, index,result_top_10, k=10, similarity_threshold=0.7, tfidf=False, strategy="uniform",
                                      normalized=False, norm_alpha=100, extra_term=10):
   
    if tfidf:
        print("tfidf")
    else:
        #i, d = get_most_similar_ids(query2.lower(), index, 100)
        i, d = get_most_similar_ids(query2, index, 100)

    
    top_n = result_top_10

    relevancies = [check_if_author_relevant_approximate(int(aid), query1, similarity_threshold, tfidf=tfidf) for aid, _
                   in top_n]

    ranking = {}

    for rank, (author, relevancy) in enumerate(zip([a[0] for a in top_n], relevancies)):
        if author not in ranking.keys():
            ranking[author] = {"relevancy": relevancy, "rank": rank}
        else:
            continue

    return ranking

def get_title_by_id(id_):
    return data[data.id == id_].title.values[0]

def retrieve_authors(query, index, k=10, strategy="binary", normalized=False, norm_alpha=100, extra_term=10):
    
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
                               "title": get_title_by_id(str(item['paper'])),
                               "year": int('2000'),
                               # "abstract": get_abstract_by_id(str(item['paper'])),
                               "tags": ["tag"],
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



embedder = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
# res = faiss.StandardGpuResources()  # use a single GPU

data_and_authors = load_data_and_authors()
data = data_and_authors[0]
authors = data_and_authors[1]

index = load_relevant_index("separate_sbert")

import pickle
# to load
with open("cluster_analysis_result", "rb") as f:
          u = pickle.Unpickler(f)
          result = u.load()
          
#result_sorted = dict(sorted(result.items(), key=lambda item: item[1] , reverse=True))ù


def produce_authors_ranking_new(result):
    sortd = [(k, v) for k, v in sorted(result.items(), key=lambda item: item[1], reverse=True)]
    return sortd

result_top_10 = produce_authors_ranking_new(result)[:10]



queries1 = ['cluster analysis', 'Image segmentation', 'Parallel algorithm', 'Monte Carlo method',
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


queries2 = queries2 = ['cluster analysis@task of grouping a set of objects in such a way that objects in the same group called a cluster are more similar in some sense or another to each other than to those in other groups clusters',
 'Image segmentation@Division of an image into sets of pixels for further processing',
 'Parallel algorithm algorithm@in computer science, a parallel algorithm, as opposed to a traditional serial algorithm, is an algorithm which can do multiple operations in a given time. It has been a tradition of computer science to describe serial algorithms in abstract machine models, often the one known as random-access machine. Similarly, many computer science researchers have used a so-called parallel random-access machine (PRAM) as a parallel abstract machine (shared-memory).Many parallel algorithms are executed concurrently – though in general concurrent algorithms are a distinct concept – and thus these concepts are often conflated, with which aspect of an algorithm is parallel and which is concurrent not being clearly distinguished. Further, non-parallel, non-concurrent algorithms are often referred to as "sequential algorithms", by contrast with concurrent algorithms.',
 'Monte Carlo method@broad class of computational algorithms using random sampling to obtain numerical results',
 'Convex optimization@subfield of mathematical optimization',
 'Dimensionality reduction@process of reducing the number of random variables under consideration',
 'Facial recognition system@computer application capable of identifying or verifying an individual person from a digital image',
 'k-nearest neighbors algorithm@In statistics, the k-nearest neighbors algorithm (k-NN) is a non parametric supervised learning method first developed by Evelyn Fix and Joseph Hodges in 1951, and later expanded by Thomas Cover. It is used for classification and regression. In both cases, the input consists of the k closest training examples in a data set. The output depends on whether k-NN is used for classification or regression:In k-NN classification, the output is a class membership. An object is classified by a plurality vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.In k-NN regression, the output is the property value for the object. This value is the average of the values of k nearest neighbors.k-NN is a type of classification where the function is only approximated locally and all computation is deferred until function evaluation. Since this algorithm relies on distance for classification, if the features represent different physical units or come in vastly different scales then normalizing the training data can improve its accuracy dramatically.Both for classification and regression, a useful technique can be to assign weights to the contributions of the neighbors, so that the nearer neighbors contribute more to the average than the more distant ones. For example, a common weighting scheme consists in giving each neighbor a weight of 1/d, where d is the distance to the neighbor.The neighbors are taken from a set of objects for which the class (for k-NN classification) or the object property value (for k-NN regression) is known. This can be thought of as the training set for the algorithm, though no explicit training step is required. A peculiarity of the k NN algorithm is that it is sensitive to the local structure of the data.',
 'Hierarchical clustering@method of cluster analysis which seeks to build a hierarchy of clusters',
 'Automatic summarization@computer-based method for shortening a text',
 'Dynamic programming@Problem optimization method.',
 'Genetic algorithm@competitive algorithm for searching a problem space',
 'Human-computer interaction@is research in the design and the use of computer technology, which focuses on the interfaces between people (users) and computers. HCI researchers observe the ways humans interact with computers and design technologies that allow humans to interact with computers in novel ways. As a field of research, human–computer interaction is situated at the intersection of computer science, behavioral sciences, design, media studies, and several other fields of study. The term was popularized by Stuart K. Card, Allen Newell, and Thomas P. Moran in their 1983 book, The Psychology of Human–Computer Interaction. The first known use was in 1975 by Carlisle. The term is intended to convey that, unlike other tools with specific and limited uses, computers have many uses which often involve an open-ended dialogue between the user and the computer. The notion of dialogue likens human computer interaction to human to human interaction: an analogy that is crucial to theoretical considerations in the field.',
 'Categorial grammar@family of formalisms in natural language syntax motivated by the principle of compositionality and organized according to the view that syntactic constituents should generally combine as functions or according to a function-argument relationship',
 'Semantic Web@extension of the Web to facilitate data exchange',
 'fuzzy logic@system for reasoning about vagueness',
 'image restoration@is the operation of taking a corrupt/noisy image and estimating the clean, original image. Corruption may come in many forms such as motion blur, noise and camera mis-focus. Image restoration is performed by reversing the process that blurred the image and such is performed by imaging a point source and use the point source image, which is called the Point Spread Function (PSF) to restore the image information lost to the blurring process. Image restoration is different from image enhancement in that the latter is designed to emphasize features of the image that make the image more pleasing to the observer, but not necessarily to produce realistic data from a scientific point of view. Image enhancement techniques (like contrast stretching or de-blurring by a nearest neighbor procedure) provided by imaging packages use no a priori model of the process that created the image. With image enhancement noise can effectively be removed by sacrificing some resolution, but this is not acceptable in many applications. In a fluorescence microscope, resolution in the z-direction is bad as it is. More advanced image processing techniques must be applied to recover the object. The objective of image restoration techniques is to reduce noise and recover resolution loss Image processing techniques are performed either in the image domain or the frequency domain. The most straightforward and a conventional technique for image restoration is deconvolution, which is performed in the frequency domain and after computing the Fourier transform of both the image and the PSF and undo the resolution loss caused by the blurring factors. This deconvolution technique, because of its direct inversion of the PSF which typically has poor matrix condition number, amplifies noise and creates an imperfect deblurred image. Also, conventionally the blurring process is assumed to be shift-invariant. Hence more sophisticated techniques, such as regularized deblurring, have been developed to offer robust recovery under different types of noises and blurring functions. It is of 3 types:1. Geometric correction 2. radiometric correction3. noise removal == References ==',
 'generative model@model for randomly generating observable data in probability and statistics',
 'search algorithm@any algorithm which solves the search problem namely to retrieve information stored within some data structure or calculated in the search space of a problem domain either with discrete or continuous values',
 'sample size determination@is the act of choosing the number of observations or replicates to include in a statistical sample. The sample size is an important feature of any empirical study in which the goal is to make inferences about a population from a sample. In practice, the sample size used in a study is usually determined based on the cost, time, or convenience of collecting the data, and the need for it to offer sufficient statistical power. In complicated studies there may be several different sample sizes: for example, in a stratified survey there would be different sizes for each stratum. In a census, data is sought for an entire population, hence the intended sample size is equal to the population. In experimental design, where a study may be divided into different treatment groups, there may be different sample sizes for each group. Sample sizes may be chosen in several ways:using experience –small samples, though sometimes unavoidable, can result in wide confidence intervals and risk of errors in statistical hypothesis testing. using a target variance for an estimate to be derived from the sample eventually obtained, i.e. if a high precision is required (narrow confidence interval) this translates to a low target variance of the estimator. using a target for the power of a statistical test to be applied once the sample is collected. using a confidence level, i.e. the larger the required confidence level, the larger the sample size (given a constant precision requirement).',
 'anomaly detection@is generally understood to be the identification of rare items, events or observations which deviate significantly from the majority of the data and do not conform to a well defined notion of normal behaviour. Such examples may arouse suspicions of being generated by a different mechanism, or appear inconsistent with the remainder of that set of data.Anomaly detection finds application in many domains including cyber security, medicine, machine vision, statistics, neuroscience, law enforcement and financial fraud---to name only a few. Anomalies were initially searched for clear rejection or omission from the data to aid statistical analysis, for example to compute the mean or standard deviation. They were also removed to better predictions from models such as linear regression, and more recently their removal aids the performance of machine learning algorithms. However, in many applications anomalies themselves are of interest and are the observations most desirous in the entire data set---which need to be identified and separated from noise or irrelevant outliers.Three broad categories of anomaly detection techniques exist. Supervised anomaly detection techniques require a data set that has been labeled as "normal" and "abnormal" and involves training a classifier. However, this approach is rarely used in anomaly detection due to the general unavailability of labelled data and the inherent unbalanced nature of the classes. Semi-supervised anomaly detection techniques assume that some portion of the data is labelled. This may be any combination of the normal or anomalous data, but more often than not the techniques construct a model representing normal behavior from a given normal training data set, and then test the likelihood of a test instance to be generated by the model.',
 'sentiment analysis@use of natural language processing text analysis and computational linguistics to identify and extract subjective information in source materials',
 'semantic similarity@is a metric defined over a set of documents or terms, where the idea of distance between items is based on the likeness of their meaning or semantic content as opposed to lexicographical similarity. These are mathematical tools used to estimate the strength of the semantic relationship between units of language, concepts or instances, through a numerical description obtained according to the comparison of information supporting their meaning or describing their nature. The term semantic similarity is often confused with semantic relatedness. Semantic relatedness includes any relation between two terms, while semantic similarity only includes "is a" relations. For example, "car" is similar to "bus", but is also related to "road" and "driving". Computationally, semantic similarity can be estimated by defining a topological similarity, by using ontologies to define the distance between terms/concepts. For example, a naive metric for the comparison of concepts ordered in a partially ordered set and represented as nodes of a directed acyclic graph (e.g., a taxonomy), would be the shortest-path linking the two concept nodes. Based on text analyses, semantic relatedness between units of language (e.g., words, sentences) can also be estimated using statistical means such as a vector space model to correlate words and textual contexts from a suitable text corpus. The evaluation of the proposed semantic similarity / relatedness measures are evaluated through two main ways. The former is based on the use of datasets designed by experts and composed of word pairs with semantic similarity / relatedness degree estimation. The second way is based on the integration of the measures inside specific applications such the information retrieval, recommender systems, natural language processing, etc.',
 'world wide web@system of interlinked hypertext documents accessed via the Internet',
 'gibbs sampling@In statistics, Gibbs sampling or a Gibbs sampler is a Markov chain Monte Carlo (MCMC) algorithm for obtaining a sequence of observations which are approximated from a specified multivariate probability distribution, when direct sampling is difficult.This sequence can be used to approximate the joint distribution (e.g., to generate a histogram of the distribution); to approximate the marginal distribution of one of the variables, or some subset of the variables (for example, the unknown parameters or latent variables); or to compute an integral (such as the expected value of one of the variables).Typically, some of the variables correspond to observations whose values are known, and hence do not need to be sampled. Gibbs sampling is commonly used as a means of statistical inference, especially Bayesian inference.It is a randomized algorithm (i.e. an algorithm that makes use of random numbers), and is an alternative to deterministic algorithms for statistical inference such as the expectation-maximization algorithm (EM). As with other MCMC algorithms, Gibbs sampling generates a Markov chain of samples, each of which is correlated with nearby samples. As a result, care must be taken if independent samples are desired. Generally, samples from the beginning of the chain (the burn-in period) may not accurately represent the desired distribution and are usually discarded.',
 'user interface@means by which a user interacts with and controls a machine',
 'belief propagation@also known as sum–product message passing, is a message-passing algorithm for performing inference on graphical models, such as Bayesian networks and Markov random fields. It calculates the marginal distribution for each unobserved node (or variable), conditional on any observed nodes (or variables). Belief propagation is commonly used in artificial intelligence and information theory, and has demonstrated empirical success in numerous applications, including low-density parity-check codes, turbo codes, free energy approximation, and satisfiability.The algorithm was first proposed by Judea Pearl in 1982, who formulated it as an exact inference algorithm on trees, later extended to polytrees. While the algorithm is not exact on general graphs, it has been shown to be a useful approximate algorithm.',
 'interpolation method for constructing new data from known data',
 'wavelet transform@mathematical technique used in data compression and analysis',
 'transfer of learning@dependency of human conduct learning or performance on prior experience',
 'topic model@In machine learning and natural language processing, a topic model is a type of statistical model for discovering the abstract "topics" that occur in a collection of documents. Topic modeling is a frequently used text-mining tool for discovery of hidden semantic structures in a text body. Intuitively, given that a document is about a particular topic, one would expect particular words to appear in the document more or less frequently: "dog" and "bone" will appear more often in documents about dogs, "cat" and "meow" will appear in documents about cats, and "the" and "is" will appear approximately equally in both. A document typically concerns multiple topics in different proportions; thus, in a document that is 10% about cats and 90% about dogs, there would probably be about 9 times more dog words than cat words. The "topics" produced by topic modeling techniques are clusters of similar words. A topic model captures this intuition in a mathematical framework, which allows examining a set of documents and discovering, based on the statistics of the words in each, what the topics might be and what each document\'s balance of topics is. Topic models are also referred to as probabilistic topic models, which refers to statistical algorithms for discovering the latent semantic structures of an extensive text body. In the age of information, the amount of the written material we encounter each day is simply beyond our processing capacity. Topic models can help to organize and offer insights for us to understand large collections of unstructured text bodies. Originally developed as a text-mining tool, topic models have been used to detect instructive structures in data such as genetic information, images, and networks. They also have applications in other fields such as bioinformatics and computer vision.',
 'clustering high-dimensional data@is the cluster analysis of data with anywhere from a few dozen to many thousands of dimensions. Such high-dimensional spaces of data are often encountered in areas such as medicine, where DNA microarray technology can produce many measurements at once, and the clustering of text documents, where, if a word-frequency vector is used, the number of dimensions equals the size of the vocabulary.',
 'game theory@branch of mathematics focused on strategic decision making',
 "biometrics@metrics related to an individual's characteristics",
 'constraint satisfaction@process of finding a solution to a set of constraints that impose conditions that the variables must satisfy',
 'combinatorial optimization@subset of mathematical optimization',
 'speech processing@study of speech signals and the processing methods of these signals',
 'multi-agent system@built of multiple interacting agents',
 'mean field theory@approximation method where the behavior of a single particle can be treated assuming all other influences are averaged',
 'social network@theoretical concept in sociology',
 'lattice model@a physical model that is defined on a lattice as opposed to the continuum of space or spacetime',
 'automatic image annotation@process which automatically assigns metadata in the form of captioning or keywords to a digital image',
 'computational geometry@is a branch of computer science devoted to the study of algorithms which can be stated in terms of geometry. Some purely geometrical problems arise out of the study of computational geometric algorithms, and such problems are also considered to be part of computational geometry. While modern computational geometry is a recent development, it is one of the oldest fields of computing with a history stretching back to antiquity. Computational complexity is central to computational geometry, with great practical significance if algorithms are used on very large datasets containing tens or hundreds of millions of points. For such sets, the difference between O(n2) and O(n log n) may be the difference between days and seconds of computation. The main impetus for the development of computational geometry as a discipline was progress in computer graphics and computer-aided design and manufacturing (CAD/CAM), but many problems in computational geometry are classical in nature, and may come from mathematical visualization. Other important applications of computational geometry include robotics (motion planning and visibility problems), geographic information systems (GIS) (geometrical location and search, route planning), integrated circuit design (IC geometry design and verification), computer-aided engineering (CAE) (mesh generation), and computer vision (3D reconstruction). The main branches of computational geometry are:Combinatorial computational geometry, also called algorithmic geometry, which deals with geometric objects as discrete entities. A groundlaying book in the subject by Preparata and Shamos dates the first use of the term "computational geometry" in this sense by 1975. Numerical computational geometry, also called machine geometry, computer-aided geometric design (CAGD), or geometric modeling, which deals primarily with representing real-world objects in forms suitable for computer computations in CAD/CAM systems. This branch may be seen as a further development of descriptive geometry and is often considered a branch of computer graphics or CAD. The term "computational geometry" in this meaning has been in use since 1971.Although most algorithms of computational geometry have been developed (and are being developed) for electronic computers, some algorithms were developed for unconventional computers (e.g. optical computers )',
 'Evolutionary algorithm@is a subset of evolutionary computation, a generic population-based metaheuristic optimization algorithm. An EA uses mechanisms inspired by biological evolution, such as reproduction, mutation, recombination, and selection. Candidate solutions to the optimization problem play the role of individuals in a population, and the fitness function determines the quality of the solutions (see also loss function). Evolution of the population then takes place after the repeated application of the above operators. Evolutionary algorithms often perform well approximating solutions to all types of problems because they ideally do not make any assumption about the underlying fitness landscape. Techniques from evolutionary algorithms applied to the modeling of biological evolution are generally limited to explorations of microevolutionary processes and planning models based upon cellular processes. In most real applications of EAs, computational complexity is a prohibiting factor. In fact, this computational complexity is due to fitness function evaluation. Fitness approximation is one of the solutions to overcome this difficulty. However, seemingly simple EA can solve often complex problems; therefore, there may be no direct link between algorithm complexity and problem complexity.',
 'web search query@is a query that a user enters into a web search engine to satisfy their information needs. Web search queries are distinctive in that they are often plain text and boolean search directives are rarely used. They vary greatly from standard query languages, which are governed by strict syntax rules as command languages with keyword or positional parameters.',
 "eye tracking@is the process of measuring either the point of gaze (where one is looking) or the motion of an eye relative to the head.An eye tracker is a device for measuring eye positions and eye movement.Eye trackers are used in research on the visual system, in psychology, in psycholinguistics, marketing, as an input device for human-computer interaction, and in product design. Eye trackers are also being increasingly used for rehabilitative and assistive applications (related,for instance, to control of wheel chairs, robotic arms and prostheses). There are a number of methods for measuring eye movement.The most popular variant uses video images from which the eye position is extracted.Other methods use search coils or are based on the electrooculogram.",
 'query optimization@feature to efficiently execute queries efficiently in DBMS softwares',
 'logic programming@programming paradigm based on formal logic',
 'Hyperspectral imaging@method to create a complete picture of the environment or various objects each pixel containing a full visible visible near infrared near infrared or infrared spectrum.',
 'Bayesian statistics@is a theory in the field of statistics based on the Bayesian interpretation of probability where probability expresses a degree of belief in an event. The degree of belief may be based on prior knowledge about the event, such as the results of previous experiments, or on personal beliefs about the event. This differs from a number of other interpretations of probability, such as the frequentist interpretation that views probability as the limit of the relative frequency of an event after many trials.Bayesian statistical methods use Bayes theorem to compute and update probabilities after obtaining new data. Bayes theorem describes the conditional probability of an event based on data as well as prior information or beliefs about the event or conditions related to the event. For example, in Bayesian inference, Bayes theorem can be used to estimate the parameters of a probability distribution or statistical model. Since Bayesian statistics treats probability as a degree of belief, Bayes theorem can directly assign a probability distribution that quantifies the belief to the parameter or set of parameters.Bayesian statistics is named after Thomas Bayes, who formulated a specific case of Bayes theorem in a paper published in 1763. In several papers spanning from the late 18th to the early 19th centuries, Pierre-Simon Laplace developed the Bayesian interpretation of probability. Laplace used methods that would now be considered Bayesian to solve a number of statistical problems. Many Bayesian methods were developed by later authors, but the term was not commonly used to describe such methods until the 1950s. During much of the 20th century, Bayesian methods were viewed unfavorably by many statisticians due to philosophical and practical considerations. Many Bayesian methods required much computation to complete, and most methods that were widely used during the century were based on the frequentist interpretation. However, with the advent of powerful computers and new algorithms like Markov chain Monte Carlo, Bayesian methods have seen increasing use within statistics in the 21st century.',
 'kernel density estimation@is a non-parametric way to estimate the probability density function of a random variable.Kernel density estimation is a fundamental data smoothing problem where inferences about the population are made, based on a finite data sample. In some fields such as signal processing and econometrics it is also termed the Parzen–Rosenblatt window method,after Emanuel Parzen and Murray Rosenblatt, who are usually credited with independently creating it in its current form. One of the famous applications of kernel density estimation is in estimating the class-conditional marginal densities of data when using a naive Bayes classifier, which can improve its prediction accuracy.',
 'learning to rank@is the application of machine learning, typically supervised, semi-supervised or reinforcement learning, in the construction of ranking models for information retrieval systems. Training data consists of lists of items with some partial order specified between items in each list. This order is typically induced by giving a numerical or ordinal score or a binary judgment (e.g. "relevant" or "not relevant") for each item. The goal of constructing the ranking model is to rank new, unseen lists in a similar way to rankings in the training data.',
 'relational database@digital database whose organization is based on the relational model of data',
 'activity recognition@filed of research related to recognizing the actions and goals of computer agents',
 'wearable computer@Small computing devices nowadays usually electronic that are worn under with or on top of clothing',
 'big data@information assets characterized by such a high volume velocity and variety to require specific technology and analytical methods for its transformation into value',
 'ensemble learning@in machine learning the use of multiple algorithms to obtain better predictive performance than from any of the constituent learning algorithms alone',
 "wordnet@is a lexical database of semantic relations between words in more than 200 languages. WordNet links words into semantic relations including synonyms, hyponyms, and meronyms. The synonyms are grouped into synsetswith short definitions and usage examples. WordNet can thus be seen as a combination and extension of a dictionary and thesaurus. While it isaccessible to human users via a web browser, its primary use is in automatic text analysis and artificial intelligence applications. WordNet was first created in the English language and the English WordNet database and software tools have been released under a BSD style license and are freely available for download from that WordNet website.",
 'medical imaging@technique and process of creating visual representations of the interior of a body',
 'deconvolution@algorithm-based process used to reverse the effects of convolution on recorded data',
 'Latent Dirichlet allocation@generative statistical model that allows sets of observations to be explained by unobserved groups that explain why some parts of the data are similar',
 'Euclidian distance@length of straight line that connects two points in a measurable space or in an observable physical space',
 'web service@service offered by an electronic device to another electronic device communicating with each other via the World Wide Web',
 'multi-task learning@form of machine learning where a model learns multiple tasks',
 'Linear separability@geometric property of a pair of sets of points in Euclidean geometry',
 'OWL-S@ontology defined using the Web Ontology Language OWL for describing Web Services.  It was designed to enable software agents to automatically discover invoke compose and monitor Web Services.',
 'Wireless sensor network@group of spatially dispersed and dedicated sensors for monitoring and recording',
 'Semantic role labeling@is the process that assigns labels to words or phrases in a sentence that indicates their semantic role in the sentence, such as that of an agent, goal, or result. It serves to find the meaning of the sentence. To do this, it detects the arguments associated with the predicate or verb of a sentence and how they are classified into their specific roles. A common example is the sentence "Mary sold the book to John." The agent is "Mary," the predicate is "sold" (or rather, "to sell,") the theme is "the book," and the recipient is "John." Another example is how "the book belongs to me" would need two labels such as "possessed" and "possessor" and "the book was sold to John" would need two other labels such as theme and recipient, despite these two clauses being similar to "subject" and "object" functions.',
 'Continuous-time Markov chain@stochastic process that satisfies the Markov property sometimes characterized as memorylessness',
 'Open Knowledge Base Connectivity@is a protocol and an API for accessing knowledge in knowledge representation systems such as ontology repositories and object–relational databases. It is somewhat complementary to the Knowledge Interchange Format that serves as a general representation language for knowledge. It is developed by SRI International\'s Artificial Intelligence Center for DARPA\'s High Performance Knowledge Base program (HPKB).',
 "Propagation of uncertainty@effect of variables' uncertainties or errors more specifically random errors on the uncertainty of a function based on them",
 'Fast Fourier transform@ON logN divide and conquer algorithm to calculate the discrete Fourier transforms',
 'Security token@peripheral device used to gain access to an electronically restricted resource',
 'Novelty detection@the identification of rare items events or observations which raise suspicions by differing significantly from the expected or majority of the data',
 'semantic grid@is an approach to grid computing in which information, computing resources and services are described using the semantic data model. In this model, the data and metadata are expressed through facts (small sentences), becoming directly understandable for humans. This makes it easier for resources to be discovered and combined automatically to create virtual organizations (VOs). The descriptions constitute metadata and are typically represented using the technologies of the Semantic Web, such as the Resource Description Framework (RDF). Like the Semantic Web, the semantic grid can be defined as"an extension of the current grid in which information and services are given well-defined meaning, better enabling computers and people to work in cooperation."This notion of the semantic grid was first articulated in the context of e-Science, observing that such an approach is necessary to achieve a high degree of easy-to-use and seamless automation, enabling flexible collaborations and computations on a global scale. The use of semantic web and other knowledge technologies in grid applications are sometimes described as the knowledge grid. Semantic grid extends this by also applying these technologies within the grid middleware. Some semantic grid activities are coordinated through the Semantic Grid Research Group of the Global Grid Forum.',
 'Knowledge extraction@creation of knowledge from structured and unstructured sources',
 'Computational biology@data-analytical and theoretical methods mathematical modeling and computational simulation techniques to the study of biological behavioral and social systems',
 'Web 2.0@World Wide Web sites that use technology beyond the static pages of earlier Web sites',
 'Network theory@study of graphs as a representation of relations between discrete objects',
 'Video denoising@process of removing noise from a video signal',
 'Quantum information science@interdisciplinary theory behind quantum computing',
 'Color quantization@is quantization applied to color spaces; it is a process that reduces the number of distinct colors used in an image, usually with the intention that the new image should be as visually similar as possible to the original image. Computer algorithms to perform color quantization on bitmaps have been studied since the 1970s. Color quantization is critical for displaying images with many colors on devices that can only display a limited number of colors, usually due to memory limitations, and enables efficient compression of certain types of images. The name "color quantization" is primarily used in computer graphics research literature; in applications, terms such as optimized palette generation, optimal palette generation, or decreasing color depth are used. Some of these are misleading, as the palettes generated by standard algorithms are not necessarily the best possible.',
 'social web@set relations linking people through the WWW',
 'entity linking@the task of assigning a unique identity to entities mentioned in text',
 'information privacy@topic regarding the appropriate collection use and dissemination of personal data in products and services and related legal and political issues',
 'random forest@statistical algorithm that is used to cluster points of data in functional groups',
 'cloud computing@form of Internet-based computing whereby shared resources software and information are provided to computers and other devices',
 'Knapsack problem@is a problem in combinatorial optimization: Given a set of items, each with a weight and a value, determine the number of each item to include in a collection so that the total weight is less than or equal to a given limit and the total value is as large as possible. It derives its name from the problem faced by someone who is constrained by a fixed-size knapsack and must fill it with the most valuable items. The problem often arises in resource allocation where the decision makers have to choose from a set of non-divisible projects or tasks under a fixed budget or time constraint, respectively. The knapsack problem has been studied for more than a century, with early works dating as far back as 1897. The name "knapsack problem" dates back to the early works of the mathematician Tobias Dantzig (1884–1956), and refers to the commonplace problem of packing the most valuable or useful items without overloading the luggage.',
 'Linear algebra@branch of mathematics that studies vector spaces',
 'batch processing@execution of a series of jobs without manual intervention',
 'rule induction@is an area of machine learning in which formal rules are extracted from a set of observations.The rules extracted may represent a full scientific model of the data, or merely represent local patterns in the data. Data mining in general and rule induction in detail are trying to create algorithms without human programming but with analyzing existing data structures.: 415- In the easiest case, a rule is expressed with “if-then statements” and was created with the ID3 algorithm for decision tree learning.: 7 : 348 Rule learning algorithm are taking training data as input and creating rules by partitioning the table with cluster analysis.: 7 A possible alternative over the ID3 algorithm is genetic programming which evolves a program until it fits to the data.: 2 Creating different algorithm and testing them with input data can be realized in the WEKA software.: 125 Additional tools are machine learning libraries for Python like scikit-learn.',
 'Uncertainty quantification@characterization and reduction of uncertainties in both computational and real world applications',
 'Computer architecture@set of rules and methods that describe the functionality organization and implementation of computer systems',
 'Best-first search@is a class of search algorithms, which explore a graph by expanding the most promising node chosen according to a specified rule. Judea Pearl described the best-first search as estimating the promise of node n by a "heuristic evaluation function f ( n ) {\displaystyle f(n)}which, in general, may depend on the description of n, the description of the goal, the information gathered by the search up to that point, and most importantly, on any extra knowledge about the problem domain."Some authors have used "best-first search" to refer specifically to a search with a heuristic that attempts to predict how close the end of a path is to a solution (or, goal), so that paths which are judged to be closer to a solution (or, goal) are extended first. This specific type of search is called greedy best-first search or pure heuristic search.Efficient selection of the current best candidate for extension is typically implemented using a priority queue. The A* search algorithm is an example of a best-first search algorithm, as is B*. Best-first algorithms are often used for path finding in combinatorial search. Neither A* nor B* is a greedy best-first search, as they incorporate the distance from the start in addition to estimated distances to the goal.',
 'Gaussian random field@is a random field involving Gaussian probability density functions of the variables. A one-dimensional GRF is also called a Gaussian process.An important special case of a GRF is the Gaussian free field. With regard to applications of GRFs, the initial conditions of physical cosmology generated by quantum mechanical fluctuations during cosmic inflation are thought to be a GRF with a nearly scale invariant spectrum.',
 'Support vector machine@set of methods for supervised statistical learning',
 'ontology language@formal language used to construct ontologies',
 'machine translation@use of software for language translation',
 'middleware@computer software that provides services to software applications',
 "Newton's method@algorithm for finding a zero of a function"]

exact = [get_author_ranking_exact_v2(query1,query2, index,result_top_10, tfidf=False, strategy="binary", normalized=True, norm_alpha=1) for query1,query2 in zip(queries1,queries2)]


approximate = [get_author_ranking_approximate_v2(query1,query2, index,result_top_10, tfidf=False, strategy="binary", normalized=True, norm_alpha=1) for query1,query2 in zip(queries1,queries2)]



exact_uniform = [get_author_ranking_exact_v2(query1,query2, index,result_top_10, tfidf=False, strategy="uniform", normalized=True, norm_alpha=1) for query1,query2 in zip(queries1,queries2)]


approximate_uniform = [get_author_ranking_approximate_v2(query1,query2, index,result_top_10, tfidf=False, strategy="uniform", normalized=True, norm_alpha=1) for query1,query2 in zip(queries1,queries2)]


#r="Exact binary MRR@10:"+ str(mean_reciprocal_rank(exact))+" / Approximate binary MRR@10:"+ str(mean_reciprocal_rank(approximate))+" / Exact binary MAP@10:"+ str(mean_average_precision(exact))+" / Approximate binary MAP@10:"+ str(mean_average_precision(approximate))+" / Exact binary MP@10:"+ str(mean_precision_at_n(exact, 10))+" / Approximate binary MP@10:"+ str(mean_precision_at_n(approximate, 10))+" / Exact binary MP@5:"+ str(mean_precision_at_n(exact, 5))+" / Approximate binary MP@5:"+ str(mean_precision_at_n(approximate, 5))+" // Exact uniform MRR@10:"+ str(mean_reciprocal_rank(exact_uniform))+" / Approximate uniform MRR@10:"+ str(mean_reciprocal_rank(approximate_uniform))+" / Exact uniform MAP@10:"+ str(mean_average_precision(exact_uniform))+" / Approximate uniform MAP@10:"+ str(mean_average_precision(approximate_uniform))+" / Exact uniform MP@10:"+ str(mean_precision_at_n(exact_uniform, 10))+" / Approximate uniform MP@10:"+ str(mean_precision_at_n(approximate_uniform, 10))+" / Exact uniform MP@5:"+ str(mean_precision_at_n(exact_uniform, 5))+" / Approximate uniform MP@5:"+ str(mean_precision_at_n(approximate_uniform, 5))
print("Exact binary MRR@10:", mean_reciprocal_rank(exact)," / Approximate binary MRR@10:", mean_reciprocal_rank(approximate)," / Exact binary MAP@10:", mean_average_precision(exact)," / Approximate binary MAP@10:", mean_average_precision(approximate)," / Exact binary MP@10:", mean_precision_at_n(exact, 10)," / Approximate binary MP@10:", mean_precision_at_n(approximate, 10)," / Exact binary MP@5:", mean_precision_at_n(exact, 5)," / Approximate binary MP@5:", mean_precision_at_n(approximate, 5)," // Exact uniform MRR@10:", mean_reciprocal_rank(exact_uniform)," / Approximate uniform MRR@10:", mean_reciprocal_rank(approximate_uniform)," / Exact uniform MAP@10:", mean_average_precision(exact_uniform)," / Approximate uniform MAP@10:", mean_average_precision(approximate_uniform)," / Exact uniform MP@10:", mean_precision_at_n(exact_uniform, 10)," / Approximate uniform MP@10:", mean_precision_at_n(approximate_uniform, 10)," / Exact uniform MP@5:", mean_precision_at_n(exact_uniform, 5)," / Approximate uniform MP@5:", mean_precision_at_n(approximate_uniform, 5))