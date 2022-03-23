# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 20:45:09 2022

@author: HP
"""
import requests
import wikipediaapi

def wikidata_then_wikipedia(query):

    url = "https://www.wikidata.org/w/api.php"

    params = {
            "action" : "wbsearchentities",
            "language" : "en",
            "format" : "json",
            "search" : query
        }

    try:
        data = requests.get(url,params=params)
        
        info = data.json()["search"][0]["description"] 
        info = info.translate({ord(i): None for i in '(,"):;!?'})
        query = query+" "+info
    except:
        wiki_wiki = wikipediaapi.Wikipedia('en')
        page_py = wiki_wiki.page(query)
        
        if page_py.exists() == False:
            print("Invalid Input try again!{query : ",query," }")
        else:
            wiki_wiki = wikipediaapi.Wikipedia('en')
            
            summary = page_py.summary.replace("\n", " ")
            summary = summary.replace("\t", " ")
            summary = summary.replace("  ", "")
            query = page_py.title + " "+summary

    
    return query


def wikipedia_then_wikidata(query):
    
    wiki_wiki = wikipediaapi.Wikipedia('en')
    page_py = wiki_wiki.page(query)
    
    if page_py.exists() == False:
        url = "https://www.wikidata.org/w/api.php"

        params = {
                "action" : "wbsearchentities",
                "language" : "en",
                "format" : "json",
                "search" : query
            }

        try:
            data = requests.get(url,params=params)
            
            info = data.json()["search"][0]["description"] 
            info = info.translate({ord(i): None for i in '(,"):;!?'})
            query = query+" "+info
        except:
            print("Invalid Input try again!{query : ",query," }")
    else:
        wiki_wiki = wikipediaapi.Wikipedia('en')
        
        summary = page_py.summary.replace("\n", " ")
        summary = summary.replace("\t", " ")
        summary = summary.replace("  ", "")
        query = page_py.title + " "+summary

    return query

def new_queries(old_queries, method="wikidata"):
    queries = []

    if method == "wikidata":
        for q in old_queries:
            queries.append(wikidata_then_wikipedia(q))
    else:
        for q in old_queries:
            queries.append(wikipedia_then_wikidata(q))
    return queries


queries_ = ['cluster analysis', 'Image segmentation', 'Parallel algorithm', 'Monte Carlo method',
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

print(new_queries(queries_, method="wikidata"))
print("\n\n\n /***************************************/\n\n\n")
print(new_queries(queries_, method="wikipedia"))
