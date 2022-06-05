# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 22:32:04 2022

"""
import pandas as pd
from sentence_transformers import SentenceTransformer, util

import ast


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


# load index sen_index

#load papers.csv(papers)
#load authors.csv(authors)



# relvents_auths_all_queries_sum_Norm_tranToScoTrue

def get_relevant_papers(file_name,strategy = 'sum',norm=True, transform_to_score_before=True):

    results_all_queries = pd.DataFrame()
    
    d_all_query = {}
    i=1
    l = len(queries)
    for q in queries:
        print('current query: ',q,' [{}/{}'.format(i,l)) 
        score_authors_dict = get_relevant_experts(q, sen_index, papers, 
                                                  authors, embedder, strategy,norm,
                                                  transform_to_score_before)
    
        
        
        d_all_query[q] = score_authors_dict
        i+=1
    df = pd.DataFrame(d_all_query)
    all_authors = authors.id.values
    l = list(df.index)
    to_drop = []
    for i in l:
        if i not in all_authors:
            # to drop
            to_drop.append(i)
    
    df.drop(to_drop, inplace=True)
    df.to_csv(file_name+".csv")
    

    
    
    relvents_auths = pd.read_csv(file_name+".csv")
    
    relvents_auths = relvents_auths.rename(columns={relvents_auths.columns[0]: 'id'})
    
    list_ids_relevant=relvents_auths["id"].tolist()
    
    
    def retrieve_author_tags_new(authors, author_id):
      
        author_id= int(author_id)
        try:
            return ast.literal_eval(authors[authors.id == author_id].tags.values[0])
        except:
            return {}
    
    
    for n in list_ids_relevant:
        tags = [t['t'].lower() for t in retrieve_author_tags_new(authors,n)]
        #print("* tags  : ",tags,"// id : ",n)
        
        if tags:
            b=1
        else:
            relvents_auths.drop(relvents_auths.index[relvents_auths['id'] == n], inplace=True)
    
    relvents_auths.to_csv(file_name+'_new.csv', index=False)



# for q in queries:
#     res = df[q].copy()
#     # sort values
#     res.sort_values(inplace=True)
#     # dict like cluster analysis' one
#     dic_q = res.to_dict()
    
    