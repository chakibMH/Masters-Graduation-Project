# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 14:46:15 2022

@author: chaki
"""

import pandas as pd
from collections import defaultdict
import pickle

def def_val():
    return []

#lecture des metrics 

our_met = pd.read_csv("our_method_metrics.csv", index_col=0)

original_met = pd.read_csv("original_method_metrics.csv", index_col=0)

metrics = ['Exact binary MRR@50', 'Exact binary MRR@10', 
               'Approximate binary MRR@50', 'Approximate binary MRR@10',
               'Exact binary MAP@50', 'Exact binary MAP@10',
       'Approximate binary MAP@50', 'Approximate binary MAP@10']


# clusters

#   poor [0-0.3]
#   medium [0.3-0.6]
#   good [0.6-0.7]
# exelent [0.7-1]
#   C1: our : exelent   their: poor
#   C2: our: poor       their: exelent
#   C3: our: poor       their: poor
#   C4: our: exelent    their: exelent
## ...etc

result = {}

for m in metrics:
    
    # clusters for that metric
    
    clusters = defaultdict(def_val)
    
    Dx = original_met[m].values
    Dy = our_met[m].values
    
    # for each metric we create 16 clusters
    
    num_req = 0
    
    
    # for each query
    for x,y in zip(Dx, Dy):
        # determiner le cluster
        
        if x < 0.3:
            original = 'p'
        elif x < 0.6:
            original = 'm'
        elif x < 0.7:
            original = 'g'
        else:
            original = 'e'
        
        if y < 0.3:
            our = 'p'
        elif y < 0.6:
            our = 'm'
        elif y < 0.7:
            our = 'g'
        else:
            our = 'e'
            
        cluster_name = original + '-' + our
        
        # query name
        
        q_name = our_met.iloc[num_req,0]
        
        num_req += 1
        
        clusters[cluster_name].append(q_name)
        
        
    result[m] = clusters            
        

# save  

# with open("clusters_Original_vs_Our_Method.pkl", "wb") as f:
#     p = pickle.Pickler(f)
#     p.dump(result)
