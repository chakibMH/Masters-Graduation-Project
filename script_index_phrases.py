# -*- coding: utf-8 -*-
""" 
  script to index the sentences.
  
  Once executed, it is faster to load the index by using load function of the custom_faiss_indexer module.

@author: chakib
"""
  
import pandas as pd

from sentence_transformers import SentenceTransformer, util

import time
#from custom_faiss_index import dataset_Indexer

# read the data set

dataset_path = "papers.csv"

papers = pd.read_csv(dataset_path)

# create an embedder
embedder = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')

# file name

filename = "roberta_emb_sentences_indexFlatL2"


s = time.time()
# index all the data set
index = dataset_Indexer(papers, embedder, filename)

e = time.time()

d = e-s
print("temps necessaire : ",d)
