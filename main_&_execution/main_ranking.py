# -*- coding: utf-8 -*-

""" 
  load the index of the sentences, then rank authors for a given query
  
  @author: chaki
  
  """
  
  
from custom_faiss_indexer import load_index
from Embedding_functions import embedde_single_query
from sentence_transformers import SentenceTransformer, util
from Ranking_author_cluster import get_relevant_experts

import pandas as pd

authors = pd.read_csv("authors.csv")
papers = pd.read_csv("papers.csv")

# this varable may change depending on location of index
filename = "roberta_emb_sentences_indexFlatL2"

# load index
index = load_index(filename)

# embedder for test
embedder = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')


# query for test

query = "cluster analysis"

# embedde the query
emb_q = embedde_single_query(query, embedder)

score_authors_dict = get_relevant_experts(query, index, papers, authors, embedder)


