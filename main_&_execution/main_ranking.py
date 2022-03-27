# -*- coding: utf-8 -*-

""" 
  load the index of the sentences, then rank authors for a given query
  
  """
  
  
from custom_faiss_indexer import load_index
from Embedding_functions import embedde_single_query
from sentence_transformers import SentenceTransformer, util
from Ranking_author_cluster import get_relevant_experts

# this varable may change depending on location of index
filename = "emb_sentence_indexFlatL2"

# load index
index = load_index(filename)

# embedder for test
embedder = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')


# query for test

query = "algorithms used in genetics"

# embedde the query
# emb_q = embedde_single_query(query, embedder)

score_authors_dict = get_relevant_experts(query, index, papers, authors, embedder)


