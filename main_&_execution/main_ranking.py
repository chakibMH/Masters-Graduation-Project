# -*- coding: utf-8 -*-

""" 
  load the index of the sentences, then rank authors for a given query
  
  """
  
  
from custom_faiss_indexer import load_index

# this varable may change depending on location of index
filename = "emb_sentence_indexFlatL2"

index = load_index(filename)

