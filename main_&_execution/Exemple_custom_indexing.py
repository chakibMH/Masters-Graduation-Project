# -*- coding: utf-8 -*-
import pandas as pd
import ast
from sentence_transformers import SentenceTransformer, util
from Embedding_functions import embedde_paper_phrases, embedde_single_query
from custom_faiss_indexer import sentence_indexFlatL2


######## test


# read papers data set

#papers = pd.read_csv("papers.csv")

# paper id for test

paper_id = 2075884494

df_sent = papers.loc[papers.id == paper_id, ['cleaned_abstract_sentences']]

abst_sen = df_sent.iloc[0,0]

# list of phrases
abst_sen_to_list = ast.literal_eval(abst_sen)

#eliminer les chaines vides
abst_sen_to_list = [e for e in abst_sen_to_list if e != '']






#lenght of sen embedding
dim = 768

# create custom  index
index = sentence_indexFlatL2(dim)




# embedde sentences

# sentence embedder for test

# to do : sci bert

embedder = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')


# embedde
flat_sentence_embeddings = embedde_paper_phrases(abst_sen_to_list, embedder)


# add array of embedded sentences to index
index.add_single_doc(flat_sentence_embeddings, paper_id)   

# # save index
# filename = 'index_sent'
# save_index(index, filename)
# # load the saved index
# new_index = load_index(filename)

#  search 

# query 

query = ["big bang theory"]

# embedding of query

emb_query = embedde_single_query(query, embedder)

print("query embedding shape : ",emb_query.shape)

index.search(np.array([emb_query]), 10)
