# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 19:39:42 2022

@author: HP
"""
import pickle
omport pandas as pd


with open("queries_100_tags_final.pkl", "rb") as f:
    queries2 = pickle.load(f)


with open("queries_100_defs_final.pkl", "rb") as f:
    queries = pickle.load(f)
    

# sen_index = load_index("arxive_scibert_sentences_strategy")

embedder = SentenceTransformer('allenai/scibert_scivocab_uncased')


data = pd.read_csv("ACM_papers.csv")
authors = pd.read_csv("ACM_authors.csv")


list_index_path = ["sen_index_sci_bert_100kb1b2","sen_index_sci_bert_155868_p2","sen_index_sci_bert_p3"]



# /****************************    start execution ***************************/

file_name = "relvents_auths_all_queries_acm_scibert_sent_min_false_withdef_hyrid"

get_relevant_authors(list_index_path, file_name,'min',False, 'hybrid',1000)


# #acm
data.rename(columns={'id_paper':'id'}, inplace=True)
authors.rename(columns={'author_id':'id'}, inplace=True)


# exact, approximate = execute(file_name)

exact, approximate = execute_without_def(file_name)


file_path = "final_results/scibert/bysent/scibert_sent_min_false_withdef_hyrid"

save_files(file_path)