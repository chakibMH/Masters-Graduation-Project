# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 19:59:08 2022

@author: HP
"""

#load index
sen_index = load_index("roberta_emb_sentences_indexFlatL2")

#load papers and authors
data = pd.read_csv("papers.csv")
authors = pd.read_csv("authors.csv")

#load embedder
embedder = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')


# fist execute ranking_author_cluster.py (the one in  \PFE_CODE ), the main_ranking_2.py
# when you execute this function, a csv file will be created (must be created in \PFE_CODE\Evaluation )

# here I wrote the name of the right file for you (don't change it !!), same as the rest of the parameters

file_name = "relvents_auths_all_queries_sum_notNorm_tranToScoTrue"
get_relevant_papers(file_name,strategy = 'sum',norm=False, transform_to_score_before=True)

# now, execute the file Our_method (you will find it in \PFE_CODE\Evaluation )

exact, approximate = execute(file_name)

# to execute, you must creat a folders as : /final_results/Our_method/sum_notNorm_tranToScoTrue/

file_path = "final_results/Our_method/sum_notNorm_tranToScoTrue/sum_notNorm_tranToScoTrue"

save_files(file_path)


