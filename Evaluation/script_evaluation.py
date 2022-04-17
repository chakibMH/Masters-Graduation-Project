# -*- coding: utf-8 -*-

from sentence_transformers import SentenceTransformer
from evaluation_functions import retrieve_author_tags,load_data_and_authors,load_relevant_index,get_author_ranking_exact_v2,get_author_ranking_approximate_v2,mean_reciprocal_rank,mean_average_precision,mean_precision_at_n
import pandas as pd
#/***********************************************************************************/

embedder = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')


data_and_authors = load_data_and_authors()
data = data_and_authors[0]
authors = data_and_authors[1]

#index = load_relevant_index("separate_sbert")


# relvents_auths_all_queries = pd.read_csv("relvents_auths_all_queries.csv",index_col=0)


# relvents_auths = pd.read_csv("relvents_auths_all_queries.csv")

# list_ids_relevant=relvents_auths["id"].tolist()

# def retrieve_author_tags_new(authors, author_id):
  
#     author_id= int(author_id)
#     try:
#         return ast.literal_eval(authors[authors.id == author_id].tags.values[0])
#     except:
#         return {}


# for n in list_ids_relevant:
#     tags = [t['t'].lower() for t in retrieve_author_tags_new(authors,n)]
    
#     if tags:
#         b=1
#     else:
#         relvents_auths.drop(relvents_auths.index[relvents_auths['id'] == n], inplace=True)

# relvents_auths.to_csv('relvents_auths_all_queries_new.csv', index=False)

relvents_auths_all_queries = pd.read_csv("relvents_auths_all_queries_new.csv",index_col=0)

queries = relvents_auths_all_queries.columns.values.tolist()



#res = relvents_auths_all_queries["cluster analysis"].copy()

#df_results = pd.DataFrame(columns=["Query","Exact binary MRR@10","Approximate binary MRR@10","Exact binary MAP@10","Approximate binary MAP@10","Exact binary MP@10:","Approximate binary MP@10","Exact binary MP@5","Approximate binary MP@5","Exact uniform MRR@10","Approximate uniform MRR@10","Exact uniform MAP@10","Approximate uniform MAP@10","Exact uniform MP@10","Approximate uniform MP@10","Exact uniform MP@5","Approximate uniform MP@5"])


#i=1
#for q in queries:
    #print("query : ",q," / indice : ":i)
    #i=i+1
    
    #res = df_read[q].copy()
    # sort values
    #res.sort_values(inplace=True)
    # dict like cluster analysis' one
    #dic_q = res.to_dict()
    #result_top_10 = produce_authors_ranking_new(dic_q)[:10]
    
    #queries1 = [q]
    #queries2 = [q]



# import time
# start_time = time.time()

# exact = [get_author_ranking_exact_v2(query1, relvents_auths_all_queries,authors,k=50, strategy="uniform", normalized=False, norm_alpha=100, extra_term=10) for query1 in queries]
  
# approximate = [get_author_ranking_approximate_v2(query1, relvents_auths_all_queries,authors,embedder, k=50, strategy="uniform",normalized=False, norm_alpha=100, extra_term=10) for query1 in queries]

# print("--- %s seconds ---" % (time.time() - start_time))

import time
start_time = time.time()

exact2=[]
approximate2=[]
for query1 in queries:
    
    
    print("query : ",query1)
    exact2.append(get_author_ranking_exact_v2(query1, relvents_auths_all_queries,authors,k=50, strategy="uniform", normalized=False, norm_alpha=100, extra_term=10) )
    approximate2.append(get_author_ranking_approximate_v2(query1, relvents_auths_all_queries,authors,embedder, k=50, strategy="uniform",normalized=False, norm_alpha=100, extra_term=10))

print("--- %s seconds ---" % (time.time() - start_time))

exact=exact2
approximate=approximate2

#///////////////////////////**********************///////////////////////

#Save ranking

#///////////////////////////**********************///////////////////////


df_results = pd.DataFrame(columns=["Query","Exact","Approximate"])

i=0
for q in queries:
    dict = {"Query":q,"Exact":exact[i],"Approximate":approximate[i]}
    df_results = df_results.append(dict, ignore_index = True)
    i=i+1

import pandas as pd
df_results.to_csv("Our_method_rankig.csv")
#///////////////////////////**********************///////////////////////

#Save results

#///////////////////////////**********************///////////////////////
l=[]
b=[]
for s in range(len(exact)):
    
    l2={k:exact[s][k] for k in list(exact[s])[:10]}
    l.append(l2)
    b2={k:approximate[s][k] for k in list(approximate[s])[:10]}
    b.append(b2)

text = "Exact binary MRR@50:"+ str(mean_reciprocal_rank(exact))+" / Exact binary MRR@10:"+ str(mean_reciprocal_rank(l))+" / Approximate binary MRR@50:"+ str(mean_reciprocal_rank(approximate))+" / Approximate binary MRR@10:"+ str(mean_reciprocal_rank(b))+" / Exact binary MAP@50:"+ str(mean_average_precision(exact))+" / Exact binary MAP@10:"+ str(mean_average_precision(l)) +" / Approximate binary MAP@50:"+ str(mean_average_precision(approximate))+" / Approximate binary MAP@10:"+ str(mean_average_precision(b))+" / Exact binary MP@[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:"+ str(mean_precision_at_n(exact, list_n=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]))+" / Approximate binary MP@[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:"+ str(mean_precision_at_n(approximate, list_n=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]))

with open('Our_method_results.txt', 'w') as f:
    f.write(text)

#///////////////////////////**********************///////////////////////

#Save metrics

#///////////////////////////**********************///////////////////////
import math

df_results_eval = pd.DataFrame(columns=["Query",
                                        "Exact binary MRR@50",
                                        "Exact binary MRR@10",
                                        "Approximate binary MRR@50",
                                        "Approximate binary MRR@10",
                                        "Exact binary MAP@50",
                                        "Exact binary MAP@10",
                                        "Approximate binary MAP@50",
                                        "Approximate binary MAP@10",
                                        "Exact binary MP@5",
                                        "Exact binary MP@10",
                                        "Exact binary MP@15",
                                        "Exact binary MP@20",
                                        "Exact binary MP@25",
                                        "Exact binary MP@30",
                                        "Exact binary MP@35",
                                        "Exact binary MP@40",
                                        "Exact binary MP@45",
                                        "Exact binary MP@50",
                                        "Approximate binary MP@5",
                                        "Approximate binary MP@10",
                                        "Approximate binary MP@15",
                                        "Approximate binary MP@20",
                                        "Approximate binary MP@25",
                                        "Approximate binary MP@30",
                                        "Approximate binary MP@35",
                                        "Approximate binary MP@40",
                                        "Approximate binary MP@45",
                                        "Approximate binary MP@50"])

i=0
for q in queries:
    l=[]
    l.append(exact[i])
    b=[]
    b.append(approximate[i])
    
    l2=[{k:exact[i][k] for k in list(exact[i])[:10]}]
    b2=[{k:approximate[i][k] for k in list(approximate[i])[:10]}]
    dic = {"Query":q,"Exact binary MRR@50":  ( 0 if math.isnan( mean_reciprocal_rank(l)) else mean_reciprocal_rank(l)),"Exact binary MRR@10":  ( 0 if math.isnan( mean_reciprocal_rank(l2)) else mean_reciprocal_rank(l2)),"Approximate binary MRR@50":( 0 if math.isnan(mean_reciprocal_rank(b)) else mean_reciprocal_rank(b)),"Approximate binary MRR@10":( 0 if math.isnan(mean_reciprocal_rank(b2)) else mean_reciprocal_rank(b2)) ,"Exact binary MAP@50":( 0 if math.isnan(mean_average_precision(l)) else mean_average_precision(l)) ,"Exact binary MAP@10":( 0 if math.isnan(mean_average_precision(l2)) else mean_average_precision(l2)),"Approximate binary MAP@50":mean_average_precision(b),"Approximate binary MAP@10":mean_average_precision(b2),"Exact binary MP@5":mean_precision_at_n(l, list_n=[5]).get(5),"Exact binary MP@10":mean_precision_at_n(l, list_n=[10]).get(10),"Exact binary MP@15":mean_precision_at_n(l, list_n=[15]).get(15),"Exact binary MP@20":mean_precision_at_n(l, list_n=[20]).get(20),"Exact binary MP@25":mean_precision_at_n(l, list_n=[25]).get(25),"Exact binary MP@30":mean_precision_at_n(l, list_n=[30]).get(30),"Exact binary MP@35":mean_precision_at_n(l, list_n=[35]).get(35),"Exact binary MP@40":mean_precision_at_n(l, list_n=[40]).get(40),"Exact binary MP@45":mean_precision_at_n(l, list_n=[45]).get(45),"Exact binary MP@50":mean_precision_at_n(l, list_n=[50]).get(50),"Approximate binary MP@5":mean_precision_at_n(b, list_n=[5]).get(5),"Approximate binary MP@10":mean_precision_at_n(b, list_n=[10]).get(10),"Approximate binary MP@15":mean_precision_at_n(b, list_n=[15]).get(15),"Approximate binary MP@20":mean_precision_at_n(b, list_n=[20]).get(20),"Approximate binary MP@25":mean_precision_at_n(b, list_n=[25]).get(25),"Approximate binary MP@30":mean_precision_at_n(b, list_n=[30]).get(30),"Approximate binary MP@35":mean_precision_at_n(b, list_n=[35]).get(35),"Approximate binary MP@40":mean_precision_at_n(b, list_n=[40]).get(40),"Approximate binary MP@45":mean_precision_at_n(b, list_n=[45]).get(45),"Approximate binary MP@50":mean_precision_at_n(b, list_n=[50]).get(50)}
    df_results_eval = df_results_eval.append(dic, ignore_index = True)
    i=i+1
    
import pandas as pd
df_results_eval.to_csv("Our_method_metrics.csv")