# -*- coding: utf-8 -*-

#/***********************************************************************************/

embedder = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
# res = faiss.StandardGpuResources()  # use a single GPU


data_and_authors = load_data_and_authors()
data = data_and_authors[0]
authors = data_and_authors[1]

index = load_relevant_index("separate_sbert")


df_read = pd.read_csv("relvents_auths_all_queries.csv",index_col=0)

queries = df_read.columns.values

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

    

exact = [get_author_ranking_exact_v2(query1,query1, index, tfidf=False, strategy="binary", normalized=True, norm_alpha=1) for query1 in queries]

    
approximate = [get_author_ranking_approximate_v2(query1,query1, index, tfidf=False, strategy="binary", normalized=True, norm_alpha=1) for query1 in queries]


exact_uniform = [get_author_ranking_exact_v2(query1,query1, index, tfidf=False, strategy="uniform", normalized=True, norm_alpha=1) for query1 in queries]


approximate_uniform = [get_author_ranking_approximate_v2(query1,query1,index, tfidf=False, strategy="uniform", normalized=True, norm_alpha=1) for query1 in queries]

#dict = {"Query":q,"Exact binary MRR@10": mean_reciprocal_rank(exact),"Approximate binary MRR@10":mean_reciprocal_rank(approximate),"Exact binary MAP@10":mean_average_precision(exact),"Approximate binary MAP@10":mean_average_precision(approximate),"Exact binary MP@10":mean_precision_at_n(exact, 10),"Approximate binary MP@10":mean_precision_at_n(approximate, 10),"Exact binary MP@5":mean_precision_at_n(exact, 5),"Approximate binary MP@5":mean_precision_at_n(approximate, 5),"Exact uniform MRR@10":mean_reciprocal_rank(exact_uniform),"Approximate uniform MRR@10":mean_reciprocal_rank(approximate_uniform),"Exact uniform MAP@10":mean_average_precision(exact_uniform),"Approximate uniform MAP@10":mean_average_precision(approximate_uniform),"Exact uniform MP@10":mean_precision_at_n(exact_uniform, 10),"Approximate uniform MP@10":mean_precision_at_n(approximate_uniform, 10),"Exact uniform MP@5":mean_precision_at_n(exact_uniform, 5),"Approximate uniform MP@5":mean_precision_at_n(approximate_uniform, 5)}
#df_results = df_results.append(dict, ignore_index = True)

print("Exact binary MRR@10:", mean_reciprocal_rank(exact)," / Approximate binary MRR@10:", mean_reciprocal_rank(approximate)," / Exact binary MAP@10:", mean_average_precision(exact)," / Approximate binary MAP@10:", mean_average_precision(approximate)," / Exact binary MP@10:", mean_precision_at_n(exact, 10)," / Approximate binary MP@10:", mean_precision_at_n(approximate, 10)," / Exact binary MP@5:", mean_precision_at_n(exact, 5)," / Approximate binary MP@5:", mean_precision_at_n(approximate, 5)," // Exact uniform MRR@10:", mean_reciprocal_rank(exact_uniform)," / Approximate uniform MRR@10:", mean_reciprocal_rank(approximate_uniform)," / Exact uniform MAP@10:", mean_average_precision(exact_uniform)," / Approximate uniform MAP@10:", mean_average_precision(approximate_uniform)," / Exact uniform MP@10:", mean_precision_at_n(exact_uniform, 10)," / Approximate uniform MP@10:", mean_precision_at_n(approximate_uniform, 10)," / Exact uniform MP@5:", mean_precision_at_n(exact_uniform, 5)," / Approximate uniform MP@5:", mean_precision_at_n(approximate_uniform, 5))

# import pandas as pd
# df_results.to_csv("our_method_evaluation_results.csv")


# eval_1=df_results['Approximate uniform MP@5'].tolist()
# eval_1 = np.array(eval_1) 
# eval_1 = [0 if pd.isna(x) else x for x in eval_1]
# pan = np.mean(eval_1)