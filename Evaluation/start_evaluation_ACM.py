# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 11:12:48 2022

@author: HP
"""
############################## IMPORTS ############################################################
import pickle
# from custom_faiss_index import *
from Our_method import *
# from Ranking_author_cluster import *



############################################################################################
################################# NOTES #################################################
#########   IF AN ERROR OCCUCS EXECUTE FILES SEPARATLY ##############################################
#########   YOU CAN EXECUTE SECTION BY SECTION MORE EASILY 
#####################################################################################################




##########################################################################################
# open the list of concepts with deff
# each concept is separated by @
# if already loaded comment this section
############################################################################################

with open("original_queries/top_100_queries_with_def_ACM_new.pkl", "rb") as f:
    queries = pickle.load(f)


with open("original_queries/top_100_queries_ACM_new.pkl", "rb") as f:
    queries2 = pickle.load(f)

#############################################################################################


queries2 = queries2[:3]


############################################################################################################
# important data (if already loaded comment this section)
#########################################################################################

#load index
# sen_index = load_index("arxive_scibert_separate_strategy_custom_index")

list_index_path = ["sen_index_sci_bert_100kb1b2","sen_index_sci_bert_155868_p2","sen_index_sci_bert_p3"]

#load papers and authors
data = pd.read_csv("ACM_papers.csv")
authors = pd.read_csv("ACM_authors.csv")

#load embedder
# embedder = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')

embedder = SentenceTransformer('allenai/scibert_scivocab_uncased')
############################################################################################




def get_relevant_authors_without_def_ACM(list_index_path,file_name,strategy = 'sum',norm=True, transform_to_score_before=True):

    results_all_queries = pd.DataFrame()
    
    d_all_query = {}
    i=1
    l = len(queries2)
    for q in queries2:
        print('current query: ',q,' [{}/{}'.format(i,l)) 
        # score_authors_dict = get_relevant_experts(q, sen_index, data, 
        #                                           authors, embedder, strategy,norm,
        #                                           transform_to_score_before, 2000)
        
        score_authors_dict = get_relevant_experts_multi_index(q, list_index_path, data, 
                                             authors, embedder, 
                                 strategy, norm)
        
        
        
        
        d_all_query[q] = score_authors_dict
        i+=1
    df = pd.DataFrame(d_all_query)
    all_authors = authors.id.values
    l = list(df.index)
    to_drop = []
    for i in l:
        if int(i) not in all_authors:
            # to drop
            to_drop.append(i)
    
    df.drop(to_drop, inplace=True)
    df.to_csv(file_name+".csv")
    

    
    
    relvents_auths = pd.read_csv(file_name+".csv")
    
    relvents_auths = relvents_auths.rename(columns={relvents_auths.columns[0]: 'id'})
    
    list_ids_relevant=relvents_auths["id"].tolist()
    
    
    def retrieve_author_tags_new(authors, author_id):
      
        author_id= int(author_id)
        try:
            return ast.literal_eval(authors[authors.id == author_id].tags.values[0])
        except:
            return {}
    
    
    for n in list_ids_relevant:
        tags = [t['t'].lower() for t in retrieve_author_tags_new(authors,n)]
        #print("* tags  : ",tags,"// id : ",n)
        
        if tags:
            b=1
        else:
            relvents_auths.drop(relvents_auths.index[relvents_auths['id'] == n], inplace=True)
    
    relvents_auths.to_csv(file_name+'_new.csv', index=False)





############################################################################################
#################################  start evaluation #################################################################
############################################################################################

file_name = "relvents_auths_all_queries_ACM_scibert_sent_without_def_sum_norm"

#get_relevant_authors(file_name,strategy = 'sum',norm=False,deff_type="hybrid",a = 0.7, b=0.3, transform_to_score_before=False)
get_relevant_authors_without_def_ACM(list_index_path,file_name,strategy = 'sum',norm=True, transform_to_score_before=True)
# free some spaces
# del(sen_index)
# exact, approximate = execute(file_name)
exact, approximate = execute_without_def(file_name)


file_path = "final_results/scibert/Original_method/Original_method_before_True_k100"

save_files(file_path)








# file_name = "relvents_auths_all_queries_arxive_scibert_separate_with_def_mean_norm_True"

# get_relevant_authors(file_name,strategy = 'sum',norm=False,deff_type="mean",a = 0.7, b=0.3, transform_to_score_before=True)

# exact, approximate = execute(file_name)

# file_path = "final_results/scibert/Def_mean_arxive_scibert_separate/Def_mean_arxive_scibert_separate_norm_True"

# save_files(file_path)














# file_name = "relvents_auths_all_queries_arxive_scibert_separate_with_def_hyb_0.5_0.5"

# get_relevant_authors(file_name,strategy = 'sum',norm=False,deff_type="hybrid",a = 0.5, b=0.5, transform_to_score_before=True)

# exact, approximate = execute(file_name)

# file_path = "final_results/scibert/Def_hyb_0.5_0.5_arxive_scibert_separate/Def_hyb_0.5_0.5_arxive_scibert_separate"

# save_files(file_path)

