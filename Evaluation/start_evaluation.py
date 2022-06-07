############################## IMPORTS ############################################################
import pickle
from custom_faiss_index import *
from Our_method import *
from Ranking_author_cluster.py import *



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

with open("original_queries/original_batch_100_concepts_with_deff", "rb") as f:
    queries = pickle.load(f)


with open("original_queries/original_batch_100_concepts.pkl", "rb") as f:
    queries2 = pickle.load(f)
    
#############################################################################################





############################################################################################################
# important data (if already loaded comment this section)
#########################################################################################

#load index
sen_index = load_index("roberta_emb_sentences_indexFlatL2")

#load papers and authors
data = pd.read_csv("papers.csv")
authors = pd.read_csv("authors.csv")

#load embedder
embedder = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
############################################################################################






############################################################################################
#################################  start evaluation #################################################################
############################################################################################

file_name = "relvents_auths_all_queries_sum_notNorm_tranToScoFalse_mean"

get_relevant_authors(file_name,strategy = 'sum',norm=False,deff_type="mean",a = 0.7, b=0.3, transform_to_score_before=False)
# free some spaces
del(sen_index)
exact, approximate = execute(file_name)


file_path = "final_results/Our_method/sum_notNorm_tranToScoFalse_mean/sum_notNorm_tranToScoFalse_mean"

save_files(file_path)



