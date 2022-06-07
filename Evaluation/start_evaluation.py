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


queries2 = ['cluster analysis', 'Image segmentation', 'Parallel algorithm', 'Monte Carlo method',
            'Convex optimization', 'Dimensionality reduction', 'Facial recognition system', 
            'k-nearest neighbors algorithm', 'Hierarchical clustering', 'Automatic summarization',
            'Dynamic programming', 'Genetic algorithm', 'Human-computer interaction', 'Categorial grammar', 
            'Semantic Web', 'fuzzy logic', 'image restoration', 'generative model', 'search algorithm',
            'sample size determination', 'anomaly detection', 'sentiment analysis', 'semantic similarity',
            'world wide web', 'gibbs sampling', 'user interface', 'belief propagation', 'interpolation', 
            'wavelet transform', 'transfer of learning', 'topic model', 'clustering high-dimensional data', 
            'game theory', 'biometrics', 'constraint satisfaction', 'combinatorial optimization', 'speech processing',
            'multi-agent system', 'mean field theory', 'social network', 'lattice model', 'automatic image annotation',
            'computational geometry', 'Evolutionary algorithm', 'web search query', 'eye tracking', 'query optimization',
            'logic programming', 'Hyperspectral imaging', 'Bayesian statistics', 'kernel density estimation',
            'learning to rank', 'relational database', 'activity recognition', 'wearable computer', 'big data', 
            'ensemble learning', 'wordnet', 'medical imaging', 'deconvolution', 'Latent Dirichlet allocation', 
            'Euclidean distance', 'web service', 'multi-task learning', 'Linear separability', 'OWL-S',
            'Wireless sensor network', 'Semantic role labeling', 'Continuous-time Markov chain', 
            'Open Knowledge Base Connectivity', 'Propagation of uncertainty', 'Fast Fourier transform', 
            'Security token', 'Novelty detection', 'semantic grid', 'Knowledge extraction', 
            'Computational biology', 'Web 2.0', 'Network theory', 'Video denoising', 'Quantum information science',
            'Color quantization', 'social web', 'entity linking', 'information privacy', 'random forest', 
            'cloud computing', 'Knapsack problem', 'Linear algebra', 'batch processing', 'rule induction', 
            'Uncertainty quantification', 'Computer architecture', 'Best-first search', 'Gaussian random field',
            'Support vector machine', 'ontology language', 'machine translation', 'middleware', 'Newton\'s method']

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

file_name = "relvents_auths_all_queries_mean_notNorm_mean"

get_relevant_authors(file_name,strategy = 'mean',norm=False,deff_type="mean",a = 0.7, b=0.3)


exact, approximate = execute(file_name)


file_path = "final_results/Our_method/mean_notNorm_mean/mean_notNorm_mean"

save_files(file_path)



