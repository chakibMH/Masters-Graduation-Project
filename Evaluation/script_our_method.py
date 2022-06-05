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




def get_relevant_authors(file_name,strategy = 'sum',norm=True, transform_to_score_before=True):

    results_all_queries = pd.DataFrame()
    
    d_all_query = {}
    i=1
    l = len(queries)
    for q in queries:
        print('current query: ',q,' [{}/{}'.format(i,l)) 
        score_authors_dict = get_relevant_experts(q, sen_index, data, 
                                                  authors, embedder, strategy,norm,
                                                  transform_to_score_before)
    
        
        
        d_all_query[q] = score_authors_dict
        i+=1
    df = pd.DataFrame(d_all_query)
    all_authors = authors.id.values
    l = list(df.index)
    to_drop = []
    for i in l:
        if i not in all_authors:
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


# fist execute ranking_author_cluster.py (the one in  \PFE_CODE ), the main_ranking_2.py
# when you execute this function, a csv file will be created (must be created in \PFE_CODE\Evaluation )

# here I wrote the name of the right file for you (don't change it !!), same as the rest of the parameters

file_name = "relvents_auths_all_queries_sum_notNorm_tranToScoTrue"

get_relevant_authors(file_name,strategy = 'sum',norm=False, transform_to_score_before=True)


# now, execute the file Our_method (you will find it in \PFE_CODE\Evaluation )

exact, approximate = execute(file_name)

# to execute, you must creat a folders as : /final_results/Our_method/sum_notNorm_tranToScoTrue/

file_path = "final_results/Our_method/sum_notNorm_tranToScoTrue/sum_notNorm_tranToScoTrue"

save_files(file_path)


