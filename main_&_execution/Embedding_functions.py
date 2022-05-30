# -*- coding: utf-8 -*-
import ast
import numpy as np
from sklearn.preprocessing import normalize

def embedde_paper_phrases(list_of_phrases, model):
    """
    Embedde phrases of a paper, using model. 
    You can give different model, for different output.

    Parameters
    ----------
    list_of_phrases : list 
        list of cleaned phrases.
    model : embedder , optional
        embedder to be applied. The default is embedder.

    Returns
    -------
    list_embd_phraes : list
        list of embedded phrases.

    """
    
    list_embd_phraes = []
    
    for p in list_of_phrases:
        
        # p is str
        emb_p = model.encode(p)
        
        # normalize vector
        
        # l2 is default norm
        norm_emb_p = np.float32(normalize([emb_p])[0])
        
        
        list_embd_phraes.append(norm_emb_p)
        
    return list_embd_phraes



def embedde_phrases_from_DataFrame(df_sents, embedder):
    """
    Embedde phrases of a paper, using embedder. 
    You can give different embedder, for different output.

    Parameters
    ----------
    list_of_phrases : DataFrame
        DataFrame of cleaned phrases.
    model : embedder , optional
        embedder to be applied. The default is embedder.

    Returns
    -------
    list_embd_phraes : list
        list of embedded phrases.

    """
    
    df_sents = df_sents.iloc[0,0]
    
    # list of phrases
    list_of_phrases = ast.literal_eval(df_sents)
    
    #eliminer les chaines vides
    list_of_phrases = [e for e in list_of_phrases if e != '']
    
    list_embd_phraes = []
    
    for p in list_of_phrases:
        
        # p is str
        emb_p = embedder.encode(p)
        
        
        # normalize 
        
        # l2 is default norm
        norm_emb_p = np.float32(normalize([emb_p])[0])
        
        list_embd_phraes.append(norm_emb_p)
        
    return list_embd_phraes



    


def embedde_single_query(query, embedder, norm=True):
    """
    

    Parameters
    ----------
    query : str
        query to be embedded.
    embedder : 
        embedding model.

    Returns
    -------
    Embedding of query --> shape (1, dim).

    """
    
    # emb_q = embedder.encode([query])
    
    emb_q = embedder.encode([query])[0]
    
    if norm:
    
        # l2 is default norm
        norm_query = np.float32(normalize(emb_q)[0])
    
    return np.array([norm_query])


def get_mean_embedding(paper_id, papers, embedder):
        """
        
    
        Parameters
        ----------
        paper_id : int
            Id of paper.
        papers : DataFrame
            paper data set.
        embedder : sentence embrdder
            Embedde paper's sentences to array.
    
        Returns
        -------
        Array
            mean of all ambedded sentences.
    
        """
        print("paper id : ",paper_id)
        df_sent = papers.loc[papers.id == paper_id, ['cleaned_abstract_sentences']]
        if df_sent.index.empty:
            #cant return None
            return np.zeros(1)
        else:
            abst_sen = df_sent.iloc[0,0]
            
            # list of phrases
            abst_sen_to_list = ast.literal_eval(abst_sen)
            
            
            flat_sentence_embeddings = embedde_paper_phrases(abst_sen_to_list, embedder)
            
            mean_emb = np.mean(flat_sentence_embeddings, axis=0)
            
            # normalize all the vector at the end
            
            mean_emb = mean_emb.reshape(1, -1)
            
            norm_mean_emb = np.float32(normalize(mean_emb)[0])
        
            return norm_mean_emb
    
    
    
    
def use_def_mean(q_concept, embedder, data_source = "wikidata_then_wikipedia"):
    
    
     concept_emb = embedde_single_query(q_concept, embedder, False)
     
     if data_source == "wikidata_then_wikipedia":
         q_def = wikidata_then_wikipedia(q_concept)
     elif data_source == "wikipedia":
         q_def = wikipedia(q_concept)
     elif data_source == "wikidata":
         q_def = wikidata(q_concept)
         
         
     q_def_emb = embedde_single_query(q_def, embedder, False)
     q_mean = (q_def_emb+ query_emb)/2
     norm_query = np.float32(normalize([emb_q])[0])
     
     return norm_query


def use_def_hybrid(q_concept, embedder, sen_index, data_source = "wikidata_then_wikipedia"):
    
    
    if data_source == "wikidata_then_wikipedia":
        q_def = wikidata_then_wikipedia(q_concept)
    elif data_source == "wikipedia":
        q_def = wikipedia(q_concept)
    elif data_source == "wikidata":
        q_def = wikidata(q_concept)
    
    concept_emb = embedde_single_query(q_concept, embedder, False)
    
    q_def_emb = embedde_single_query(q_def, embedder, False)
    
    df_concept = sen_index.search(concept_emb, 1000)
    
    df_def = sen_index.search(q_def_emb, 1000)
    

    
    