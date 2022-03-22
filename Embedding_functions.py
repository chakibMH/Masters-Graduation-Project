# -*- coding: utf-8 -*-
import ast

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
        
        list_embd_phraes.append(emb_p)
        
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
    
    abst_sen = df_sents.iloc[0,0]
    
    # list of phrases
    list_of_phrases = ast.literal_eval(abst_sen)
    
    #eliminer les chaines vides
    list_of_phrases = [e for e in list_of_phrases if e != '']
    
    list_embd_phraes = []
    
    for p in list_of_phrases:
        
        # p is str
        emb_p = embedder.encode(p)
        
        list_embd_phraes.append(emb_p)
        
    return list_embd_phraes


