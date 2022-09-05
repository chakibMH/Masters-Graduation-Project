# -*- coding: utf-8 -*-
"""
Created on Tue May 31 20:30:26 2022

@author: chaki
"""

from sentence_transformers import SentenceTransformer, LoggingHandler
import logging
import pandas as pd
import ast
import numpy as np
from sklearn.preprocessing import normalize
import torch
import time
import logging
import faiss
# from BST import *
import pickle

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

model = SentenceTransformer('allenai/scibert_scivocab_uncased')
# model = SentenceTransformer('roberta-base-nli-stsb-mean-tokens') # working
# model = SentenceTransformer('allenai/roberta-base')

# indices for each id
ids_indices = {}
indices_in_order = []
sentences_in_order = []
cpt = 0
norm_emb = []
##################### normalization fct

def L2_norm(emb_list):
    norm_list = []

    for e in emb_list:
        
        norm_emb_e = np.float32(normalize([e])[0])
        norm_list.append(norm_emb_e)
        
    return norm_list

################# get start & end indices + sentences in order


def setup(x):
    
    global cpt
    global indices_in_order
    global sentences_in_order
    
    #fixe the indices
    x2list = ast.literal_eval(x)
    
    le = len(x2list)
    
    l = [cpt, cpt+le]
    
    cpt += le
    
    indices_in_order.append(l)
    
    #fixe the sentences
    
    sentences_in_order += x2list
    
#################################################


def start_indexing(dataset, sent_index):
    

    global cpt
    global indices_in_order
    global sentences_in_order 
    global ids_indices
    
    # init
    
    ids_indices = {}
    indices_in_order = []
    sentences_in_order = []
        
    
    cpt = 0
    
    # drop duplicate documents to avoid errors
    
    dataset = dataset.drop_duplicates(['id_paper'])
    
        
    
    #call setup
    
    print("setup...")
    dataset.cleaned_abstract_sentences.map(lambda x:setup(x))
    
    
    all_ids = dataset.id_paper.values
    
    # for p_id in all_ids:
    #     x = dataset.loc[dataset.id_paper == p_id, ['cleaned_abstract_sentences']]
    #     x = x.iloc[0,0]
    #     #fixe the indices
    #     x2list = ast.literal_eval(x)
        
    #     le = len(x2list)
        
    #     l = [cpt, cpt+le]
        
    #     cpt += le
        
    #     ids_indices[p_id] = l
        
    
    # create the dict
    ids_indices = {ki:vi for ki, vi in zip(all_ids, indices_in_order)}
    
    print("setup finished.")
    
    # start embedding
    
    
    
    print("start embedding...")

    emb = model.encode(sentences_in_order)


    print("embedding finished.")
    
    # normalize the embedding
    
    print("start L2 normalization...")
    
    norm_emb = L2_norm(emb)
    
    del(emb)
    
    print("normalization finished.")
    
    # index the results 
    

    
    print("start indexing...")
    
    #transform the cpu index to gpu index to make it faster
    
    # res = faiss.StandardGpuResources() 
    
    # sent_index._faiss_index = faiss.index_cpu_to_gpu(res, 0, sent_index._faiss_index)
    
    total = len(all_ids)
    n=1
    s = time.time()
    for p_id in all_ids:
        
        print("[ {} / {} ] ".format(n,total))
        
        ind_range = ids_indices[p_id]
        sent_index.add_single_doc(norm_emb[ind_range[0]: ind_range[1]], p_id)
        
        n+=1
    e = time.time()
    
    d = e-s
    d=d/60
    print("temps necessaire pour indexation de {} : {} min".format(total,d))
    
    print("indexing finished.")
    
    # back to cpu index for better compatibility with our hardware
    
    # sent_index._faiss_index = faiss.index_gpu_to_cpu(sent_index._faiss_index)
        
    # save the index
    
    return sent_index
        
        
def start_indexing_separate_strategy(dataset, sent_index):
    

    global cpt
    global indices_in_order
    global sentences_in_order 
    global ids_indices
    global norm_emb
    # init
    
    ids_indices = {}
    indices_in_order = []
    sentences_in_order = []
    norm_emb = []
    
    cpt = 0
    
    # drop duplicate documents to avoid errors
    
    dataset = dataset.drop_duplicates(['id_paper'])
    
        
    
    #call setup
    
    print("setup...")
    dataset.cleaned_abstract_sentences.map(lambda x:setup(x))
    
    
    all_ids = dataset.id_paper.values
    
    # for p_id in all_ids:
    #     x = dataset.loc[dataset.id_paper == p_id, ['cleaned_abstract_sentences']]
    #     x = x.iloc[0,0]
    #     #fixe the indices
    #     x2list = ast.literal_eval(x)
        
    #     le = len(x2list)
        
    #     l = [cpt, cpt+le]
        
    #     cpt += le
        
    #     ids_indices[p_id] = l
        
    
    # create the dict
    ids_indices = {ki:vi for ki, vi in zip(all_ids, indices_in_order)}
    
    print("setup finished.")
    
    # start embedding
    
    
    
    print("start embedding...")

    emb = model.encode(sentences_in_order)


    print("embedding finished.")
    
    # normalize the embedding
    
    print("start L2 normalization...")
    
    norm_emb = L2_norm(emb)
    
    del(emb)
    
    print("normalization finished.")
    
    
    
    
    # index the results 
    

    
    print("start indexing...")
    
    #transform the cpu index to gpu index to make it faster
    
    # res = faiss.StandardGpuResources() 
    
    # sent_index._faiss_index = faiss.index_cpu_to_gpu(res, 0, sent_index._faiss_index)
    
    total = len(all_ids)
    n=1
    s = time.time()
    # get the dim
    # dim = norm_emb[0].shape[1]
    # print("dim is ", dim)
    for p_id in all_ids:
        
        print("[ {} / {} ] ".format(n,total))
        
        ind_range = ids_indices[p_id]
        
        
        #create the average
        nb_sents = ind_range[1]-1-ind_range[0]

        

        if nb_sents != 0:
            # select the emb of sentences and sum them up
            absEmb = sum(norm_emb[ind_range[0]: ind_range[1]-1])
            # take the average of absEmb
            absEmb /= nb_sents
                                                
            # avg with emb of title (separate strategy)
            embTitle = norm_emb[ind_range[1]-1]
            sepVec = absEmb + embTitle
            sepVec /= 2
            
            #sepVec=sepVec.reshape(-1,)
            # index single vector
            sent_index.add_single_doc([sepVec], p_id)
        
        n+=1
        
    e = time.time()
    
    d = e-s
    d=d/60
    print("temps necessaire pour indexation de {} : {} min".format(total,d))
    
    print("indexing finished.")
    
    # back to cpu index for better compatibility with our hardware
    
    # sent_index._faiss_index = faiss.index_gpu_to_cpu(sent_index._faiss_index)
        
    # save the index
    
    return sent_index   


def update_log(le, ti):
    """
    

    Parameters
    ----------
    le : int
        how much embeddings.
    ti : float
        time spent to finish the job.

    add a new entry to log file.
    format of file is as follow: 
        le,ti

    """
    
    with open("ACM_authors_profiles/log_CUDA.txt", "a") as f:
        txt = str(le)+","+str(ti)+"\n"
        f.write(txt)
    
    

def construct_profile(id_p, sepVec, d_authors, bt):
    """
    

    Parameters
    ----------
    id_p: str
    sepVec : bp.array 
        Representing the embedding of document (separate strategy).
    l_authors : dict
        list of dict, containing all authors of this paper.
    bt:binaryTree
        
    add a new entry to author profile file, if it exists.
    otherwise, create the file, and add the new entry.
    format of file is as follow:
     {  id_p1: emb_v1}
     {  id_p2: emb_v2}
     {  id_p3: emb_v3}
        ...
     {  id_pn: emb_vn}
    """
    
    entry = {id_p:sepVec}
    
        
    d_authors = ast.literal_eval(d_authors)
    
    for dict_a in d_authors:
        # search id in a binary tree 
        # get the id
        a_id = dict_a['id']
        #transform to str
        a_id = str(a_id)
        
        sr = recursive_Tree_Search(bt, a_id)
        if sr == None:
            # this file doesn't exist (first appearence of this author)
            # create a new file with the author's id
            # add the vectors to this file
            l=[entry]
            with open("ACM_authors_profiles/"+a_id, "wb") as f:
                p = pickle.Pickler(f)
                p.dump(l)
            
            # add this id to bst
            bt = insert_BST(bt, a_id)
        else:
            # this file existe
            # load it
            with open("ACM_authors_profiles/"+a_id, "rb") as f:
                # add the new vector to his file
                u = pickle.Unpickler(f)
                l = u.load()
                
            # add new entry    
            l.append(entry)
            
            # save the updated user profile again
            with open("ACM_authors_profiles/"+a_id, "wb") as f:
                p = pickle.Pickler(f)
                p.dump(l)
    
        
    return bt

    
    

def index_profiles(dataset, step=10000):
    
    
    
    
    # read the start document number.
    with open("ACM_authors_profiles/current_doc.txt", "r") as f:
        c = f.read()
        c=int(c)
     
    
    # delimit the dataset    
    dataset = dataset.iloc[c:c+step]
    
    global cpt
    global indices_in_order
    global sentences_in_order 
    global ids_indices
    global norm_emb
    # init
    
    ids_indices = {}
    indices_in_order = []
    sentences_in_order = []
    norm_emb = []
    
    cpt = 0
    
    # drop duplicate documents to avoid errors
    
    dataset = dataset.drop_duplicates(['id_paper'])
    
        
    
    #call setup
    
    print("setup...")
    dataset.cleaned_abstract_sentences.map(lambda x:setup(x))
    
    
    all_ids = dataset.id_paper.values
    
    # for p_id in all_ids:
    #     x = dataset.loc[dataset.id_paper == p_id, ['cleaned_abstract_sentences']]
    #     x = x.iloc[0,0]
    #     #fixe the indices
    #     x2list = ast.literal_eval(x)
        
    #     le = len(x2list)
        
    #     l = [cpt, cpt+le]
        
    #     cpt += le
        
    #     ids_indices[p_id] = l
        
    
    # create the dict
    ids_indices = {ki:vi for ki, vi in zip(all_ids, indices_in_order)}
    
    print("setup finished.")
    
    # start embedding
    
    
    
    print("start embedding...")

    s = time.time()
    emb = model.encode(sentences_in_order)
    e = time.time()
    
    ti = e - s
    
    le = len(sentences_in_order)

    update_log(le, ti)

    print("embedding finished. --number of vectors: ", le,"--time spent: ",ti)
    
    # normalize the embedding
    
    print("start L2 normalization...")
    
    norm_emb = L2_norm(emb)
    
    del(emb)
    
    print("normalization finished.")
    
    
    # read the current binary tree
    with open("ACM_authors_profiles/bst_ids", "rb") as f:
        u = pickle.Unpickler(f)
        bt = u.load()
    
    
    
    total = len(all_ids)
    
    n=1
    
    for p_id in all_ids:
        
        print("[ {} / {} ] ".format(n,total))
        
        ind_range = ids_indices[p_id]
        
        
        #create the average
        nb_sents = ind_range[1]-1-ind_range[0]

        

        if nb_sents != 0:
            # select the emb of sentences and sum them up
            absEmb = sum(norm_emb[ind_range[0]: ind_range[1]-1])
            # take the average of absEmb
            absEmb /= nb_sents
                                                
            # avg with emb of title (separate strategy)
            embTitle = norm_emb[ind_range[1]-1]
            sepVec = absEmb + embTitle
            sepVec /= 2
            
            # add this vector to all the concerned authors
            l_authors = dataset.loc[dataset.id_paper == p_id, ['authors']]
            l_authors = l_authors.values[0][0]
            #l_authors = ast.literal_eval(l_authors)
            bt = construct_profile(p_id, sepVec,l_authors, bt)
        
        n+=1
        
    # update the current document number
    
    with open("ACM_authors_profiles/current_doc.txt", "w") as f:
        new_start = str(c+step)
        f.write(new_start)
        
    # save the bst after the modifictions that may have occured inside
    # construct_profile
    with open('ACM_authors_profiles/bst_ids', 'wb') as f:
        p = pickle.Pickler(f)
        p.dump(bt)   
        
        
def start_indexing_v2(dataset, sent_index, step=1000,strategy='ph'):
    """
    
    Allows :
        choosing strategy:
            * phrase indexing
            * separate strategy
        update logfile
        creating an index or not
        save mean emb of each doc in a dataframe

    Parameters
    ----------
    dataset : TYPE
        DESCRIPTION.
    sent_index : TYPE
        DESCRIPTION.
    step : TYPE, optional
        DESCRIPTION. The default is 1000.
    strategy : TYPE, optional
        DESCRIPTION. The default is 'ph'.

    Returns
    -------
    None.

    """
    
    
    # # read the start document number.
    # with open("ACM_authors_profiles/current_doc.txt", "r") as f:
    #     c = f.read()
    #     c=int(c)
     
    
    # # delimit the dataset    
    # dataset = dataset.iloc[c:c+step]
    
    global cpt
    global indices_in_order
    global sentences_in_order 
    global ids_indices
    global norm_emb
    # init
    
    ids_indices = {}
    indices_in_order = []
    sentences_in_order = []
    norm_emb = []
    
    cpt = 0
    
    # drop duplicate documents to avoid errors
    
    dataset = dataset.drop_duplicates(['id_paper'])    
    
    #call setup
    
    print("setup...")
    dataset.cleaned_abstract_sentences.map(lambda x:setup(x))
    
    
    all_ids = dataset.id_paper.values
    
        
    
    # create the dict
    ids_indices = {ki:vi for ki, vi in zip(all_ids, indices_in_order)}
    
    print("setup finished.")
    
    # start embedding
    
    
    
    print("start embedding...")

    s = time.time()
    emb = model.encode(sentences_in_order)
    e = time.time()
    
    ti = e - s
    
    le = len(sentences_in_order)

    update_log(le, ti)

    print("embedding finished. --number of vectors: ", le,"--time spent: ",ti)
    
    print("start L2 normalization...")
    
    norm_emb = L2_norm(emb)
    
    del(emb)
    
    print("normalization finished.")
    
    total = len(all_ids)
    
    n=1
    data=[]
    
    for p_id in all_ids:
        
        print("[ {} / {} ] ".format(n,total))
        
        ind_range = ids_indices[p_id]
        
        
        #create the average
        nb_sents = ind_range[1]-1-ind_range[0]
        
        # creating a mean emb for aech doc
        if nb_sents != 0:
            # select the emb of sentences and sum them up
            absEmb = sum(norm_emb[ind_range[0]: ind_range[1]-1])
            # take the average of absEmb
            absEmb /= nb_sents
                                                
            # avg with emb of title (separate strategy)
            embTitle = norm_emb[ind_range[1]-1]
            sepVec = absEmb + embTitle
            sepVec /= 2
            
            data.append({'id_paper':p_id, 'mean_embedding':str(sepVec)})
        
        n+=1
        
        if sent_index != None:
            if strategy == 'ph':
                # index phrases of each doc
                sent_index.add_single_doc(norm_emb[ind_range[0]: ind_range[1]], p_id)
            elif strategy == 'sep':
                #index only the mean emb for each doc
                #sepVec=sepVec.reshape(-1,)
                # index single vector
                sent_index.add_single_doc([sepVec], p_id)
            else:
                print("error : no other strategies")
                
                
    return    sent_index, pd.DataFrame(data)
    
    
    
    