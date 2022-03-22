# -*- coding: utf-8 -*-
import faiss
import pickle
import pandas as pd
from Embedding_functions import embedde_phrases_from_DataFrame


class sentence_indexFlatL2():
    
    
    
    def __init__(self, dim):
        
        # self._dim = dim
        self._meta = {}
        self._meta['dim'] = dim
        self._dim = dim
        self._faiss_index = faiss.IndexFlatL2(dim) 
    
    def add_single_doc(self, sen_batch, paper_id):
        """
        

        Parameters
        ----------
        sen_batch : list of 
            DESCRIPTION.
        paper_id : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        i = self._faiss_index.ntotal
        order = 0
        i = 1
        total = len(sen_batch)
        for v in sen_batch:
            print(" "*12+"Phrase [ ", i," / ", total, " ]")
            self._meta[i] = {'paper_id':paper_id,'order_in_paper':order}
            i += 1
            order += 1
            v = v.reshape((1,self._dim))
            self._faiss_index.add(v)
            
        def search(self, query, k=1000):
            """
            

            Parameters
            ----------
            query : numpay array
                embedding of the query.
            k : int, optional
                number of returned phrases. The default is 1000.

            Returns
            -------
            DataFrame with those columns:
                id_ph: Id of phrase
                paper_id: Id of paper
                dist_phrase_with_query: score between phrase and query

            """
            
            D, I = self._faiss_index.search(query, k)
            
            
            # create data frame for results
            
            data = []
            
            
            
            for id_ph, dist in zip(I,D):
                
                paper_id = self._meta[i]['paper_id']
                
                data.append([id_ph, paper_id, dist])
            
            df = pd.DataFrame(columns=["id_ph", "paper_id", "dist_phrase_with_query"], data=data)
            
            return df
            
            
def save_index(index, filename):
    """
 save the index in filename

    Parameters
    ----------
    index : custom index
        .
    filename : str
        .

    Returns
    -------
    None.

    """
    
    #save meta data
    
    with open(filename+"_metaData", 'wb') as f:
        p = pickle.Pickler(f)
        p.dump(index._meta)
    
    #save index
    faiss_indx = index._faiss_index
    faiss.write_index(faiss_indx, filename)
    
def load_index(filename):
    """
    load  index from filename


    Parameters
    ----------
    filename : str
        .

    Returns
    -------
    sen_ind : custom index
        .

    """
    
    #load meta data
    with open("."+filename+"_metaData", 'rb')  as f:
        u = pickle.Unpickler(f)
        meta = u.load()
        
    #load fais index
    i = faiss.read_index(filename)
    

    sen_ind = sentence_indexFlatL2(meta['dim'])
    sen_ind._meta = meta
    sen_ind._faiss_index = i
    
    return sen_ind
    

def dataset_Indexer(papers, embedder, filename):
    """
    
    Index all the dataset's papers, return and  save the index into filename
    
    must have   cleaned_abstract_sentences   column

    Parameters
    ----------
    papers : Dataframe
        data set of all the papers.
        must have cleaned_abstract_sentences  column
    filename : str
        filename.

    Returns
    -------
    index.

    """
    print(" start indexing ...")
    
    # get all papers' id
    
    all_ids = papers.id.values
    
    
    # get first id, to get the dim
    first_id = all_ids[0]
    
    
    # embedde first doc 
    df_sents = papers.loc[papers.id == first_id, ['cleaned_abstract_sentences']]
    
    sent_batch = embedde_phrases_from_DataFrame(df_sents, embedder)
    
    
    # get the dim
    dim = sent_batch[0].shape[0]
    
    # create custom index object
    index = sentence_indexFlatL2(dim)
    
    # add first paper' sentences
    
    index.add_single_doc(sent_batch, first_id)
    
    
    # add all other papers' sentences
    i = 2
    total = all_ids.shape[0]
    for paper_id in all_ids[1:]:
        
        print("## progression document : [ ", i, " / ", total, " ]")
        i += 1
        
        # get the data frame
        df_sents = papers.loc[papers.id == paper_id, ['cleaned_abstract_sentences']]
        
        sent_batch = embedde_phrases_from_DataFrame(df_sents, embedder)
        
        index.add_single_doc(sent_batch, paper_id)
        
    # save index
    
    save_index(index, filename)
    
    #return index
    return index
    
        
    
    
    
    
    
    
            




