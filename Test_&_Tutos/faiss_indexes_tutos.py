# -*- coding: utf-8 -*-

"""different vectors of the faiss library.

    Each vector has it specifications. details, bellow.
    
    (maderthoumch ga3 khasou chwiya)
    
    -hna f le cas ta3na  flat index yakfi  psk la taille des vecteur << 1M
    - donc soit indexIDMAP  soit indexFlatL2
    
    faiss link:
        https://github.com/facebookresearch/faiss/wiki/Getting-started
        """
    
"""problems :
    
    -  add with ids is not working with    IndexIVFFlat
    
    """

import numpy as np
import faiss

### flat vector
### recherche exaustive
### can't insert vector with id and therefor can't return the id of vector (with reconstruct)
### 

#exemple of databse
dim = 700
nb_rows = 4000
database = np.random.random((nb_rows, dim)).astype('float32')


#query
# list of queries
nb_query = 10
list_query = np.random.random((nb_query, dim)).astype('float32')
# single query

single_query = np.random.random((1, dim)).astype('float32')






#           IndexFlatL2   :  flat index   (brut force)  ( Exact  Result  )

#Building a flat index and adding the vectors to it
####
#    The simplest version that just performs brute-force L2 distance search on them: IndexFlatL2 (brut force = exaustive search)

# do not require training (brut force)
# Impossible to add a vector with an ID, the IDs returned by a search are created sequentially when adding a new row.
# can reconstruct a vector from its ID


index_flatt = faiss.IndexFlatL2(dim)

# to check if this index needs training
print("is trained : ",index_flatt.is_trained)

index_flatt.add(database)                  # add vectors to the index

#total nb of rows
print("n total : ",index_flatt.ntotal)





#                       search   IndexFlatL2

# The basic search operation that can be performed on an index is the k-nearest-neighbor search,
# ie. for each query vector, find its k nearest neighbors in the database.

#The result of this operation can be conveniently stored in an integer matrix of size nq-by-k,
# where row i contains the IDs of the neighbors (id fhad le cas mahi ta3 sa7 mais adrhoum lindex sequentielement) of query vector i, 
# sorted by increasing distance.
# In addition to this matrix, the search operation returns a nq-by-k floating-point matrix with the corresponding squared distances.

k = 10
D, I = index_flatt.search(list_query, k)     #  search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries

#search first elements  
v_fist = database[0]

#reshape  from (700,)   to  (1,700)

v_fist = v_fist.reshape((1, dim))
D, I = index_flatt.search(v_fist, k)


#                Reconstruct    IndexFlatL2


# first row of the database (because it is inserted sequentially)
fist_vect = index_flatt.reconstruct(0)

#                Remove : this will change the numbering








###############       *****    *************    ***



#####################  FASTER   Search   (  not  exact Results  )

# use IndexIVFFlat index

# faster then flat indexes
# requires a training stage

# params :
    # require an index as param called quantizer, which  could be simply a flat index
    # dimension (nb of columns)
    # nlist : nb of cells (check more details)
    # nprob : nb of cells that are visited to perform a search
    
    


# more details :
        # To speed up the search, it is possible to segment the dataset into pieces (cells, nlist).
        # We define Voronoi cells in the d-dimensional space, and each database vector falls in one of the cells. 
        # At search time, only the database vectors y contained in the cell the query x falls in and a few 
        # neighboring ones are compared against the query vector.




############     Implementation  IndexIVFFlat

# nb cells (clusters)
nlist = 100
k = 4
quantizer = faiss.IndexFlatL2(dim)  # the other index
index = faiss.IndexIVFFlat(index_flatt, dim, nlist)
assert not index.is_trained
#train to define the cells
index.train(database)
assert index.is_trained

index.add(database)                  # add may be a bit slower as well

#            search
D, I = index.search(list_query, k)     # actual search
print(I[-5:])                  # neighbors of the 5 last queries
index.nprobe = 10              # default nprobe is 1, try a few more
D, I = index.search(list_query, k)
print(I[-5:])                  # neighbors of the 5 last queries



#         Reconstruct  IndexIVFFlat

#   For IVF (and IMI) indexes, before attempting to use the reconstruct method, we need to call the make_direct_map method 
    # - otherwise we will return a RunetimeError.
    
    
# in this case we get the sequentiel order   ---   we can add vectors with specific IDS
index.make_direct_map()
    
    
v = index.reconstruct(0)[:dim]
    
    
#          ADD   vectors  witth   ID

# array of ids (not working)
ids = np.arange(database.shape[0]).astype(np.int64)

#set first id to 6000
ids[0] = 6000

index.train(database)
index.add_with_ids(database, ids)



#####                         IndexIDMap    ==    like  flat   index   but support  add with  IDs


index = faiss.IndexFlatL2(database.shape[1]) 
ids = np.arange(database.shape[0]).astype(np.int64)
#set first id to 6000
ids[0] = 6000
#index.add_with_ids(database, ids)  # this will crash, because IndexFlatL2 does not support add_with_ids
index_MAP = faiss.IndexIDMap(index)
index_MAP.add_with_ids(database, ids) # works, the vectors are stored in the underlying index


# reconstruct  reconstruct not implemented for this type of index



    # ***************** ********************   ****************     **********
    #              other operations


# save index


# load index

