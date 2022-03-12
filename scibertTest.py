# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 12:10:08 2022

@author: HP
"""
from transformers import *

from Embeddings_test import get_sim_score

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
#model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

#tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased')
#model = AutoModel.from_pretrained('allenai/scibert_scivocab_cased')

#***************Vocabulary****************
'''
text_as_string =''
text = 'entreptise technologists may harbor doubts scalability performance list industry-specific platform opportunities continues grow'
print(text)

auto_tokes  = tokenizer.tokenize(text)
print(auto_tokes)
'''
#*****************************************


from sentence_transformers import SentenceTransformer, models
from torch import nn


word_embedding_model = models.Transformer('allenai/scibert_scivocab_uncased', max_seq_length=(512))


pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),\
                               pooling_mode_max_tokens=False, \
                               pooling_mode_mean_tokens=False,\
                               pooling_mode_mean_sqrt_len_tokens=True)
    
dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=768, activation_function=nn.Tanh())

sciBert_model = SentenceTransformer(modules=[word_embedding_model,pooling_model])

first_batch_sentences = ['The cat sits outside',
             'A man is playing guitar',
             'I love pasta',
             'The new movie is awesome',
             'The cat plays in the garden',
             'A woman watches TV',
             'The new movie is so great',
             'Do you like pizza?']

get_sim_score(first_batch_sentences, sciBert_model)

