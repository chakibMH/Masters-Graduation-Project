# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 11:45:26 2022

@author: chaki
"""

#sentence BERT Models
from sentence_transformers import SentenceTransformer, util

sen_model_Mini = SentenceTransformer('all-MiniLM-L6-v2')

sen_model_Roberta = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')

first_batch_sentences = ['The cat sits outside',
             'A man is playing guitar',
             'I love pasta',
             'The new movie is awesome',
             'The cat plays in the garden',
             'A woman watches TV',
             'The new movie is so great',
             'Do you like pizza?']

second_batch_sen = ['Football is an awesome activity',
                    'Football is an awesome game',
                    'Soccer is a wonderful sport',
                    'Football is an excellent sport',
                    'Soccer is a great game',
                    'A wonderful activity is soccer',
                    'A very good sport is for example football']

third_batch_sen = ['clustering',
                   'unsupervised classification']

def get_sim_score(sentences, model):

    embeddings = model.encode(sentences)
    
    cosine_scores = util.cos_sim(embeddings, embeddings)
    
    #Find the pairs with the highest cosine similarity scores
    pairs = []
    for i in range(len(cosine_scores)-1):
        for j in range(i+1, len(cosine_scores)):
            pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})
    #Sort scores in decreasing order
    pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
    for pair in pairs[0:10]:
        i, j = pair['index']
        print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], sentences[j], pair['score']))
                    