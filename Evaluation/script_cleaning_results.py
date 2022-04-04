# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 14:56:26 2022

@author: chaki

cleaning the result, o that only authors that are in dataset appears.
"""

import pandas as pd

relvents_auths_all_queries = pd.read_csv("relvents_auths_all_queries.csv",index_col=0)

# cleaning dataset

authors = pd.read_csv("authors.csv")

not_in = [i for i in relvents_auths_all_queries.index if i   not in authors.id.values]

df_read.drop(not_in, inplace=True)