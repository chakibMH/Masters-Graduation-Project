# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 09:36:12 2022

@author: HP
"""

import pandas as pd

df=pd.read_csv("ACM_papers_DB.csv")

id_papers = df[df.columns[1]].values.tolist()

id_papers = list(dict.fromkeys(id_papers))

del df["index"]

df.reset_index(inplace=True)


df_new = df.drop(df[(df["author_subject_areas"] == "[]") | (df["author_keywords"] == "[]")].index)

g = df.loc[df['id_paper'] == "10.1145/3219166.3219193"]

aut=df.groupby(["author_name"])["author_name"].count()
aut.sort_values(ascending=False)


s=aut.loc[(aut > 100)].iloc[:300]
s.sort_values(ascending=False)
list_names_top_300_prolific = s.index
list_names_top_300_prolific = list(list_names_top_300_prolific)


p=aut.loc[(aut < 100) & (aut > 20)]
list_names_top_200_less_prolific = list(p.index)

import random


m=[]
# initializing the value of n
n = 200

# traversing and printing random elements
for i in range(n):
	
	# end = " " so that we get output in single line
	m.append(random.choice(list_names_top_200_less_prolific))
    
list_names_top_200_less_prolific = m

list_top_500 = list_names_top_300_prolific + list_names_top_200_less_prolific 



import pickle
with open('ex_id_papers.pkl', 'wb') as f:
   pickle.dump(id_papers, f)
   
   
   
   
   
   
import pickle
with open('ex_id_papers.pkl', 'rb') as f:
   id_papers= pickle.load(f)
   
   
l_a=[] 
   
for d in data_:
    for x in d[2]:
        l_a.append(x)
        
        
l_exist=[]


for f in l_a:
    if f in auts:
        l_exist.append(f)


for c in l_exist:
    if c in l_a:
        l_a.remove(c)

