# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 08:47:16 2022

@author: HP
"""

import pandas as pd
import ast
import random
from collections import Counter
import operator


def create_data_dz(filename):

    with open(filename+"_cleaned.txt", encoding="utf8") as file:
        lines = file.readlines()
        
    
    
    list_final = []
    
    for i in lines:
        list_final.append(i.split("\t"))
        
    titles = []
    abstracts = []
    tags = []
    authors = []
    
    import ast
    
    for p in list_final:
        
        try:
            titles.append(p[0])
        except:
            titles.append("")
            
        try:
            abstracts.append(p[1])
        except:
            abstracts.append("")
            
        try:
            tags.append(p[2])
        except:
            tags.append([])
        
        try:
            
            p[3]=p[3].replace("|first", "")
            p[3]=p[3].replace("|middle", "")
            p[3]=p[3].replace("|last", "")
            
            authors.append(p[3])
        except:
            authors.append([])
        
    
    
    
    
    # initialize data of lists.
    data = {'Title': titles,
    		'Abstract': abstracts,
            'Concepts': tags,
            'Authors' : authors}
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Print the output.
    
    
    df_cleaned = clean_DB(df)
    
    df_cleaned_with_titles = clean_titles_dz(df_cleaned)
    
    df_cleaned_with_titles.to_csv(filename+".csv", index=False)
    
    return df_cleaned_with_titles




def create_dict_tags(filename):
    
    # df = create_data_dz("setif")  
    
    df = pd.read_csv(filename+'.csv', index_col=0)
    
    
    keywords = df.author_keywords.values.tolist()
    
    keywords.pop(62022)
    
    List_keywords = []
    i=0
    for l in keywords:
        print(i)
        i=i+1
        if l != "[]" :
            l_new = ast.literal_eval(l) 
            for u in l_new:
               List_keywords.append(u.lower())
    
    # List_keywords = list(dict.fromkeys(List_keywords))
    
    
    subject_areas = df.author_subject_areas.values.tolist()
    
    
    subject_areas.pop(62022)
    List_subject_areas = []
    i=0
    for l in subject_areas:
        print(i)
        i=i+1
        if l != "[]" :
            l_new = ast.literal_eval(l) 
            for u in l_new:
               List_subject_areas.append(u.lower())
    
    # List_subject_areas = list(dict.fromkeys(List_subject_areas))
    
    
    
    all_tags = List_subject_areas + List_keywords
    
    
    # all_tags_no_dup = list(dict.fromkeys(all_tags))
    
    
    
    length_counts = Counter(word for word in all_tags)
    tags_dict = dict(length_counts)
    
    sorted_d = dict( sorted(tags_dict.items(), key=operator.itemgetter(1),reverse=True))
    
    return sorted_d
    


cdt = create_dict_tags('total_not_clean_dup')


# import pickle
# with open('sorted_all_tags.pkl', 'wb') as f:
#    pickle.dump(sorted_d, f)
   

# qeries_100 = random.sample(all_tags, 100)

# qeries_100 = list(dict.fromkeys(qeries_100))



# with open('qeries_100.pkl', 'wb') as f:
#    pickle.dump(qeries_100, f)















