# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 16:11:19 2022

@author: HP
"""
import matplotlib.pyplot as plt
 

def save_plot(df_original,df_method):
    
    list_original = [] 
    for (columnName, columnData) in df_original.iteritems():
        if columnName != 'Unnamed: 0' and columnName != 'Query':
            list_original.append(columnData.values.tolist())
            
    list_method = [] 
    for (columnName, columnData) in df_method.iteritems():
        if columnName != 'Unnamed: 0' and columnName != 'Query':
            list_method.append(columnData.values.tolist())
    
    
    list_columns = ['Exact binary MRR@50', 'Exact binary MRR@10',
           'Approximate binary MRR@50', 'Approximate binary MRR@10',
           'Exact binary MAP@50', 'Exact binary MAP@10',
           'Approximate binary MAP@50', 'Approximate binary MAP@10',
           'Exact binary MP@5', 'Exact binary MP@10', 'Exact binary MP@15',
           'Exact binary MP@20', 'Exact binary MP@25', 'Exact binary MP@30',
           'Exact binary MP@35', 'Exact binary MP@40', 'Exact binary MP@45',
           'Exact binary MP@50', 'Approximate binary MP@5',
           'Approximate binary MP@10', 'Approximate binary MP@15',
           'Approximate binary MP@20', 'Approximate binary MP@25',
           'Approximate binary MP@30', 'Approximate binary MP@35',
           'Approximate binary MP@40', 'Approximate binary MP@45',
           'Approximate binary MP@50']
    
    i=0
    
    for metric in list_columns:
        x = list_original[i]
        y = list_method[i]
        i=i+1
       
        plt.scatter(x, y)
         
        plt.xlabel('Original method')
        plt.ylabel('Def method')
        plt.title(metric)
        name = metric.replace(" ", "_")
        plt.savefig("Plots/Original_DefMethod/"+name+".jpg")