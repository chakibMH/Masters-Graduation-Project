# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 23:10:59 2022

@author: HP
"""

#pip install arxiv

import arxiv
import csv
import pandas as pd

search = arxiv.Search(
  query = "au:Maite AND au:Taboada",   
  sort_by = arxiv.SortCriterion.SubmittedDate
)


with open('abstract.csv', 'w', newline='', encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["title","authors", "abstract","doi","primary_category","categories"])
    for result in search.results():
        writer.writerow([result.title,result.authors, result.summary,result.doi,result.primary_category,result.categories])



df = pd.read_csv("abstract.csv")
print(df.columns)
print(df.shape)

a=df["authors"]
print(a)

i=0
for j in a.values:
    if "Maite Taboada" in j:
        i=i+1
print(i)