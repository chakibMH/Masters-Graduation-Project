# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 17:13:52 2022

@author: HP
"""

#pip install lxml
#pip install requests
#pip install beautifulsoup4


import requests
from bs4 import BeautifulSoup
import csv
from itertools import zip_longest

job_titles = []
company_names = []
links = []
reqr = []

req = requests.Session()


result = req.get("https://wuzzuf.net/search/jobs/?q=python&a=hpb")

src = result.content

#print(src)

soup = BeautifulSoup(src, "lxml")

job_title = soup.find_all("h2", {"class":"css-m604qf"})

#print(job_title)

company_name = soup.find_all("a", {"class":"css-17s97q8"})

#print(company_name)


for i in range(len(job_title)):
    job_titles.append(job_title[i].text)
    print("\n\n\n",job_title[i])
    links.append("https://wuzzuf.net"+job_title[i].find("a").attrs["href"])
    company_names.append(company_name[i].text)

#print(job_titles,company_names)

for link in links:
    result = requests.get(link)
    src = result.content
    soup = BeautifulSoup(src, "lxml")
    #print(soup)
    requirements  = soup.find("div",{"class","css-1t5f0fr"}).ul
    respon_text = ""
    for li in requirements.find_all("li"):
        respon_text += li.text+" | "
    respon_text = respon_text[:-2]
    reqr.append(respon_text)
    
#csv file

file_list = [job_titles,company_names,links,reqr]
exported = zip_longest(*file_list)

with open("/Users/HP/Documents/GitHub/PFE_CODE/jobs.csv","w") as myfile:
    wr = csv.writer(myfile)
    wr.writerow(["job title","copany name","links","requirements"])
    wr.writerows(exported)


'''

import cloudscraper
import requests
from bs4 import BeautifulSoup
import csv
from itertools import zip_longest

job_titles = []
company_names = []

scraper = cloudscraper.create_scraper() # returns a CloudScraper instance

src = scraper.get("https://www.scopus.com/authid/detail.uri?authorId=6506891776").text

print(src)
soup = BeautifulSoup(src, "lxml")
#print(soup)
job_title = soup.find_all("div", {"class":"info-field__label sc-els-info-field sc-els-info-field-s"})

#print(job_title)

#company_name = soup.find_all("a", {"class":"css-17s97q8"})

#print(company_name)


for i in range(len(job_title)):
    job_titles.append(job_title[i].text)
    #company_names.append(company_name[i].text)

#print(job_titles,company_names)
print(job_titles)
print(len(job_titles))

#csv file

file_list = [job_titles,company_names]
exported = zip_longest(*file_list)

with open("/Users/HP/Documents/GitHub/PFE_CODE/jobs.csv","w") as myfile:
    wr = csv.writer(myfile)
    wr.writerow(["job title","copany name"])
    wr.writerows(exported)
'''












