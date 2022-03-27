# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 17:06:43 2022

@author: HP
"""
import cloudscraper
import requests
from bs4 import BeautifulSoup
import csv
from itertools import zip_longest

job_titles = []
company_names = []

scraper = cloudscraper.create_scraper() # returns a CloudScraper instance

src = scraper.get("https://www.scopus.com/results/authorNamesList.uri?sort=count-f&src=al&sid=bac2aafb8ae4dd90999db0aac3ccb462&sot=al&sdt=al&sl=44&s=AUTHLASTNAME%28Elhadad%29+AND+AUTHFIRST%28Michael%29&st1=Elhadad&st2=Michael&orcidId=&selectionPageSearch=anl&reselectAuthor=false&activeFlag=true&showDocument=false&resultsPerPage=20&offset=1&jtp=false&currentPage=1&previousSelectionCount=0&tooManySelections=false&previousResultCount=0&authSubject=LFSC&authSubject=HLSC&authSubject=PHSC&authSubject=SOSC&exactAuthorSearch=false&showFullList=false&authorPreferredName=&origin=searchauthorfreelookup&affiliationId=&txGid=830e9161ff27c22dd087a0226b31e7ff").text

#print(src)

soup = BeautifulSoup(src, "lxml")
#print(soup)
paper_title = soup.find_all("td", {"class":"authorResultsNamesCol col20"})

text = str(paper_title)
pos = text.find("Elhadad, Michael")
#print(paper_title)

left = 'href=\"'
right = '\" title='

url_ = text[text.index(left)+len(left):text.index(right)]

print(url_)





# scraper = cloudscraper.create_scraper() # returns a CloudScraper instance

# url_ ="https://www.scopus.com/inward/authorDetails.uri?authorID=6506184054&partnerID=5ESL7QZV&md5=f165b4b58a8d94dc3760bbcd095d128a"
# src = scraper.get(url_).text


# soup = BeautifulSoup(src, "lxml")

# print(soup)
# # author_name = soup.find_all("div", {"class":"list-title margin-size-24-t margin-size-0-b text-width-32"})
# # print(author_name)

# text = str(soup)
# pos = text.find("Discover the most reliable, relevant, up-to-date research. All in one place.")

# print(pos)