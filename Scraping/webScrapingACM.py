# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 20:41:28 2022
@author: HP
"""

import requests
from bs4 import BeautifulSoup
import csv
from itertools import zip_longest
import pandas as pd
import re
import ast
import unicodedata
from scraping_utility import match_name
import time


def get_data(name):
    
    def collect_abstracts(a):
        #get id author
        

        id = a[1]
            
        link="https://dl.acm.org"+id
        #print("id  : ",id)
        page_num = 0
        while True :
            link_all_papers="https://dl.acm.org"+id+"/publications?Role=author&pageSize=50&startPage="+str(page_num)
            
            #get all papers
            
            result = requests.get(link_all_papers)
            src = result.content
            soup = BeautifulSoup(src, "lxml")
            
            num = soup.find_all("span",{"class":"result__count"})
            
            num =str(num[0])
            
            left="<span class=\"result__count\">"
            right=" Results</span>"
            
            page_limit = int( num[num.index(left)+len(left):num.index(right)])
            #print(page_limit)
            
            if (page_num > page_limit // 50):
                #print("pages ended")
                break
            
            papers_title = soup.find_all("h5", {"class":"issue-item__title"})
        
            #print(papers_title)
            
            for i in range(len(papers_title)):
                papers_titles.append(papers_title[i].text)
                
                links_papers.append("https://dl.acm.org"+papers_title[i].find("a").attrs["href"])
                
                #dfObj = pd.DataFrame(columns=['paper_title', 'url', 'Action'])
            
            #print(len(papers_titles))  
            #print(len(links_papers))
            
            #get abstracts
            df = pd.DataFrame(columns=['title', 'abstract'])
            for i in range(len(links_papers)):
                result = req.get(links_papers[i])
        
                src = result.content
        
                #print(src)
        
                soup = BeautifulSoup(src, "lxml")
        
                abstract = soup.find_all("div", {"class":"abstractSection abstractInFull"})
                title = soup.find_all("h1",{"class":"citation__title"})
                for i in range(len(abstract)):
                    if i==0:
                        abs_ = abstract[i].text
                        abs_ = abs_.replace("\n", "")
                        
                        df2 = {'title': title[i].text, 'abstract': abs_}
                        df = df.append(df2, ignore_index = True)
            #abstracts.pop()
            
            page_num +=1
            #print("page switched")
        return df
        

    authors_list = []
    abstracts = []
    links = []
    papers_titles = []
    links_papers = []
    
    
    req = requests.Session()
    
    name_url = name.replace(" ", "+")
    
    result = req.get("https://dl.acm.org/action/doSearch?AllField="+name_url)
    
    src = result.content
    
    #print(src)
    
    
    soup = BeautifulSoup(src, "lxml")
    
    txt = soup.find_all("div", {"class","issue-item__content-right"})
    txt = str(txt)
    # pos = txt.find(name)
    
    # name_2 = strip_accents(name)
    # pos_2 = txt.find(name_2)
    authors_list_ = []
    authors_list_ = get_list_authors(txt)
    
    find_author(authors_list,name)
    a=find_author(authors_list_,name)
    
    if a != False :
        
        return collect_abstracts(a)
    else: 
        a  = find_author(authors_list_,strip_accents(name))
        if a != False :
            return collect_abstracts(a)
        else :
            print("there is no such an author !")
            dfObj = pd.DataFrame(columns=['title', 'abstract'])
            return dfObj
        
# def construct_csv(list_authors):
#     df_final = pd.DataFrame(columns=['author', 'papers'])
#     i=1
#     for a in list_authors:
#         print("author : ",i)
#         i=i+1
#         df = get_data(a)
#         #print(df)
#         list_papers = []
#         if df.empty == False :
            
#             for x in df.itertuples():
#                 list_papers.append([x.title,x.abstract])
            
#             if len(list_papers)==0:
#                 list_papers.append("No data available")
            
#         df_ = {'author': a, 'papers': list_papers}
#         df_final = df_final.append(df_, ignore_index = True)
        
#     #print(df_final)
#     df_final.to_csv("authors_data_ACM.csv", encoding='utf-8',index=False)
#     return df_final

def construct_csv(list_authors):
    
    i=1
    for a in list_authors:
        print("author : ",i)
        i=i+1
        df = get_data(a)
        #print(df)
        list_papers = []
        if df.empty == False :
            
            for x in df.itertuples():
                list_papers.append([x.title,x.abstract])
            
            if len(list_papers)==0:
                list_papers.append("No data available")
            
        list_final = [a, list_papers]
        with open(r'authors_data_ACM.csv', 'a', newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(list_final)
        


def get_real_npubs(authors, papers):
    
    auths = authors.id.values
    
    all_p_ids = papers.id.values
    
    real_number = []
    i=0
    for auth_id in auths:
        i+=1
        #print(i)
        
        p_ids = get_papers_of_author(auth_id, authors)
        
        cpt = 0
        
        for p in p_ids:
            
            if p in all_p_ids:
                cpt += 1
        
        real_number.append(cpt)
        
    
    df_final = pd.DataFrame({'author':auths, 'papers': real_number})
    
    return df_final
        
        
def find_author(authors_list,name):
    
    for a in authors_list:
        if  match_name(a[0],name) :
            return a
            break
    
    return False

def get_list_authors(txt):

    authors_list = []
    
    txt_len = len(txt)
    pos_deb = txt.find("href=\"/profile") 
    
    
    pos_fin = txt.find("><img alt=") 
    
    
    
    while pos_deb != -1 and pos_fin != -1 :
        
        text=txt[pos_deb:pos_fin]
    
        
        pos_deb_1 = text.find("/profile") 
    
    
        pos_fin_1 = text.find("\" title") 
    
    
        url=text[pos_deb_1:pos_fin_1]
    
        
        text_len = len(text) 
     
        pos_deb_1 = text.find("title") + 7
    
        name=text[pos_deb_1:(text_len-1)]
    
        authors_list.append([name,url])
        #*********************************************************#
        txt = txt[(pos_deb+text_len+6):(txt_len-1)]
    
        txt_len  = len(txt)
        
        pos_deb = txt.find("href=\"/profile") 
    
        pos_fin = txt.find("><img alt=") 
    
    
    return authors_list

def get_papers_of_author(auth_id, auth):
    
    auth_row = auth.loc[auth.id == auth_id,['pubs']]
    
    # list of dict 
    auth_papers = ast.literal_eval(auth_row.iloc[0,0])
    
    #list of papers id
    p_ids = [int(d['i']) for d in auth_papers]
    
    return p_ids
    

def get_authors_name(authors):
    
    return list(authors.name.values)
    

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')
#exemple
# #["Janez Brank", "Hoda Heidari","Eden Chlamtáč"]
# list_authors = ["Janez Brank", "Hoda Heidari","A.S.L.O Campanharo"]
# df = construct_csv(list_authors)

# df_all_authors = pd.read_csv ('authors.csv')
# all_names = df_all_authors.name
# list_authors = all_names.head(20)
# df = construct_csv(list_authors)
# names_l1 = ['Steven J. Benson', 'Prasoon Goyal', 'Michael O. Duff', 'Hui Tong', 'Justin B. Kinney', 
#             'Jingyan Wang', 'Xing Lin', 'Steven Riley', 'Noa Fish', 'Ahmed Touati', 'Malea Kneen', 
#             'Jacky Shunjie Zhen', 'David Degras', 'Mathias Unberath', 'Shebuti Rayana', 'Zenon W. Pylyshyn',
#             'Michael H R Stanley', 'Irene Aldridge', 'Thomas M. Walski', 'Chahira Lhioui', 'Barak Libai', 
#             'Michal Kempka', 'Yvonne Choquet-Bruhat', 'Jillian H. Fecteau', 'Craig R. Hullett', 'Andrzej Pelc',
#             'Sabine Glesner', 'Shengcai Liao', 'Marie Csete', 'David Knowles', 'Piyush Gupta', 'Heiko Dietze',
#             'Kerstin Hammernik', 'Artem Grotov', 'Luc Berthouze', 'John Miles Smith', 'Brian Michael Adams',
#             'Michele Benzi', 'Ruiji Fu', 'Hongliang Jin', 'Eric A. Wan', 'Ponnada A. Narayana', 'Craige Roberts',
#             'Mel Slater', 'Jiahao Chen', 'Susan J. Hespos', 'Bo Hedén', 'Guoxin Su', 'Julian Straub', 'Gerhard Paass', 
#             'Makbule Gulcin Ozsoy', 'Jingchen Liu', 'Probal Sengupta', 'Michael D. Wagner', 'Gintare Karolina Dziugaite',
#             'Emmanuel J. Cand', 'Seokju Lee', 'Oren Tsur', 'Alexander Schick', 'Eric L. Miller', 'Unsang Park', 
#             'Jianshu Weng', 'Michael Wibral', 'Yufei Li', 'Çağatay Demiralp', 'Alfonso Gerevini', 'Seif El-Din Bairakdar', 
#             'Lina Merchan', 'Alberto Fernández', 'A.O. Nicholls', 'Payman Mohassel', 'M. V. Garzelli', 'Geoffrey Sampson',
#             'Ali Shiravi', 'Victor Garcia', 'Nick C. Fox', 'Wen Zhang', 'Nikolaus Bee', 'Felix von Reischach', 'Peng Wang',
#             'Henrik Skibbe', 'Zhiyuan Chen', 'Yanan Fan', 'Xianghua Ying', 'Augustinos I. Dimitras', 'Kevin T. Kelly',
#             'Jasna Maver', 'Samah Jamal Fodeh', 'Ruth S. DeFries', 'Marius Lindauer', 'Hayman', 'T. W. Anderson',
#             'Ian Hacking', 'Francisco Fernández-Navarro', 'János Komlós', 'David McSherry', 'Jean-Rémi King', 
#             'Mauricio Santillana', 'Zhiqiang Tan', 'Di Zhao', 'Phillip D. Summers', 'Siddhartha Saha', 'Layla Oesper', 
#             'Peyman Milanfar', 'Piotr Bojanowski', 'Joachim Lambek', 'Andreas Weber', 'Kyle A. Beauchamp',
#             'Mikhail Prokopenko', 'Nikolas List', 'Alan R. Aronson', 'Jost-Hinrich Eschenburg', 'R. Duncan Luce', 
#             'Robert Spence', 'Chris Grier', 'Michael Schwarz', 'Amir H. Banihashemi', 'Dorthe Meyer', 'Qing Li', 
#             'Anika Gross', 'Gwo-Fong Lin', 'Wei He', 'Xu He', 'Abhyuday N Jagannatha', 'Themistoklis Charalambous', 
#             'Sylvia Glaßer', 'Eleanor Wong', 'Tianyi Jiang', 'Chee-Seng. Chow', 'Emmanouil Benetos', 'Peter List', 
#             'Francisco Javier Díez', 'Krishnakumar Balasubramanian', 'Weiwei Sun', 'Pitu B. Mirchandani', 
#             'Kirill Neklyudov', 'George Gargov', 'E. Llobet', 'Marco F. Huber', 'Yeong-Taeg Kim', 'James L. Peterson', 
#             'Feng Zhao', 'Stefano Palminteri', 'Roger Ratcliff', 'Piero Fariselli', 'S. Boukharouba', 'Ying Wen', 
#             'Mohammad Yaqub', 'Enrique Alfonseca', 'Haim H. Permuter', 'Si Wu', 'Keith Ball', 'Steve Chien', 'Arjun Jain',
#             'Tomasz Piotr Kucner', 'Pawan Nunthanid', 'SueYeon Chung', 'Georgios Theocharous', 'Vera Kurkova',
#             'René Ranftl', 'Julien Cornebise', 'Richard L. Bankert', 'James Theiler', 'Shing-Chow Chan', 'Karen Coyle',
#             'David Finch', 'Jorma Laaksonen', 'Alexander Kotov', 'Thomas Pfeil', 'Sachithra Hemachandra', 'David C. Burr',
#             'Kleomenis Katevas', 'Brent Wenerstrom', 'Ger Koole', 'Sonya Cates', 'Sanjeev Arora', 'Kui Jia', 
#             'Nicolas Sidère', 'Reinhard Selten', 'Hendrik P. Lopuhaä', 'Nick Yee', 'Carl E. Landwehr', 'Guido Governatori',
#             'Luc Berthouze', 'Vincenzo Nicosia', 'Zhuoran Yang', 'Valentina Zanardi', 'Loris Bazzani', 'E. Solak', 
#             'Mustapha Bouhtou', 'Anna Dreber', 'Narendra Ahuja', 'Michel Dumontier', 'Chong Yaw Wee', 'D. Guest', 
#             'Giorgio Tomasi', 'Jiawen Chen', 'Gordon Wetzstein', 'A. Hannachi', 'Anjuli Kannan', 'Mark A. Przybocki',
#             'Samy Bengio', 'Gerard A. Silvestri', 'Polina Mamoshina', 'Alexandru T. Balaban', 'Andrew S. Gordon', 
#             'Georges Le Goualher', 'Raciel Yera Toledo', 'Esca Tutorial Day', 'Joseph K. Bradley', 'Manchester Syntax',
#             'Rui Wang', 'Penelope Eckert', 'Tuncer C. Aysal', 'Abdelkader Mokkadem', 'Michael P. Wellman', 'Xiangbo Feng', 
#             'Alberto Camacho', 'Peter Buchholz', 'Vladimir Filkov', 'Dov Katz', 'Toby H. W. Lam', 'Lachlan L. H. Andrew',
#             'Ramazan Gençay', 'Chul Min Lee', 'Souneil Park', 'Harold P. Benson', 'Rasmus Kyng', 'Thomas Berg', 
#             'Mabkhout M. Al-Dousari', 'Fu Jie Huang', 'Marc Jeannerod', 'Ali Pesaranghader', 'N. Arumainayagam', 
#             'Ou Tan', 'Nachiketa Sahoo', 'Frédéric Ferraty', 'Jinwook Seo', 'Bart Bogaerts', 'Ernst Hairer', 
#             'Mirjam Minor', 'Antonio Monroy', 'Jue Wang', 'Rohini K. Srihari', 'Congzheng Song', 'Cooper Bills', 
#             'Xiao-Hu Yu', 'An Sing Chen', 'Ana C. Huamán Quispe', 'Babak Rahbarinia', 'Bruno Bouzy', 'Carlos A. Leon',
#             'Mohamed Ahmed', 'Minghui Liao', 'Amirreza Shaban', 'Bin Yang', 'Thomas D. Ndousse', 'Michael Morak', 
#             'Ahmad El Sallab', 'Wei-Chen Cheng', 'Ji Li', 'Anup Basu', 'S. S. Ravi', 'Michael A. Erdmann', 'A. Lapedes', 'Arash Mokhber', 'Connelly Barnes', 'T. Charles Clancy', 'Kevin Bleakley', 'Hao Quan', 'Emilie Kaufmann', 'Zeju Li', 'Rul Yamaguchi', 'Trinh Minh Tri Do', 'S. Esakkirajan', 'John T. Rickard', 'Olivier Catoni', 'Xavier Dutreilh', 'Ujjwal Maulik', 'Gert Storms', 'Stig Kjær Andersen', 'Jan P. H. van Santen', 'Gábor Melis', 'Michael Wooldridge', 'Jean van Heijenoort', 'Blai Bonet', 'Kevin Swersky', 'Jose San Pedro', 'Ohlsson Stellan', 'Xu Han', 'Piyasak Jeatrakul', 'Gultekin Ozsoyoglu', 'Aron Monszpart', 'C. K. Chow', 'Arnav Bhavsar', 'Ping Xuan', 'Robert R. Prechter', 'Daniel Tunkelang', 'Jacob Devlin', 'Ernesto A. B. F. Lima', 'Ben Tordoff', 'Markus Maurer', 'Sandra Fortini', 'Pedro Mendes', 'Gerald M. Reaven', 'Zhuoliang Kang', 'Biao Zhang', 'Yue Cao', 'Dario Figueira', 'Christian Meilicke', 'Mo Chen', 'Grant McKenzie', 'Ralf Der', 'Kristina Lerman', 'Xiaochun Cao', 'Matthias Strobbe', 'Hans Nyquist', 'Scott L. Painter', 'G. Sithole', 'Yoshinobu Kawahara', 'Ozan Sener', 'Junshan Zhang', 'Daofu Liu', 'Dirk Kraft', 'Xiang Ma', 'Chris Edwards', 'Jerry E. Pratt', 'Michael F. Barnsley', 'Roger A. Horn', 'S. James Press', 'Ari Holtzman', 'Leszek Pacholski', 'Andrea L. Bertozzi', 'George B. Dantzig', 'Philip J. Fleming', 'Ibrahim Mohamed Hassan Saleh', 'Raymond Phan', 'Ciprian Chelba', 'Sylvain Veilleux', 'Ron J. Weiss', 'Joydeep Ghosh', 'Katherine P. Liao', 'Minjie Wu', 'Goran Gogic', 'Lucas Fidon', 'Yuanjun Gao', 'Matteo Masotti', 'Joshua W. K. Ho', 'Kjetil Moløkken', 'Gregory A. Bryant', 'Stan Z. Li', 'Michael Hoey', 'Matthew J. C. Crump', 'Darrin C. Edwards', 'Ju-Chiang Wang', 'Bo Sun', 'Josip Krapac', 'Chenguang Wang', 'Mário S. Alvim', 'Min Zhao', 'Bruno Nicenboim', 'Lars Bretzner', 'Abhishek Bhattacharya', 'Nahla Ben Amor', 'Akshay Asthana', 'Pietro Ducange', 'Xiaorui Ma', 'Samaneh Moghaddam', 'Soonmin Bae', 'Guy Pierra', 'Jessica D. Tenenbaum', 'Luo-Luo Jiang', 'Beata Jarosiewicz', 'Anton Maximilian Schäfer', 'Samanwoy Ghosh-Dastidar', 'Rudolf Mayer', 'Sariel Har-Peled', 'S. Corrsin', 'Robert Oliver Castle', 'Jim Kleban', 'Gavin E. Crooks', 'Alan J. Laub', 'Paolo Raiteri', 'Kwang Won Sok', 'Ali Ghadirzadeh', 'Andrew J. Connolly', 'George E. Billman', 'Zexuan Zhu', 'Takuya Narihira', 'Samuel R. Buss', 'Pascal Bérard', 'Liping Jing', 'Prakhar Biyani', 'Sam Alxatib', 'William R. Murray', 'D. G. Horvitz', 'Lucia Wittner', 'Olle Häggström', 'R. Douc', 'Shaojun Wang']



#/************************************************************************/

#   run this part of the code only the first time you create the csv file 

#/***********************************************************************/

# fields=['author', 'papers']
# with open(r'authors_data_ACM.csv', 'a', newline='', encoding="utf-8") as f:
#     writer = csv.writer(f)
#     writer.writerow(fields)

#/****************************************************************/

list_authors = ['Emmanuel J. Cand', 'Seokju Lee', 'Oren Tsur']    
start = time.time()
construct_csv(list_authors)
end = time.time()
print("time: ",end - start)

