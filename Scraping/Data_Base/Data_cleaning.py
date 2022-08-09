import pandas as pd
import string
import re
import os
from num2words import num2words
from math import log
import sys
import time
import ast
# from BST import insert_BST,recursive_Tree_Search
import traceback
# from wiktionaryparser import WiktionaryParser
from scraping_utility import match_name_from_list, match_name

# parser to search the wrd on wiktionary
# parser = WiktionaryParser()


# from nltk.corpus import words
   




# # Build a cost dictionary, assuming Zipf's law and cost = -math.log(probability).
# words = open("words-by-frequency.txt").read().split()
# wordslist = open("wordlist.txt").read().split()

# # # # check_dict = set(words+wordslist)



# wordcost = dict((k, log((i+1)*log(len(words)))) for i,k in enumerate(words))
# maxword = max(len(x) for x in words)

# # #################################
# cpt = 0


# change = {}
# change_from_wiki = []

# exceptions_list = []

# cleaned_abs_serie = pd.Series()


# # ########################################
# # list of 400k+ english words
# check_dict_400k = open("check_dict_400k.txt").read().split()
# # transform chek dict from list to binary tree
# tree_root = None
# i = 1
# l= len(check_dict_400k)
# for elt in check_dict_400k:
#     print(" [ {} / {} ]".format(i,l))
#     i+=1
#     tree_root = insert_BST(tree_root, elt.lower())
####################################################

def get_description(df):
    
    # df = pd.read_csv(filename)
    
    # df = df.drop_duplicates()
    
    print("columns : ",df.columns,"\n")
    
    
    print("shape : ",df.shape,"\n")
    
    
    
    
    no_abst = df.loc[df.abstract=='No abstract available.'].shape[0]
    
    total_articles = df.shape[0]
    
    print("no abstract : ",no_abst*100/total_articles, " %\n")
    
    v = df.loc[df.index_terms=='[]'].shape[0]
    
    print("no article tags (tree) : ", v*100/total_articles, " %\n")
    
    
    df.groupby(by="author_name") 
    
    g=df.groupby(by=["author_name"])["author_name"].count()
    
    total_authors = g.shape[0]
    
    print("number of authors: ", total_authors)
    
    
    
    no_tags=len(set(df.loc[df.author_subject_areas == '[]'].author_name.values))
    
    print("no authors tags : ", no_tags*100/total_authors, " %\n")
    
    
    # authors with no tags stats
    
    # df_notags = df.loc[df.author_subject_areas == '[]']
    
    # g_no_tags = df_notags.groupby(by=["author_name"])["author_name"].count()
    
    # g_no_tags.mean()
    
    
    
    print("stats : \n",
    "median: ", g.median() ,"\n",
    "mean: ",g.mean(), "\n",
    "Q1: ",g.quantile(0.25), "\n",
    "Q3: ",g.quantile(0.75), "\n",
    "min: ",g.min(), "\n",
    "max: ", g.max(), "\n")
    

def remove_empty_abstract(df):
    """
    

    Parameters
    ----------
    df : DataFrame
        .

    Returns
    -------
    DataFrame
        .

    """
    
    df = df.drop(df.loc[df.abstract == 'No abstract available.'].index)
    
    df.dropna(subset=['abstract'], inplace=True)
    
    return df


def remove_no_tags_authors(df):
    """
    

    Parameters
    ----------
    df : DataFrame
        DESCRIPTION.

    Returns
    -------
    DataFrame
        DESCRIPTION.

    """
    
    return df.drop(df[df.author_subject_areas == '[]'].index)


CUSTOM_STOPS = {"abstract", "background", "background.", "abstract.", "method",
                "result", "conclusion", "conclusions", "discussion", "PRON", "registration", "url"}

# punctuation = [’!"#$%&()*+,-./:;<=>?@[]\ˆ ‘{}| \’
# remove extra spaces

def remove_extra_spaces(txt):
    """
    

    Parameters
    ----------
    txt : str
        

    Returns
    -------
    str
        

    """
    
    return " ".join(txt.split())


# revome extra ./,# \n
# transform to miniscule
# consider what's in parentheses as a separate sentence



def remove_url(txt):
    
    regex_url = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"

    return re.sub(regex_url, "", txt)

def remove_email(txt):
    
    regex_email = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    return re.sub(regex_email, "", txt)

def remove_numbers(txt):
    
    regex_num = r"\([0-9]+\)"
    
    return re.sub(regex_num, "", txt)





def infer_spaces(s):
    """Uses dynamic programming to infer the location of spaces in a string
    without spaces."""

    # Find the best match for the i first characters, assuming cost has
    # been built for the i-1 first characters.
    # Returns a pair (match_cost, match_length).
    def best_match(i):
        candidates = enumerate(reversed(cost[max(0, i-maxword):i]))
        return min((c + wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)

    # Build the cost array.
    cost = [0]
    for i in range(1,len(s)+1):
        c,k = best_match(i)
        cost.append(c)

    # Backtrack to recover the minimal-cost string.
    out = []
    i = len(s)
    while i>0:
        c,k = best_match(i)
        assert c == cost[i]
        out.append(s[i-k:i])
        i -= k

    return " ".join(reversed(out))






def correct_words(s):
    
    sen2words = s.split(" ")
    
    
    
    res_list = []
    
    global change 
    global change_from_wiki
    global exceptions_list
    global cpt
    
    for w in sen2words:
        
        # search in list
        # if w in check_dict_400k  :
        if w != '':
        
            # search with BST
            if recursive_Tree_Search(tree_root,w) != None:
        
                res_list.append(w)
            
    
                
            else:
                
                try:
                    l_res = parser.fetch(w)
                except:
                    try:
                        l_res = parser.fetch(w)
                    except Exception as ex:
                        
                        exceptions_list.append((cpt,w))
                        print("An exception occurred")
                        print(traceback.format_exc())
                        l_res = []
                
                
                
                if l_res != []:
                    
                    res_list.append(w)
                    
                    change_from_wiki.append(w)
                    
                    
                    
                else:
                
                # correcting the word
                
            
                    w_after = infer_spaces(w)
                    
                    change[w] = w_after
            
                    res_list.append(w_after)
    
            
            # this word is correct ( belongs to the english dictionary) 
            
     
            
            # if w in check_dict  :
            
            #     res_list.append(w)
            
    
                
            # else:
                
            #     # correcting the word
                
            #     w_after = infer_spaces(w)
                
            #     change[w] = w_after

        #     res_list.append(w_after)
        
    return " ".join(res_list)
    

    
def clean_abstract(abstract):
    
    global cpt 
    
    cpt += 1
    
    print("\n","*"*10,"actual doc: ",cpt,"\n")
    
        # lower case
    abstract = abstract.lower()
    
    #remove the first 'abstract' word
    
    abstract = abstract.replace("abstract","", 1)
    

    # treating numbers
    
    #abstract = remove_numbers(abstract)
    
    # reomve ,
    
    abstract = abstract.replace(",","")
    
    # replace numbers with words
    
    abstract = re.sub(r"(\d+\.\d+)", lambda x:num2words(x.group(0)), abstract)
    
    abstract = re.sub(r"(\d+)", lambda x:num2words(x.group(0)), abstract)
    
    
    
    # remove email
    
    abstract = remove_email(abstract)
    
    # remove url
    
    abstract = remove_url(abstract)
    
    #remove extra spaces
    
    #abstract = remove_extra_spaces(abstract)
    

    

    # separer les phrases
    list_sen = abstract.split(".")
    
    cleaned_abstract = []
    
    for s in list_sen:
        
        
        if s != '':
            
            # replace the - with a space
            
         #    s = s.replace('-', ' ')
         # #   s = s.replace('°', ' degree ')
                
         #    # remove ponctuation ### 
         #    ### change it to space
            
         #    punc = string.punctuation + '—'+'–'+'−'+'“'+'∼'+'®'+'’'+'”'
            
         #    s = s.translate(str.maketrans('', '', punc))
         
             
            s = remove_punc(s)
            
            s = remove_extra_spaces(s)
            
            # correct words
            
            s = correct_words(s)
            
            s = remove_extra_spaces(s)
                
            cleaned_abstract.append(s)
      
            
    
    return cleaned_abstract
            
            
            
def remove_punc(s):
    
    punc = string.punctuation + '—'+'–'+'−'+'“'+'∼'+'®'+'’'+'”'
    
    
    s_res = ''
    
    
    for c in s:
        if c == '°':
            s_res += ' degree '
        elif c in punc:
            s_res  += ' '        
        else:
            s_res += c
            
            
    return s_res
    
def clean_DB(df):
    
    global cleaned_abs_serie
    
    start = time.time()
    
    # redirict prints to a file

    # sys.stdout = open("clean_log.txt", "w")
    
    #remove duplicates
    
    df = df.drop_duplicates()
    
    # remove rows with no abstract
    
    df = remove_empty_abstract(df)
    
    #df = remove_no_tags_authors(df)
    
    
    #cleaning absttract
    
    cleaned_abs_serie = df.abstract.map(lambda x:clean_abstract(x))
    
    df['cleaned_abstract_sentences'] = cleaned_abs_serie.copy()
    
    
    # in case rediricting prints
    
    # sys.stdout.close()
    
    end = time.time()
    
    print("time : ",end-start)
    
    global cpt
    
    cpt = 0
    
    return df

def clean_single_title(title):
    
    title = title.lower()
    
    #remove the first 'abstract' word
    
    

    # treating numbers
    
    #abstract = remove_numbers(abstract)
    
    # reomve ,
    
    title = title.replace(",","")
    
    # replace numbers with words
    
    title = re.sub(r"(\d+\.\d+)", lambda x:num2words(x.group(0)), title)
    
    title = re.sub(r"(\d+)", lambda x:num2words(x.group(0)), title)
    
    
    
    # remove email
    
    title = remove_email(title)
    
    # remove url
    
    title = remove_url(title)
    
    title = remove_punc(title)
    
    title = remove_extra_spaces(title)
    
    return title
     
    
def clean_titles(db_papers):
    
    db_papers['cleaned_title'] = db_papers.title.map(lambda x:clean_single_title(x)).copy()
    
    all_ids = db_papers.id_paper.values
    
    n=0
    
    len_all_ids = len(all_ids)
    for i in all_ids:
        print("[ {} / {}]".format(n, len_all_ids))
        n+=1
        tc = db_papers.loc[db_papers.id_paper == i, ['cleaned_title']].iloc[0,0]

        
    
        sen_list = db_papers.loc[db_papers.id_paper == i, ['cleaned_abstract_sentences']].iloc[0,0]
        sen_list = ast.literal_eval(sen_list)
        sen_list.append(tc)
        
        db_papers.loc[db_papers.id_paper==i, 'cleaned_abstract_sentences'] = str(sen_list)
    
    return db_papers
    

def merge_data_sets(folder):
    """
    

    Parameters
    ----------
    folder : str
        DESCRIPTION.

    Returns
    -------
    final_df : pandas Dataframe
        DESCRIPTION.

    """
    
    list_files = os.listdir(folder)
    
    list_df = []
    for f in list_files:
        list_df.append(pd.read_csv(folder+"/"+f))
    
    
    #print(sum([df.shape[0] for df in list_df]))
    
    final_df = pd.concat(list_df, axis=0)
    
    final_df.reset_index(inplace=True)
    
    return final_df


def merge_clean_data_sets(folder="clean"):
    """
    

    Parameters
    ----------
    folder : str
        DESCRIPTION.

    Returns
    -------
    final_df : pandas Dataframe
        DESCRIPTION.

    """
    
    list_files = os.listdir(folder)
    
    list_df = []
    for f in list_files:
        list_df.append(pd.read_csv(folder+"/"+f))
    
    
    #print(sum([df.shape[0] for df in list_df]))
    
    final_df = pd.concat(list_df, axis=0)
    
    final_df = final_df[['id_paper', 'title', 'abstract',
       'paper_citation', 'revue', 'index_terms', 'author_name',
       'author_average_citation_per_article', 'author_citation_count',
       'author_publication_counts', 'author_publication_years',
       'papers_available_for_download', 'author_subject_areas',
       'author_keywords', 'cleaned_abstract_sentences']]
    
    #final_df.reset_index(inplace=True)
    
    return final_df



def get_word_len10(a):
    
    l_words = a.split(" ")
    
    return [w for w in l_words if 'z' in w.lower()]



def get_aut_db(db):
    """
    Generate a dataset with authors' informations

    Parameters
    ----------
    db : pandas.DataFrame
        

    Returns
    -------
    authors_df : pandas.DataFrame
        

    """
    
    names = db.author_name.values
    names = set(names)
    names = list(names)
    
    db.dropna(inplace=True)
    
    all_val = []
    col = ['author_name', 'author_average_citation_per_article',
                                        'author_citation_count', 'author_publication_counts',
                                        'author_publication_years', 'papers_available_for_download',
                                        'author_subject_areas', 'author_keywords']
    
    
    for n in names:
    
        print(n)
        a = db.loc[db.author_name == n,col].iloc[0]

        all_val.append(a.values)
        
    
    authors_df = pd.DataFrame(data=all_val, columns=col)
    
    return authors_df
# def merge_df(filename, dfs_list):
def count_nb_ph(x):
    
    x = ast.literal_eval(x)
    
    return len(x)
    
    

# def check_for_mostly_numeric_string(token):
#     """
#     Check whether the token is numerical.
    
#     :param token: A single text token.
    
#     :return A boolean indicating whether the token is a numerical.
#     """
#     int_chars = []
#     alpha_chars = []

#     for ch in token:
#         if ch.isnumeric():
#             int_chars.append(ch)
#         elif ch.isalpha():
#             alpha_chars.append(ch)

#     if len(int_chars) > len(alpha_chars):
#         return True
#     else:
#         return False
    
    
    
# remove e-mails (regex)

def clean_tag(list_tags):
    c_tags = []
    for t in list_tags:
        
        t = str(t).lower()
        # remove email
        
        t = remove_email(t)
        
        # remove url
        
        t = remove_url(t)
        
        t = remove_punc(t)
        
        t = remove_extra_spaces(t)
        
        t = remove_numbers(t)
    
        c_tags.append(t)
        
    return c_tags


def create_tags(x):
    
    # print(x.author_subject_areas)
    # print(x.author_keywords)
    
    sa = ast.literal_eval(x.author_subject_areas)
    
    kw = ast.literal_eval(x.author_keywords)
    
    tags = []

    if(type(sa) == str):
        c_words = clean_tag([sa])
        tags += c_words
    elif type(sa) == list:
        c_words = clean_tag(sa)
        tags += c_words
    else:
        print("wrong type")
        
    if(type(kw) == str):
        c_words = clean_tag([kw])
        tags += c_words
    elif type(kw) == list:
        c_words = clean_tag(kw)
        tags += c_words
    else:
        print("wrong type")
        
    if tags == []:
        tags.append("no tags")
        
    print("tags : ", tags)
    
    return tags
# acm_all_authors.drop(acm_all_authors.loc[acm_all_authors.author_keywords == 'author_keywords'].index, inplace = True)
# acm_all_authors['tags'] = acm_all_authors.apply(lambda x:create_tags(x), axis = 1)
# auth_db = db_papers[['author_name', 'author_average_citation_per_article',
#        'author_citation_count', 'author_publication_counts',
#        'author_publication_years', 'papers_available_for_download',
#        'author_subject_areas', 'author_keywords']].copy()
# auth_db.drop_duplicates(['author_name'], inplace=True)
    
# auth_db['tags']=auth_db.apply(create_tags, axis=1)

# selected_names = set()
# def norm_names(x):
#     """
#     Check if name match a name in the selected_names list, then return it.
#     If it doesn't match then it is the first time, return it.

#     Parameters
#     ----------
#     x : str
#         name.

#     Returns
#     -------
#     name.

#     """
#     global selected_names
    
auths4doc = {}
# debug =[]
#final_names

# def possible_change(n_candidate):
#     global


# def get_authors(x):
    
#     global auths4doc
#     p_id = x.id_paper
#     if p_id in auths4doc.keys():
#         n = x.author_name
#         list_names = auths4doc[p_id]
#         # add 1 more element to the list
#         # if is not matching a name already added
        
#         if match_name_from_list(list_names, n) == False:
#             auths4doc[p_id].append(n)
#         else:
#             print("name matching name: ",n," list : ",list_names)
#     else:
#         #create new entry
        
#         auths4doc[p_id] = [x.author_name]
        

# total_not_clean.apply(lambda x:get_authors(x), axis=1)     
        
# from difflib import SequenceMatcher
    
# def similar(a, b):
#     return SequenceMatcher(None, a, b).ratio()


###########
# script
#########
# get ids of authors from the base authors dataset

# df_auths4doc = pd.DataFrame(auths4doc.items(), columns=['id_paper', 'authors_name'])


# # create a unique id for each author

# v=auths4doc.values()
# v=list(v)
# names=[]
# for l in v:
#     for n in l:
#         names.append(n)

# set_names = set(names)
###################################################################################
##################################### PART  DOCUMENT   DZ ############################
###############################################################################


# actual_id = 0
# doc_ids4titles = {}
# set_titles = set()


# def title_to_id(x):
#     global doc_ids4titles

    
#     return doc_ids4titles[x]


# def define_id(dataset):

#     global actual_id
#     global doc_ids4titles 
#     global set_titles

#     print("config ids: start")
#     dataset_titles = dataset.Title.values.tolist()
#     print("config ids: finish")
#     for t in dataset_titles:
#         if t not in set_titles:
#             set_titles.add(t)
#             actual_id += 1
#             doc_ids4titles[t] = actual_id
    
#     dataset['id_paper'] = dataset.Title.map(lambda x:doc_ids4titles[x])
    
#     return dataset
    

# def define_id_multi_dataset(list_filnames):
    
#     global actual_id
#     global doc_ids4titles 
#     global set_titles
    
#     print("init")
    
#     actual_id = 0
#     doc_ids4titles = {}
#     set_titles = set()
    
#     print("init finish")

    
    
#     for fn in list_filnames:
#         print("start dataset : "+fn)
#         df = pd.read_csv(fn)
        
#         df_with_ids = define_id(df)
        
#         #save
        
#         df_with_ids.to_csv("with_ids/"+fn+".csv", index = False)
        
        
        
        

    
    
    
# def to_id(x):
#     global ids4names
#     #x = ast.literal_eval(x)
#     # print(type(x))
#     ids = []
    
#     for n in x:
#         ids.append(ids4names[n])
    
#     return ids
    
# def to_dict_id(x):
#     global ids4names
#     ld = []
    
#     for n in x:
#         d={}
#         d['name'] = n
#         d['id'] = ids4names[n]
#         ld.append(d)
   
#     return ld
    
        
    
    
# create new columns
df_auths4doc['authors'] = df_auths4doc.authors_name.map(lambda x:to_dict_id(x))




# script assign id

ids4names=auth.drop_duplicates(['name']).loc[:,['name', 'id']].set_index('name').to_dict()

ids4names = ids4names['id']

total = pd.read_csv("total_not_clean_dup.csv")


no_ids = []
final_name_id = {}




uniq_names=total.author_name.values.tolist()

uniq_names=list(set(uniq_names))

ava_names = list(ids4names.keys())

def assign_id(n):
    global final_name_id
    global ava_names
    global ids4names
    
    # if n matches a name already added we skip
    test_res = match_name_from_list(list(final_name_id.keys()), n)
    if test_res is None:
        
    
        found = False
        
        for kn in ava_names:
            if (match_name(n, kn) == True) or (match_name(kn, n) == True):
                found = True
                break
            
        if found:
            # select the first id
            final_name_id[n] = ids4names[kn]
            return {'name': n, 'id': final_name_id[n]}
        else:
            return False
    
    else:
        print(n+" matches "+test_res+" in final dict.")
        return {'name':test_res, 'id': final_name_id[test_res]}
        
    

total['author_ref'] = total.author_name.map(lambda x:assign_id(x))

############# create new names

auth_only=total.author_name.drop_duplicates()

new_auth_names = []
cpt = 0
dict_res_2 = {}
def create_new_names(x):
    
    global dict_res_2
    global cpt
    global new_auth_names
    
    print(cpt)
    cpt += 1
    
    res = match_name_from_list(new_auth_names, x)
    if res is None:
        # it's a new name or empty list
        new_auth_names.append(x)
        dict_res_2[x] = x
        return x
    else:
        # it atches, return it
        dict_res_2[x] = res
        # print(res)
        return res
    
auth_only['new_names'] = auth_only.map(lambda x: create_new_names(x))
        

    
# create ids4names

ids4names = {}
current_id = 0
def assign_unique_id4name(x):
    
    global ids4names
    global current_id
    
    ids4names[x] = current_id
    current_id += 1
    
# assign a unique id for each new name
df_new_names.new_names.map(lambda x: assign_unique_id4name(x))




    

# create a list of authors names for each document     
#########################
auths4doc = {}

def get_authors(x):
    
    global auths4doc
    p_id = x.id_paper
    if p_id in auths4doc.keys():
        n = x.new_names
        list_names = auths4doc[p_id]
        # add 1 more element to the list
        # if is not matching a name already added
        
        #if match_name_from_list(list_names, n) == None:
        if n not in list_names:
            auths4doc[p_id].append(n)
        else:
            print("name matching name: ",n," list : ",list_names)
    else:
        #create new entry
        
        auths4doc[p_id] = [x.new_names]
        

total_new_names.apply(lambda x:get_authors(x), axis=1)   


# auths4doc to df

df_auth4doc = pd.DataFrame(auths4doc.items(), columns=['id_paper', 'list_authors_name'])

# merge


total_new_names = total_new_names.merge(df_auth4doc, on='id_paper')
##########################   
 
    
def to_dict_id(x):
    global ids4names
    ld = []
    
    for n in x:
        d={}
        d['name'] = n
        d['id'] = ids4names[n]
        ld.append(d)
   
    return ld


# create new columns
total_new_names['authors'] = total_new_names.list_authors_name.map(lambda x:to_dict_id(x))
    
# transform ids4names to DF

df_ids4names = pd.DataFrame(ids4names.items(), columns=['new_names', 'id'])
  
# create [pubs]

d_pubs = {}

def add_pub(x):
    global d_pubs
    
    n = x.new_names
    p = x.id_paper
    if n in d_pubs.keys():
        d_pubs[n].append(p)
    else:
        d_pubs[n] = [p]
    
total_new_names.apply(lambda x: add_pub(x), axis=1)
    
df_d_pubs = pd.DataFrame(d_pubs.items(), columns=['new_names', 'pubs'])

total_new_names = total_new_names.merge(df_d_pubs, on = 'new_names')

acm_all_authors = total_new_names.drop_duplicates('new_names')
acm_all_authors = acm_all_authors[['author_name', 'author_average_citation_per_article',
       'author_citation_count', 'author_publication_counts',
        'papers_available_for_download',
       'author_subject_areas', 'author_keywords', 'new_names','pubs']]


# 1) drop duplicates titles from final

# clean DZ DB

# split row


splited_df = pd.DataFrame()

def split_row_names(db):
    
    global splited_df


# use defines fcts   
    
    

