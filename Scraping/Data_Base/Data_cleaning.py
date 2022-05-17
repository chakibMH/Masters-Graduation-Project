import pandas as pd
import string
import re
import os
from num2words import num2words
from math import log
import sys
# from wiktionaryparser import WiktionaryParser

#from nltk.corpus import words




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




# Build a cost dictionary, assuming Zipf's law and cost = -math.log(probability).
words = open("words-by-frequency.txt").read().split()

#list of 300k + english words
check_dict = open("english_words_list.txt").read().split()

# add custom words to the dictionary
#check_dict += ['eigen']


wordcost = dict((k, log((i+1)*log(len(words)))) for i,k in enumerate(words))
maxword = max(len(x) for x in words)

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
    
    for w in sen2words:
        
        
        
        
        if w in check_dict:
            
            # this word is correct ( belongs to the english dictionary) 
            
            res_list.append(w)
            
        else:
            # correcting the word
            print("before:",w )
            w_after = infer_spaces(w)
            print("after:",w_after)
            res_list.append(w_after)
        
    return " ".join(res_list)
    
cpt = 0
    
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
            
            s = s.replace('-', ' ')
                
            # remove ponctuation
            
            punc = string.punctuation + '–'
            
            s = s.translate(str.maketrans('', '', punc))
            
            s = remove_extra_spaces(s)
            
            # correct words
            
            s = correct_words(s)
            
            s = remove_extra_spaces(s)
                
            cleaned_abstract.append(s)
      
            
    
    return cleaned_abstract
            
            
            
            
    
    
def clean_DB(df):
    
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
    
    return df
    
    
    
    
    

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




        
    
    