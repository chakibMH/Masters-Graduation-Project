# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 21:23:14 2022

@author: HP
"""
import string
import re
from num2words import num2words


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


def remove_url(txt):
    
    regex_url = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"

    return re.sub(regex_url, "", txt)

def remove_email(txt):
    
    regex_email = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    return re.sub(regex_email, "", txt)

def remove_numbers(txt):
    
    regex_num = r"\([0-9]+\)"
    
    return re.sub(regex_num, "", txt)

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


def clean_def(sent):
    
    sent = sent.lower()

    # treating numbers
    
    # reomve ,
    
    sent = sent.replace(",","")
    
    # replace numbers with words
    
    sent = re.sub(r"(\d+\.\d+)", lambda x:num2words(x.group(0)), sent)
    
    sent = re.sub(r"(\d+)", lambda x:num2words(x.group(0)), sent)
    
    
    
    # remove email
    
    sent = remove_email(sent)
    
    # remove url
    
    sent = remove_url(sent)
    
    sent = remove_punc(sent)
    #remove extra spaces
    sent = remove_extra_spaces(sent)
    
    return sent
    
