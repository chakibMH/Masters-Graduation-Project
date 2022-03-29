# -*- coding: utf-8 -*-


import re

def match_name(to_check, name):
    
    #garder q un seul espace
    
    name = re.sub(r"[ ]+", " ", name)
    
    to_check = re.sub(r"[ ]+", " ", to_check)
    
    # miniscule
    
    name = name.lower()
    
    to_check = to_check.lower()
    
    # if exact matching then true
    
    if to_check == name:
        return True
    else:
        
        # split 
        
        name_to_list = name.split(" ")
        
        matching = True
        # order doesn't matter
        for w in name_to_list:
            #check if it has dot on it
            a = re.search(r"[a-z]\.", w)
            if a != None:
                # it has a dot in it
                #
                # if [a-z]. is not in the name to check then we search for regex
                # 
                # else we continue
                if (w not in to_check):
                    
                    # create a regex based on first char of the word with '.'
                    first_char = w[0]
                    w_pattern = first_char+"[a-z]+"
                    b = re.search(w_pattern, to_check)
                    if b == None:
                        
                        matching = False
                        break
                    

            else:
                # no dot check exact text matching
                w_pattern = r"\b"+w+r"\b"
                b = re.search(w_pattern, to_check)
                if b == None:
                    
                    matching = False
                    break
            
            
        return matching
            
            