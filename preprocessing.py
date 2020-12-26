# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 02:24:39 2020

@author: Ege
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import string #for removing the punctuations

#Import the json data
df = pd.read_json('data\wos2class.json')
#Check for Nan Values
df.isnull().values.any()
#Encode the label as 0 and 1 ( 0  for Chemistry, 1 for Material Science)
df["Binary_Label"] = (df["Label"].astype('category')).cat.codes


#Convert title and abstract as list of words
df['Title_List']= [(sentence.lower()).split()  for sentence in df['Title']]
df['Abstract_List']= [(sentence.lower()).split()  for sentence in df['Abstract']]

#Find the maximum length of the Title and Abstract
m = len(df)
max_Title = max([len(word_list) for word_list in df['Title_List']]) #36
max_Abstract = max([len(word_list) for word_list in df['Abstract_List']]) #630

#Look how distribution changes for length_size of Titles and Abstracts
title_list = [(sentence.lower()).split()  for sentence in df['Title']]
title_lengths = [len(element) for element in title_list]
plt.hist(title_lengths, bins='auto')
#There are really less amount of words after 30 we can just give 30 instead of 36

abstract_list = [(sentence.lower()).split()  for sentence in df['Abstract']]
abstract_lengths = [len(element) for element in abstract_list]
plt.hist(abstract_lengths, bins='auto')
#Turns out 630 for max length was outlier for the abstract data. We can give 400

#Prepare the title and abstract arrays as an input for embedding.

def sentences_to_indices(column,word_to_index,max_len):
    
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    
    Arguments:
    column -- The column of sentences which is going to be converted
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    unknown_word_counter = 0
    unique_unknown_words = set()
    unique_words = set()
    
    table_ = str.maketrans('', '', string.punctuation) #for removing any punctuations
    #Number of samples
    m = len(column)                                  
    #initialize a the array for Title_indices
    X_indices = np.zeros((m,max_len))
    
    for i in range(m):
        
        sentence_without_punc = column[i].translate(table_)                       
        sentence_words = (sentence_without_punc.lower()).split()
        
        #print(sentence_words)
        j = 0
        
        for w in sentence_words:
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            #print(w)
            
            try:
                X_indices[i, j] = word_to_index[w]
            except:
                print('unknown word: ',w)
                X_indices[i, j] = word_to_index['unk']
                unknown_word_counter += 1
                unique_unknown_words.add(w)
                
            finally:
                unique_words.add(w)
                j = j+1           
    
    print('total unique words', len(unique_words))
    print('total unique unknown words', len(unique_unknown_words))
    print('Counter of unknown words: ', unknown_word_counter)
    X_indices = X_indices.tolist()
    return X_indices

    
df['Title_indices'] = sentences_to_indices(df['Title'],word_to_index,36)
    


