#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 15:19:59 2020

@author: ege
"""

import numpy as np
import pandas as pd
np.random.seed(0)
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
import string #for removing the punctuations
np.random.seed(1)

def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding= "utf8") as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map


# You can download the data from here: http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

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
    
    #Normally its string punctuation
    punctuations = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'
    table_ = str.maketrans('', '', punctuations) #for removing any punctuations
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


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    
    vocab_len = len(word_to_index) + 1                  
    emb_dim = word_to_vec_map["cucumber"].shape[0]      # define dimensionality of your GloVe word vectors (= 50)
    

    emb_matrix = np.zeros((vocab_len,emb_dim))
    for word, idx in word_to_index.items(): #key and value
        
        emb_matrix[idx, :] = word_to_vec_map[word]


    # Make it non-trainable.
    embedding_layer = Embedding(input_dim = vocab_len, output_dim = emb_dim, trainable = False)    
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer

#Check if its working
embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
print("weights[0][1][3] =", embedding_layer.get_weights()[0][1][3])
"""
weights[0][1][3] = -0.3403
"""

#maxLen for Title is 36
maxLen = 36

def custom_model(input_shape, word_to_vec_map, word_to_index):
    """
    Function creating the model's graph.
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """
    
    sentence_indices = Input(shape = input_shape, dtype='int32')   
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    embeddings = embedding_layer(sentence_indices)   
    
    X = LSTM(units=128, return_sequences = True)(embeddings)

    X = Dropout(rate = 0.8)(X)

    X = LSTM(units=128, return_sequences = False)(X)

    X = Dropout(rate = 0.8)(X)

    X = Dense(units=2)(X)

    X = Activation('softmax')(X)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs= sentence_indices, outputs= X)
    
    ### END CODE HERE ###
    
    return model



X_train =  pd.read_json('data/wos2class.train.json')
X_train_Title = list(X_train['Title'])
X_train_indices = sentences_to_indices(X_train_Title, word_to_index, maxLen)

y_train = X_train['Binary_Label']
y_train = pd.get_dummies(y_train)
y_train = y_train.values.tolist()

X_test = pd.read_json('data/wos2class.test.json')
X_test_Title = list(X_test['Title'])

X_test_indices = sentences_to_indices(X_test_Title, word_to_index, maxLen)

y_test = X_test['Binary_Label']
y_test = pd.get_dummies(y_test)
y_test = y_test.values.tolist()

model = custom_model((maxLen,), word_to_vec_map, word_to_index)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train_indices, y_train, epochs = 8, batch_size = 32, shuffle=True, validation_data=(X_test_indices,y_test))

model.save('model')
