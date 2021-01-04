# -*- coding: utf-8 -*-

"""
Created on Fri Jan  1 20:49:56 2021

@author: ege
"""
import numpy as np
import pandas as pd

np.random.seed(0)
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, SpatialDropout1D, Bidirectional, concatenate
from keras import regularizers
from keras.layers.embeddings import Embedding
# from keras.preprocessing import sequence
# from keras.initializers import glorot_uniform
# import string  # for removing the punctuations

# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.models import Model
# from keras.layers import Input

import nltk
import re
# from string import punctuation
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# import nltk.stem as stemmer
from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import sent_tokenize
from nltk import WordPunctTokenizer

import wikipedia
from nltk.tokenize import sent_tokenize

# from keras.preprocessing.text import Tokenizer
from gensim.models.fasttext import FastText

# NLP
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from keras.utils import np_utils

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

en_stop = set(STOP_WORDS)

# Research papers will often frequently use words that don't actually contribute to the meaning
# and are not considered everyday stopwords.

custom_stop_words = [
    'doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure',
    'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.',
    'al.', 'elsevier', 'pmc', 'czi', 'www'
]
for word in custom_stop_words:
    en_stop.add(word)

# Import dataset and split to train,validation,test
train = pd.read_json('data/wos2class.train.json')
test = pd.read_json('data/wos2class.test.json')

train_title = list(train['Title'])
train_abstract = list(train['Abstract'])

# Combination of both title and abstract
train_combine = np.empty_like(train_title)
for i in range(len(train_title)):
    train_combine[i] = train_title[i] + ' <sep> ' + train_abstract[i]


# Dataset has an article without an abstract lets find it and fix it.
def find_empty_abstract(train_combine):
    counter = 0
    number = None
    temp = None
    for i in range(len(train_combine)):
        words = train_combine[i].split()
        for j in range(len(words)):
            if words[j] == '<sep>':
                try:
                    temp = words[j + 1]
                    pass
                except:
                    counter += 1
                    number = i
                    break;
    print('The row which has missing abstract: ', number)
    if number is not None:
        print('Add train_title as abstract again.')
        print('Such as "train_abstract[number] = train_title[number]" ')
    return number


number = find_empty_abstract(train_combine)
train_combine[number] = train_combine[number] + train_title[number]
train_abstract[number] = train_title[number]

y_train = train['Binary_Label']
# train_combine = train_title.copy()
# train_combine.extend(train_abstract)
# y_train_combine = np.concatenate([y_train,y_train])

test_title = list(test['Title'])
test_abstract = list(test['Abstract'])
# Combination of both title and abstract for test dataset
test_combine = np.empty_like(test_title)
for i in range(len(test_title)):
    test_combine[i] = test_title[i] + ' <sep> ' + test_abstract[i]

y_test = test['Binary_Label']

# test_combine = test_title.copy()
# test_combine.extend(test_abstract)
# y_test_combine = np.concatenate([y_test,y_test])


y_test = np_utils.to_categorical(y_test)
y_train = np_utils.to_categorical(y_train)


def preprocess_text(document):
    stemmer = WordNetLemmatizer()
    # Remove  all the special characters
    document = re.sub(r'\W', ' ', str(document))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to lower_case
    document = document.lower()

    # Lemmatization
    tokens = document.split()
    tokens = [stemmer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if word not in en_stop]
    tokens = [word for word in tokens if len(word) > 3]

    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


# Decide the maximum length of your data
title_len = 32
abstract_len = 96
# Prepare FastText Training Data
final_corpus = [preprocess_text(sentence) for sentence in train_combine if sentence.strip() != '']

chemistry = wikipedia.page('Chemistry').content
material_science = wikipedia.page('Material Science').content
chemical_element = wikipedia.page('Chemical element').content

chemistry = sent_tokenize(chemistry)
material_science = sent_tokenize(material_science)
chemical_element = sent_tokenize(chemical_element)
chemistry.extend(material_science)
chemistry.extend(chemical_element)

final_corpus.extend(chemistry)

word_punctuation_tokenizer = WordPunctTokenizer()
word_tokenized_corpus = [word_punctuation_tokenizer.tokenize(sent) for sent in final_corpus]

tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=True)
tokenizer.fit_on_texts(word_tokenized_corpus)

# Train a FastText Model:
# Hyperparameters of the model:
embedding_size = 100
window_size = 40
min_word = 5
down_sampling = 1e-2

# Train the FastText Model:
ft_model = FastText(word_tokenized_corpus,
                    size=embedding_size,
                    window=window_size,
                    min_count=min_word,
                    sample=down_sampling,
                    sg=1,
                    iter=100)

# Extract fasttext learned embedding and put them in a numpy array
embedding_matrix_ft = np.random.random((len(tokenizer.word_index) + 1, ft_model.vector_size))
pas = 0
for word, i in tokenizer.word_index.items():

    try:
        embedding_matrix_ft[i] = ft_model.wv[word]
    except:
        pas += 1
print(embedding_matrix_ft.shape)

ft_model.save("fastText_model/fastText_combine.bin")

# Prepare Training data for the LSTM model
train_title_corpus = [preprocess_text(sentence) for sentence in train_title if sentence.strip() != '']
train_abstract_corpus = [preprocess_text(sentence) for sentence in train_abstract if sentence.strip() != '']

# word_punctuation_tokenizer = WordPunctTokenizer()
word_tokenized_train_title_corpus = [word_punctuation_tokenizer.tokenize(sent) for sent in train_title_corpus]
word_tokenized_train_abstract_corpus = [word_punctuation_tokenizer.tokenize(sent) for sent in train_abstract_corpus]

# tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=True, num_words=15000)
# tokenizer.fit_on_texts(word_tokenized_corpus)

sequence_title = tokenizer.texts_to_sequences(word_tokenized_train_title_corpus)
sequence_title = tf.keras.preprocessing.sequence.pad_sequences(sequence_title, maxlen=title_len)

sequence_abstract = tokenizer.texts_to_sequences(word_tokenized_train_abstract_corpus)
sequence_abstract = tf.keras.preprocessing.sequence.pad_sequences(sequence_abstract, maxlen=abstract_len)

# Prepare Test data for the LSTM model
unseen_titles = [preprocess_text(sentence) for sentence in test_title if sentence.strip() != '']
unseen_abstract = [preprocess_text(sentence) for sentence in test_abstract if sentence.strip() != '']

word_tokenized_test_title_corpus = [word_punctuation_tokenizer.tokenize(sent) for sent in unseen_titles]
word_tokenized_test_abstract_corpus = [word_punctuation_tokenizer.tokenize(sent) for sent in unseen_abstract]

sequence_unseen_title = tokenizer.texts_to_sequences(word_tokenized_test_title_corpus)
sequence_unseen_title = tf.keras.preprocessing.sequence.pad_sequences(sequence_unseen_title, maxlen=title_len)

sequence_unseen_abstract = tokenizer.texts_to_sequences(word_tokenized_test_abstract_corpus)
sequence_unseen_abstract = tf.keras.preprocessing.sequence.pad_sequences(sequence_unseen_abstract, maxlen=abstract_len)

# LSTM Model Training

# define a keras model and load the pretrained fasttext weights matrix

title_inp = Input(shape=(title_len,))
abstract_inp = Input(shape=(abstract_len,))

title_emb = Embedding(len(tokenizer.word_index) + 1, ft_model.vector_size,
                      weights=[embedding_matrix_ft], trainable=False)(title_inp)

abstract_emb = Embedding(len(tokenizer.word_index) + 1, ft_model.vector_size,
                         weights=[embedding_matrix_ft], trainable=False)(abstract_inp)

x1 = Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=regularizers.l2(0.01)))(title_emb)
# x1 = Dropout(0.5)(x1)
x2 = Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=regularizers.l2(0.01)))(
    abstract_emb)
# x2 = Dropout(0.5)(x2)
concat = concatenate([x1, x2])
classifier = Dense(64, activation='relu')(concat)
output = Dense(2, activation='sigmoid')(classifier)

model = Model(inputs=[title_inp, abstract_inp], outputs=[output])
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', 'AUC', 'Precision', 'Recall'])
# model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])
history = model.fit([sequence_title, sequence_abstract], y_train, epochs=20,
                    validation_data=([sequence_unseen_title, sequence_unseen_abstract], y_test))

# Train the embeddings
model.layers[2].trainable = True
model.layers[3].trainable = True

model.layers[4].trainable = False
model.layers[5].trainable = False
model.layers[7].trainable = False

history = model.fit([sequence_title, sequence_abstract], y_train, epochs=5,
                    validation_data=([sequence_unseen_title, sequence_unseen_abstract], y_test))

# Train a bit more
model.layers[2].trainable = False
model.layers[3].trainable = False

model.layers[4].trainable = True
model.layers[5].trainable = True
model.layers[7].trainable = True
model.summary()

history = model.fit([sequence_title, sequence_abstract], y_train, epochs=5,
                    validation_data=([sequence_unseen_title, sequence_unseen_abstract], y_test))
