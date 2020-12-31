#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 00:51:58 2020

@author: ege
"""
#Import Libraries
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
import os
import re
import shutil
import string

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from sklearn.model_selection import train_test_split #for splitting the validation set.

#Import dataset and split to train,validation,test
train = pd.read_json('data/wos2class.train.json')
test  = pd.read_json('data/wos2class.test.json')

train, validation = train_test_split(train, test_size=0.2, random_state=42, stratify=train['Binary_Label'])

train.pop('Abstract')
train_target = train.pop('Binary_Label')

test.pop('Abstract')
test_target = test.pop('Binary_Label')

validation.pop('Abstract')
validation_target = validation.pop('Binary_Label')

batch_size = 32
seed = 42

train = (train.to_numpy()).ravel()
test = (test.to_numpy()).ravel()
validation = (validation.to_numpy()).ravel()

train_data = tf.data.Dataset.from_tensor_slices((train, train_target.values)).batch(batch_size)
test_data = tf.data.Dataset.from_tensor_slices((test, test_target.values)).batch(batch_size)
validation_data = tf.data.Dataset.from_tensor_slices((validation, validation_target.values)).batch(batch_size)

#Sanity Check
for text_batch, label_batch in train_data.take(1):
  for i in range(3):
    print("Title", text_batch.numpy()[i])
    print("Label", label_batch.numpy()[i])

###############################################################################git
#Prepare the dataset for training

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')

#Text Vectorization
max_features = 10000
sequence_length = 36

vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

#Make a text-only dataset (without labels), then call adapt:
train_text = train_data.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

#Create a helper function to see the result of vectorize_layer
def vectorize_text(text,label):
    text = tf.expand_dims(text,-1)
    return vectorize_layer(text), label

#Retrieve a batch (of 32 Titles and labels) from the dataset
text_batch, label_batch = next(iter(train_data))
first_title, first_label = text_batch[0], label_batch[0]
print("Title of the Article:", first_title)
print("Label", first_label.numpy())
print('Vectorized_title', vectorize_text(first_title,first_label))

#Use get_vocabulary() to see the words in the dictionary
print('1287 --->', vectorize_layer.get_vocabulary()[1287])
print('888--->', vectorize_layer.get_vocabulary()[888])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

train_data_ds = train_data.map(vectorize_text)
validation_data_ds = validation_data.map(vectorize_text)
test_data_ds = test_data.map(vectorize_text)


#Performance Configuration
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_data_ds = train_data_ds.cache().prefetch(buffer_size=AUTOTUNE)
validation_data_ds = validation_data_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_data_ds = test_data_ds.cache().prefetch(buffer_size=AUTOTUNE)

#Create the model
embedding_dim = 50

model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1)])

model.summary()

#Loss function and optimizer
model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

#Train the model
epochs = 10

history = model.fit(
    train_data_ds,
    validation_data = validation_data_ds,
    epochs = epochs)


#Evaluate the model
loss, accuracy = model.evaluate(test_data_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

###############################################################################
#Create a plot of accuracy and loss over time
history_dict = history.history
history_dict.keys()

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()
