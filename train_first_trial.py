# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 21:59:04 2020

@author: Ege
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import joblib

df1 = joblib.load('df_MVP_840.pkl')

X =  df1['Title_averages']
X = pd.DataFrame.from_dict(dict(zip(X.index, X.values))).T
y = df1['Binary_Label']


# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(df1, test_size=0.2, random_state=42, stratify=df1['Binary_Label'])

#Export train and test data:
X_train.to_json(r'data/wos2class.train.json')
X_test.to_json(r'data/wos2class.test.json')

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import Model
from keras.layers import Input
#from keras.metrics import accuracy
model = Sequential()
#get number of columns in training data
n_cols = X_train.shape[1]

#add model layers
model.add(Dense(300, activation='relu', input_shape=(n_cols,)))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())
#compile model using mse as a measure of model performance
model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])

history = model.fit(X_train,y_train, epochs=100, validation_data=(X_cv,y_cv))

#Predict CV Set
model.evaluate(X_cv,y_cv)
history.history

# list all data in history
print(history.history.keys())


# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
