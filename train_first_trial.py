# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 21:59:04 2020

@author: Ege
"""

X =  df['Title_averages']
y = df['Binary_Label']

X = pd.DataFrame.from_records(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test_cv, y_train, y_test_cv = train_test_split(X ,y, test_size = 0.2)
X_test, X_cv, y_test, y_cv = train_test_split(X_test_cv,y_test_cv, test_size = 0.5)

from keras.models import Sequential
from keras.layers import Dense#create model
from keras.metrics import accuracy
model = Sequential()
#get number of columns in training data
n_cols = X_train.shape[1]

#add model layers
model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
model.add(Dense(1, activation='sigmoid'))

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
