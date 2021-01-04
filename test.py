#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 15:19:59 2020

@author: ege
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras as K

np.random.seed(0)
import joblib

np.random.seed(1)

preds = joblib.load('preds.pkl')
y_test = joblib.load('y_test.pkl')
y_test = y_test > 0.5
y_test = np.argmax(y_test, axis=1)
preds = preds > 0.5
preds = np.argmax(preds, axis=1)

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

# summarize history for loss
plt.plot(history.history['auc'])
plt.plot(history.history['val_auc'])
plt.title('Model AUC')
plt.ylabel('auc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['precision'])
plt.plot(history.history['val_precision'])
plt.title('Model Presicion')
plt.ylabel('Precision')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['recall'])
plt.plot(history.history['val_recall'])
plt.title('Model Recall')
plt.ylabel('Recall')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# F1 Score
from sklearn.metrics import f1_score

f1_score(y_test, preds, average='macro')

recall = history.history['recall']
precision = history.history['precision']

val_recall = history.history['val_recall']
val_precision = history.history['val_precision']

epsilon = 1e-07

f1 = np.zeros(20)
for i in range(20):
    f1[i] = 2 * ((precision[i] * recall[i]) / (precision[i] + recall[i] + epsilon))
    print(f1[i])

val_f1 = np.zeros(20)
for i in range(20):
    val_f1[i] = 2 * ((val_precision[i] * val_recall[i]) / (val_precision[i] + val_recall[i] + epsilon))
    print(val_f1[i])

plt.plot(f1)
plt.plot(val_f1)
plt.title('Model F1 Score')
plt.ylabel('F1 Score')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


def classification_report(test, pred, model_name='Bidirectional LSTM Model with Multiple Inputs'):
    from sklearn.metrics import precision_score, recall_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score

    print(model_name, ":\n")
    print("Accuracy Score: ", '{:,.3f}'.format(float(accuracy_score(test, pred)) * 100), "%")
    print("     Precision: ", '{:,.3f}'.format(float(precision_score(test, pred, average='macro')) * 100), "%")
    print("        Recall: ", '{:,.3f}'.format(float(recall_score(test, pred, average='macro')) * 100), "%")
    print("      F1 score: ", '{:,.3f}'.format(float(f1_score(test, pred, average='macro')) * 100), "%")


classification_report(y_test, preds)
