#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 06:09:58 2020

@author: ege
"""
import pandas as pd

#Import dataset and split to train,validation,test
train = pd.read_json('data/wos2class.train.json')
test  = pd.read_json('data/wos2class.test.json')

#train, validation = train_test_split(train, test_size=0.2, random_state=42, stratify=train['Binary_Label'])

train.pop('Abstract')
train_target = train.pop('Binary_Label')

test.pop('Abstract')
test_target = test.pop('Binary_Label')

#validation.pop('Abstract')
#validation_target = validation.pop('Binary_Label')

batch_size = 32
seed = 42

train = (train.to_numpy()).ravel()
test = (test.to_numpy()).ravel()
#validation = (validation.to_numpy()).ravel()

train_data = tf.data.Dataset.from_tensor_slices((train, train_target.values)).batch(batch_size)
test_data = tf.data.Dataset.from_tensor_slices((test, test_target.values)).batch(batch_size)
#validation_data = tf.data.Dataset.from_tensor_slices((validation, validation_target.values)).batch(batch_size)


#Define your model
def build_classifier_model():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=False, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.5)(net)
  net = tf.keras.layers.Dense(128)(net)
  net = tf.keras.layers.Dropout(0.5)(net)
  
  net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
  
  model = tf.keras.Model(text_input, net)
  model.summary()
  return model

classifier_model = build_classifier_model()
bert_raw_result = classifier_model(tf.constant(text_test))
print(tf.sigmoid(bert_raw_result))

tf.keras.utils.plot_model(classifier_model)

loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()

epochs = 50
steps_per_epoch = tf.data.experimental.cardinality(train_data).numpy()
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)

init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

classifier_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)

print(f'Training model with {tfhub_handle_encoder}')
history = classifier_model.fit(x=train_data,
                               validation_data=test_data,
                               epochs=epochs)
model.layers