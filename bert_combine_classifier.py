#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 08:53:00 2020

@author: ege
"""

#Import dataset and split to train,validation,test
train = pd.read_json('data/wos2class.train.json')
test  = pd.read_json('data/wos2class.test.json')

#train, validation = train_test_split(train, test_size=0.2, random_state=42, stratify=train['Binary_Label'])

train['Combine'] = train['Title']  + ' ' + train['Abstract']
train.pop('Abstract')
train.pop('Title')
train_target = train.pop('Binary_Label')

test['Combine'] = test['Title']  + ' ' + test['Abstract']
test.pop('Abstract')
test.pop('Title')
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
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(256)(net)
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

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=5, mode='min')
mc = ModelCheckpoint('name_of_the_model_file.hdf5', monitor='val_loss')


print(f'Training model with {tfhub_handle_encoder}')

history = classifier_model.fit(x=train_data,
                               validation_data=test_data,
                               epochs=epochs, callbacks= [rlrop, mc])

history_dict = history.history
print(history_dict.keys())

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)
fig = plt.figure(figsize=(10, 6))
fig.tight_layout()

plt.subplot(2, 1, 1)
# "bo" is for "blue dot"
plt.plot(epochs, loss, 'r', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
# plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')