#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 19:22:23 2018

@author: changlongjiang
"""

import numpy as np
import pandas as pd
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input
from keras.layers import Conv1D, GlobalMaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Sequential, Model
from keras.models import model_from_json

def clean_up(string):
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    string = re.sub(r"\n", "", string)
    string = re.sub(r'[a-z]*[:.]+\S+', '',string)
    return string.strip().lower()
def get_word_embedding(DIR):    
    d = {}
    f = open(DIR)
    for line in f:
        v = line.split()
        word = v[0]
        vec = np.asarray(v[1:], dtype='float32')
        d[word] = vec
    f.close()
    return d

MAX_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.05
Yelp_DATA = '/Volumes/Dragon/dataset/review.csv'
df = pd.read_csv(Yelp_DATA, sep=',',index_col=0)
GLOVE_DIR = "./Glove/glove.6B.100d.txt"
texts=[]
labels=[]
for i in range(len(df.values)):
    try:
        texts.append(clean_up(df['texts'][i]))
    except:
        texts.append('None')
    labels.append(df['stars'][i]-1)
word_embeddings = get_word_embedding(GLOVE_DIR)
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', len(texts))
print('Shape of label tensor:', len(labels))
print('Total %s word vectors.' % len(word_embeddings))



tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=MAX_LENGTH)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]


nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]



embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = word_embeddings.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
print ('Length of embedding_matrix:', embedding_matrix.shape[0])
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            mask_zero=False,
                            input_length=MAX_LENGTH,
                            trainable=False)

print('Traing and validation set number of positive and negative reviews')
print(y_train.sum(axis=0))
print(y_val.sum(axis=0))



sequence_input = Input(shape=(MAX_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
dense_1 = Dense(100,activation='tanh')(embedded_sequences)
max_pooling = GlobalMaxPooling1D()(dense_1)
dense_2 = Dense(5, activation='softmax')(max_pooling)


model = Model(sequence_input, dense_2)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.summary()


# reference: https://machinelearningmastery.com/save-load-keras-deep-learning-models/
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=10, batch_size=50)

# serialize model to JSON
MLP_model_json = model.to_json()
with open("./Model/Yelp_MLP_model.json", "w") as json_file:
    json_file.write(MLP_model_json)
    
# serialize weights to HDF5
model.save_weights("./Model/Yelp_MLP_model.h5")
print("Saved model to disk")


# reference: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
import matplotlib.pyplot as plt
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('mlp model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./Results/Yelp_MLP_accuracy.jpg')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('mlp model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./Results/Yelp_MLP_loss.jpg')
plt.show()

