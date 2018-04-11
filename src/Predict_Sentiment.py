#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 11:19:12 2018

@author: changlongjiang
"""


from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

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

def predict_sentiment(texts,model_location,weights_location,GLOVE_DIR,MAX_LENGTH = 1000,
                      MAX_NB_WORDS = 20000):
    '''
    model_location = '../Model/B_LSTM_model.json'
    weights_location = '../Model/B_LSTM_model.h5'
    GLOVE_DIR = "./Glove/glove.6B.100d.txt"
    MAX_LENGTH = 1000
    MAX_NB_WORDS = 20000
    '''
    #word_embeddings = get_word_embedding(GLOVE_DIR)
    json_file = open(model_location, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_location)
    print("Loaded model from disk")
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    #word_index = tokenizer.word_index
    data = pad_sequences(sequences, maxlen=MAX_LENGTH)
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    X = data[indices]
    # predict test data
    result = loaded_model.predict(X)
    
    return result


#print(test)