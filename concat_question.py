# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 22:34:47 2018

@author: mjh0208
"""
import os

import keras.backend as K
from keras import regularizers
from keras.models import Sequential
from keras.models import Input
from keras.layers import LSTM, Dense, Dropout
from keras.models import load_model
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.layers import concatenate
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
import numpy as np
import preprocess
import pandas as pd
from keras.models import Model


week_steps=8 # week steps to unroll
n_inputs=50 # rows of 27 features
n_classes=4 # customer behavior class


def create_model(learning_rate, n_lstm_layers,
                 n_lstm_nodes, n_dense_layers, n_dense_nodes, activation, dropout):
    
    LSTM_input = Input(shape = (week_steps, n_inputs), name = "LSTM_input")
    Guild_input = Input(shape = (1,) , name = "Guild_input")
    
    x = LSTM_input
    
    ## LSTM layer
    for i in range(n_lstm_layers):
        if(i != n_lstm_layers-1):
            x = LSTM(n_lstm_nodes, input_shape=(week_steps, n_inputs),return_sequences=True,
                 kernel_initializer="he_normal", recurrent_initializer="he_normal")(x)
        else:
            x = LSTM(n_lstm_nodes, input_shape=(week_steps, n_inputs),
                         kernel_initializer="he_normal", recurrent_initializer="he_normal")(x)
        x = Dropout(dropout)(x)
        
    merged = concatenate([x, Guild_input])
    Dense_input = merged
    
    for i in range(n_dense_layers):
        Dense_input = Dense(units=n_dense_nodes, activation=activation, 
                            kernel_initializer = "he_normal")(Dense_input)
        Dense_input = BatchNormalization()(Dense_input)
        Dense_input = Dropout(dropout)(Dense_input)
        
    Output = Dense(n_classes, activation='softmax', kernel_initializer = "he_normal", 
                   name="output")(Dense_input)
    model = Model(inputs=[LSTM_input, Guild_input], outputs=Output)               
    
    optimizer = RMSprop(lr=learning_rate, decay=0.001)  ## lr 0.001 , decay = 0.0001
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=[preprocess.f1])
    return model
    
model = create_model(0.01, 1, 512, 1, 32, 'elu' ,0.4)

    # Use Keras to train the model.
history = model.fit({"LSTM_input": X, "Guild_input":X_guild},
                    {"output" :y},
                       epochs=20,
                       batch_size=128,
                       validation_split = 0.2,
                       callbacks = [metrics])
