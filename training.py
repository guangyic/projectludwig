'''
Author:     Chua Guang Yi
Project:    Ludwig S.T. Meinung (Ludwig for short)

'''

# Import necessary packages
from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Merge, LSTM 
from keras.optimizers import RMSprop
import numpy as np

#----------------------------CALLED BY MAIN FUNCTION-------------------------#

''' 
Build a 3-layer LSTM from a training corpus
Called in generate in main.py 
'''
def build_model(corpus, val_indices, bars, epochs=700):
    
    # Number of different values or words in corpus
    N_values = len(set(corpus))

    step = 3
    sentences = []
    next_values = []
    
    for i in range(0, len(corpus) - bars, step):
        sentences.append(corpus[i: i + bars])
        next_values.append(corpus[i + bars])
    print('nb sequences:', len(sentences))

    # Transform data into binary matrices
    X = np.zeros((len(sentences), bars, N_values), dtype=np.bool)
    y = np.zeros((len(sentences), N_values), dtype=np.bool)
    
    for i, sentence in enumerate(sentences):
        for t, val in enumerate(sentence):
            X[i, t, val_indices[val]] = 1
            y[i, val_indices[next_values[i]]] = 1
    
    # Create the model with 3 layers    
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(bars, N_values)))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128, return_sequences=False))

    # Add dropout for regularization purposes
    model.add(Dropout(0.2))

    # Add dense layer and softmax for output
    model.add(Dense(N_values))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        
    model.fit(X, y, batch_size=128, nb_epoch=epochs)

    # Change name here if want to name model something else. Must have .h5 at end
    model.save('model_music_gen.h5')
    
    return model