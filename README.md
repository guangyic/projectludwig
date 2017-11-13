# Project Ludwig: Music Generation through Deep Learning
---

![Beethoven](https://github.com/guangyic/projectludwig/blob/master/siteelements/beethoven.png?raw=true=100x150)

---
# Preparation
1. Create a folder named "midi" in your working directory
2. These scripts were written in a Linux environment in macOS High Sierra. I used Python 2.7.14. If you are using Windows, ensure that the commands that reference directories use ' / ' instead of ' \ ' (I have some of these filled in here and there when I used my Windows PC to train my model)
3. Ensure that you have all of the following scripts: 
	* main.py
	* extract.py
	* musicology.py
	* training.py
	* check.py
4. If you wish to use your own MIDI files, place them in the "midi" folder. Ideally, have 3-5 pieces inside to provide some diversity for your model to train on.

## Required Standard Python Packages
- collections
- copy
- datetime
- itertools
- os
- pdb
- pickle
- random
- sys

## Required Non-Standard Python Packages 
- [Keras](https://keras.io)
- [MuseScore](https://musescore.org)
- [music21](http://web.mit.edu/music21/)
- [NumPy](http://www.numpy.org)
- [PyGame](https://www.pygame.org/news)
- [TensorFlow](https://www.tensorflow.org)

---
# Implementation

## Step 1: Reading & Converting

## Step 2: Training the Model

### Model
The model uses Keras with a TensorFlow backend. I found that a three-layer LSTM model works best in terms of rate of convergence. This is modified text from the Keras documentation for text generation, as what we are accomplishing here is very similar.

```python
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(bars, N_values)))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(N_values))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    
model.fit(X, y, batch_size=128, nb_epoch=epochs)
model.save('model_music_gen.h5')
```

## Step 3: Prediction & Production

---
# Results
