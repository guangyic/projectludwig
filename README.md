# Project Ludwig: Music Generation through Deep Learning
---
## __Introduction__

![Beethoven](https://github.com/guangyic/projectludwig/blob/master/siteelements/beethoven.png?raw=true)

![Neuron](https://github.com/guangyic/projectludwig/blob/master/siteelements/neuron1.png?raw=true)

![Neuron Network](https://github.com/guangyic/projectludwig/blob/master/siteelements/neuron2.png?raw=true)
---
## __Preparation__
1. Create a folder named "midi" in your working directory
2. These scripts were written in a Linux environment in macOS High Sierra. I used Python 2.7.14. If you are using Windows, ensure that the commands that reference directories use ' / ' instead of ' \ ' (I have some of these filled in here and there when I used my Windows PC to train my model)
3. Ensure that you have all of the following scripts: 
	* main.py
	* extract.py
	* musicology.py
	* training.py
	* check.py
4. If you wish to use your own MIDI files, place them in the "midi" folder. Ideally, have 3-5 pieces inside to provide some diversity for your model to train on.

### _Required Standard Python Packages_
- collections
- copy
- datetime
- itertools
- os
- pdb
- pickle
- random
- sys

### _Required Non-Standard Python Packages_
- [Keras](https://keras.io)
- [MuseScore](https://musescore.org)
- [music21](http://web.mit.edu/music21/)
- [NumPy](http://www.numpy.org)
- [PyGame](https://www.pygame.org/news)
- [TensorFlow](https://www.tensorflow.org)

---
## __Implementation__

### _Step 1: Reading & Converting_

```python
"C,0.250,<P4,m-2>"
```

The first variable will indicate:
- C: A chord
- S: A single pitch in the scale
- R: A rest
- X: A single pitch that does not belong to the scale

The second variable indicates the duration of the note in relation to the bar. In the above example, a duration of 0.250 indicates that the note/chord duration is 1/4 of the overall bar (for you musicians out there, assuming the time signature is 4/4, this would indicate that the note is a quarter/crotchet note). 

Finally, the last set of variables indicates the maximum distance (higher, then lower) the note/chord is relative to the previous note/chord. 

### _Step 2: Training the Model_

#### Model
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

### _Step 3: Prediction & Production_

---
## __Results__
[![SoundCloud](https://github.com/guangyic/projectludwig/blob/master/siteelements/soundcloud.png?raw=true)](https://soundcloud.com/guang-yi-chua/sets/project-ludwig-output)

## Areas for Improvement
Currently, this model and code makes the assumption that the time and key signature of the piece is constant. With pieces that involve temporal or key modulation, this could lead to odd syncopation in the production of the original music, or to sequences of notes that do not necessarily make sense from a music theory standpoint. 