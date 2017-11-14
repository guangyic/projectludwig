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

### _Step 0: Get some MIDIs_

If you do not have a ready source of MIDIs to train, or want to try something else other than what I have provided in my Github, feel free to look at some of the following:
1. [Classical Archives](https://www.classicalarchives.com/midi.html)
2. [Free MIDI (pop sub-section)](https://freemidi.org/genre-pop)
3. [Kunst der Fuge](http://www.kunstderfuge.com/beethoven/chamber.htm)
4. [MIDI World (pop sub-section)](http://www.midiworld.com/search/?q=pop)
5. [Piano MIDI](http://www.piano-midi.de)

As mentioned above, ensure that these files are placed inside a folder named "midi" in your working directory.

### _Step 1: Reading & Converting_

Before anything can be done, we need to first convert the MIDI to a Python-readable format. In order to do this, I use music21.converter and move everything into a dictionary.

```python
# Parse the MIDI data for separate melody and accompaniment parts, then flatten the parts into one line
midi_data = converter.parse(data)
midi_data_flat = midi_data.flat

# Extracts the time signature and the metronome marking from the MIDI
timesig = midi_data_flat.getElementsByClass(meter.TimeSignature)[0]
mmark = midi_data_flat.getElementsByClass(tempo.MetronomeMark)[0]

# Extracts the notes and chords from the MIDI        
notes = midi_data_flat.getElementsByClass(note.Note).notes
chords = midi_data_flat.getElementsByClass(chord.Chord)

# Extracts the temporal offsets of the notes and chords from the MIDI                
notes_offsets = [a.offset for a in notes]
chords_offsets = [b.offset for b in chords]
 
# Stores in the dictionary      
dictionary[str(training_set[n])] = {"notes":notes, "notes_offsets":notes_offsets, "chords":chords, "chords_offsets":chords_offsets, "timesig":timesig, "metronome":mmark}}
```

#### Chord and Note Variables
When looking in the dictionary, the notes will appear in the following format:

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

After the music has been turned into Python-readable data, I convert the data into sets and dictionaries in preparation for feeding the music into a neural network.

```python
# Excerpt of converting the list of music variables 
grouped_string = "".join(theory_list_fin)
corpus = grouped_string.split(' ')
values = set(corpus)
val_indices = dict((v, i) for i, v in enumerate(values))
indices_val = dict((i, v) for i, v in enumerate(values))
```

### _Step 2: Training the Model_

#### Preparing for the Model

Now that the music is in a Python-readable format, it needs to turn into a format that models understand. Generally, models will accept matrices comprised of either booleans or scalar values. As there is a mixture of both categorical and continuous variables, I opted to use the boolean method.

After getting the dimensions of the data that needs to be fed into the model, I formatted the X and y variables into the correct shape and then populated it with the respective boolean values corresponding to the type of note/chord as well as the duration and interval distances. 

```python
X = np.zeros((len(sentences), bars, N_values), dtype=np.bool)
y = np.zeros((len(sentences), N_values), dtype=np.bool)
    
for i, sentence in enumerate(sentences):
    for t, val in enumerate(sentence):
        X[i, t, val_indices[val]] = 1
        y[i, val_indices[next_values[i]]] = 1
```

#### Model
The model uses Keras with a TensorFlow backend. I found that a three-layer LSTM model works best in terms of rate of convergence. This is modified text from the Keras documentation for text generation, as what we are accomplishing here is very similar.

I use a default of 700 epochs for training - depending on how much music is fed in, this number may need to increase or can be decreased.

```python
# Create the LSTM model and create the layers
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(bars, N_values)))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, return_sequences=False))

# Add a dropout layer to provide some regularization
model.add(Dropout(0.2))

# Add a fully-connected dense layer with a softmax activation
# Use softmax because we want the probailities to add up to 1
model.add(Dense(N_values))
model.add(Activation('softmax'))

# Compile the model
# RMSprop converges faster than the other optimizers (adadelta is next fastest)
# Categorical_crossentropy loss function allows faster convergence without the slowdown in learning inherent in other
# loss functions
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# Fit the model, then save it    
model.fit(X, y, batch_size=128, nb_epoch=epochs)
model.save('model_music_gen.h5')
```
__Disclaimer:__ Again, depending on the length of the piece(s) and their complexity, training to convergence can take anywhere from 4 seconds per epoch up to close to 60 seconds per epoch. If you have access to a cloud GPU or your own CUDA-enabled GPU, this will speed up training significantly. 

### _Step 3: Prediction & Production_

---
## __Results__
[![SoundCloud](https://github.com/guangyic/projectludwig/blob/master/siteelements/soundcloud.png?raw=true)](https://soundcloud.com/guang-yi-chua/sets/project-ludwig-output)

## Areas for Improvement
Currently, this model and code makes the assumption that the time and key signature of the piece is constant. With pieces that involve temporal or key modulation, this could lead to odd syncopation in the production of the original music, or to sequences of notes that do not necessarily make sense from a music theory standpoint. 

I extract the time signature and metronome markings, but at the moment, do not use it for anything. Future versions will find a way to incorporate these into the generations. 

---
## Conclusion
