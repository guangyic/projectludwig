# Project Ludwig: Music Generation through Deep Learning
---
## __Introduction__
Recently, machine learning and artificial intelligence has gotten a great deal of attention in the media. Other words that are usually heard in the same breath are deep learning, neural networks, and big data. With this focus on this subset of technology, companies and independent people have designed and implemented them to various degrees. Here are some of the most famous examples that may come to mind:
- [Alexa](https://www.technologyreview.com/s/608571/alexa-understand-me/)
- [AlphaGo](https://deepmind.com/research/alphago/)
- [Deep Blue Chess AI](https://www.wired.com/2017/05/what-deep-blue-tells-us-about-ai-in-2017/)

With my background in neuroscience and classical music, I latched on to the subset of people that are interested in  machine-generated music. The most important questions I wonder about are:
1. What purpose does music serve humanity, if any? 
2. Is music inherently human?
	* Can a machine learn to create music?
	* Can the music a machine composes evoke the same emotion that human music can?
	* Can a machine learn to compose like the musical giants of the past and present? 
		- Examples: Adele, Bach, Beethoven, Chopin, Coldplay, Gershwin, Taylor Swift

### Ludwig S.T. Meinung
With these questions in mind, I (as a fledgling data scientist) set out to work on my first machine-generated music project. Instead of simply naming my project using the "Deep _______" convention, I tried giving my model a name. I got this idea from an article (which I, of course, cannot find any more) about a scientist that gave his model a full name and tried using it in daily conversation. For demonstration purposes, I'm going to call his model "John Smith".
> Person: Hey this music is pretty good. Who's it by?
> Researcher: John Smith.
> Person: Oh yeah? I've never heard of that guy!

In a way, John Smith has passed the [Turing Test](https://en.wikipedia.org/wiki/Turing_test) of music, where the person listening could not tell that the music he likes has been machine generated. With this idea in mind, I wanted to achieve something similar with my own model. For inspiration, I turned to the given name of one of my heroes, [Ludwig van Beethoven](https://www.biography.com/people/ludwig-van-beethoven-9204862). (Photo credit goes to [ClassicFM](http://www.classicfm.com/composers/beethoven/guides/reasons-love-beethoven/))

![Beethoven](https://github.com/guangyic/projectludwig/blob/master/siteelements/beethoven.png?raw=true)

The last word, "Meinung", is the German word for mind or soul, which resonated with me due to the philosophical(?) nature of the question "is music inherently human?" from above. 

Finally, I added "S" and "T" as a further joke, because if you take the initials, it spells "L.S.T.M", which stands for long-short term memory, which is the deep learning model I use for this project. I will go into what LSTM is below.


### Machine Learning and Deep Learning
Neural networks were inspired by the brain. Just as a primer, a neuron is the basic building block of the brain. These neurons carry information everywhere through networks. The sum of all these neuronal networks is what we call the brain. Here is an example of how two neurons work together to pass information, and a stained image of what a neuron network looks like (images from [Khan Academy](https://www.khanacademy.org/science/biology/human-biology/neuron-nervous-system/a/overview-of-neuron-structure-and-function) and [Every Stock Photo](http://www.everystockphoto.com/photo.php?imageId=309897), respectively)
![Neuron](https://github.com/guangyic/projectludwig/blob/master/siteelements/neuron1.png?raw=true)
![Neuron Network](https://github.com/guangyic/projectludwig/blob/master/siteelements/neuron2.png?raw=true)

As a point of reference, the largest neural network implementations that are currently being used number around  thousands or tens of thousands of neurons; the human brain contains 86,000,000,000 (86 billion) neurons. 

#### Classical Neural Networks and Long-short Term Memory (LSTM)
The diagram below shows a classical neural network ([Image Source](https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/300px-Colored_neural_network.svg.png)). The basic neural network has one input layer, one hidden layer, and one output layer. The hidden layer is where all the training happens.
![Classical Neural Network](https://github.com/guangyic/projectludwig/blob/master/siteelements/classicalneural.png?raw=true)

Any classical neural network that has more than one hidden layer, by definition, is considered deep learning. The issue with basic neural networks is that it has a memory problem. As an example, someone is trying to produce some text based on a book. In English, we learned that nouns, pronouns, and verbs belong to specific parts of a sentence. Here is a simple sentence:
> Dogs chase cats
> (Noun-verb-noun)

Using our own memory, we know that the verb comes after the noun, and then the next noun comes after the verb. However, training the neural network without a sense of memory may produce something like this:

> Dogs dogs dogs dogs cats cats chase cats chase dogs chase chase dogs

How do we solve this problem? This is where recursive neural networks come in ([Image Source](http://colah.github.io/posts/2015-08-Understanding-LSTMs))

![RNN](https://github.com/guangyic/projectludwig/blob/master/siteelements/rnn.png?raw=true)

A recursive neural network (RNN) takes the output and reinserts it into the model as an input. Using this, the model gains a sense of memory and is able to draw context out of the input. An LSTM is a special type of RNN that solves an issue called the long-term dependency problem. [Colah](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) does a fantastic job explaining the advntages of LSTMs. For my purposes here, I will just say that music generation requires the specialized LSTM due to how long musical phrases can be. 

---
I believe that should be enough background behind what questions I am looking to answer, and how I go about implementing this project. The following provides information and instructions on how to replicate what I did. 

Happy music generation!

---
## __Preparation__
1. Create a folder named "midi" in your working directory
2. These scripts were written in a Linux environment in macOS High Sierra. I used Python 2.7.14. If you are using Windows, ensure that the commands that reference directories use ' / ' instead of ' \ ' (I have some of these filled in here and there from when I used my Windows PC to train my model)
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

### _Required Non-Standard Python Packages (version numbers I used included)_
- [Keras](https://keras.io) ver 2.0.8
- [MuseScore](https://musescore.org) ver 2.1.0 rev 871c8ce
- [music21](http://web.mit.edu/music21/) ver 4.1.0
- [NumPy](http://www.numpy.org) ver 1.13.3
- [PyGame](https://www.pygame.org/news) ver 1.9.3
- [TensorFlow](https://www.tensorflow.org) ver 1.3.0

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
dictionary[str(training_set[n])] = {"notes":notes, "notes_offsets":notes_offsets, "chords":chords,
"chords_offsets":chords_offsets, "timesig":timesig, "metronome":mmark}}
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
Once the model has finished training, music generation can begin! 

In order to do this, the predictions from the model have to move back from Python-readable format to MIDI. At the moment, the output focuses on the chords from the original music in order to make predictions.

```python
# set up audio stream for generated music
out_stream = stream.Stream()

# Generation loop

# Establish the offset of each note/chord (starts at 0)
curr_offset = 0.0
loopEnd = len(giant_piece_chords)

# Create a music21 stream of the chords from the original music
curr_chords_master = stream.Voice()
for loopIndex in range(loopEnd):
    # get chords from file
    curr_chords_master.insert((giant_piece_chords_offsets[loopIndex].offset 
                               % 4), giant_piece_chords[loopIndex])

# The number of bars the output song will be
midi_loop_duration = 24

# This will create a list of random chords from the original music
# Using these chords, the model will predict which notes will
# follow based on the previous notes/chords
random_chord_picker = random.sample(range(loopEnd), midi_loop_duration)
```

Once this is done, a series of functions will take over to generate rests, chords, and notes based on the model. As these are done, another group of functions will remove notes or chords that are repeated or too close together (which creates dissonant sound). 

Finally, the music21 notation can be played with PyPlayer and then exported to MIDI format.

```python
# Play the final stream through output (see 'play' lambda function below)
play = lambda x: midi.realtime.StreamPlayer(x).play()
play(out_stream)

# Export music as MIDI
mf = midi.translate.streamToMidiFile(out_stream)
mf.open(out_fn, 'wb')
mf.write()
mf.close()
```

---
## __Results__

I have uploaded some of the outputs that I produced using the Beethoven sonatas onto SoundCloud. I completed these with 700 epochs. 

[![SoundCloud](https://github.com/guangyic/projectludwig/blob/master/siteelements/soundcloud.png?raw=true)](https://soundcloud.com/guang-yi-chua/sets/project-ludwig-output)

MuseScore also allows you to view the output in standard musical notation (should there be any interest in attempting to play this in real life).

![Music Output](https://github.com/guangyic/projectludwig/blob/master/siteelements/output_music.png?raw=true)

### Areas for Improvement
Currently, this model and code makes the assumption that the time and key signature of the piece is constant. With pieces that involve temporal or key modulation, this could lead to odd syncopation in the production of the original music, or to sequences of notes that do not necessarily make sense from a music theory standpoint. 

The current musical output is intended for piano. Looking at the sample output above, it would appear that the model does not yet know how to discriminate what is playable by human hands in terms of range and interval distance. I am not sure if this can be improved at this point in time, but it should be possible to implement output for more than one instrument.

I extract the time signature and metronome markings, but at the moment, do not use it for anything. Future versions will find a way to incorporate these into the generations. 

---
## Conclusion
This is an elementary implementation of deep learning with regards to musical training and production. Due to time constraints, I was not able to implement everything I had set out to do. Writing the rules for basic music theory (tonic-predominant-dominant-tonic) and variations of this proved difficult to do within the time frame, so the production rules are not as rigorous as I would like. I also outlined some of the areas I would like to improve above.

With this said, I gained a great understanding of how neural networks work in a real world application, and I hope that anyone that wishes to learn from or improve this code feels free to do so.
