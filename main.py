'''
Author:     Chua Guang Yi
Project:    Ludwig S.T. Meinung (Ludwig for short)

'''

# Import necessary packages
from __future__ import print_function
from music21 import *
import datetime
import os
import numpy as np
import sys
import pickle

# Import from accompanying python scripts
from extract import *
from training import *
from check import *
from keras.models import load_model

#----------------------------INITIATING FUNCTION------------------------------#

''' 
Runs main function -- generating, playing, then storing a musical sequence 
with the files in your MIDI folder.
Calling "python main.py 5000" runs 5000 epochs. Default is 700 
'''
def initiate(N):
    try:
        epochs = int(N[1])
    except:
        epochs = 700 # default is 700

    # Finds MIDI files for training and generation
    cwd = os.getcwd() + '/midi'
    training_set = [s for s in os.listdir(cwd) if s.endswith('.mid')]
    midi_number = len(training_set)
    out_fn = 'midi/' 'ludwig' + str(epochs) + '_epochs.midi'

    generate(training_set, out_fn, epochs, midi_number, cwd)

# If run as script, execute main function above
if __name__ == '__initiate__':
    import sys
    initiate(sys.argv)
    
#----------------------------MAIN FUNCTION-----------------------------------#
''' 
Generates musical sequence based on the given training pieces and settings
Plays then stores (MIDI file) the generated output
Called by initiate (the initiating function) above 
'''

def generate(training_set, out_fn, epochs, midi_number, cwd):
    # Model settings
    bars = 100
    max_tries = 100
    diversity = 0.5

    # Tempo setting
    bpm = 100

    # Read sequences from training MIDI files to make it Python readable
    print(datetime.datetime.now())
    print("Parsing MIDI files and obtaining theory syntax...")
    dictionary, theory_syntax = get_musical_data(training_set, midi_number, cwd)
    #pickle.dump(theory_syntax, open("theory_syntax.p", "wb"))
    
    print("Preparing theory arrays for training...")    
    theory_list_fin, corpus, values, val_indices, indices_val = get_corpus_data(theory_syntax)
#    pickle.dump(theory_list_fin, open("theory_list_fin.p", "wb"))
#    pickle.dump(corpus, open("corpus.p", "wb"))
#    pickle.dump(values, open("values.p", "wb"))
#    pickle.dump(val_indices, open("val_indices.p", "wb"))
#    pickle.dump(indices_val, open("indices_val.p", "wb"))
#    
#    # To load pickles, uncomment below
#    #dictionary = pickle.load(open("dictionary.p", "rb"))
#    theory_syntax = pickle.load(open("theory_syntax.p", "rb"))
#    theory_list_fin = pickle.load(open("theory_list_fin.p", "rb"))
#    corpus = pickle.load(open("corpus.p", "rb"))
#    values = pickle.load(open("values.p", "rb"))
#    val_indices = pickle.load(open("val_indices.p", "rb"))
#    indices_val = pickle.load(open("indices_val.p", "rb"))
    
    # Compress dictionary elements into one giant piece
    giant_piece_chords = []
    for a in dictionary :
        for b in dictionary[a]["chords"] :
            giant_piece_chords.append(b)
            
    giant_piece_notes = []
    for a in dictionary :
        for b in dictionary[a]["notes"] :
            giant_piece_notes.append(b)
            
    giant_piece_notes_offsets = []
    for a in dictionary :
        for b in dictionary[a]["notes"] :
            giant_piece_notes_offsets.append(b)
            
    giant_piece_chords_offsets = []
    for a in dictionary :
        for b in dictionary[a]["notes"] :
            giant_piece_chords_offsets.append(b)
  
    # Build model
    model = build_model(corpus=corpus, val_indices=val_indices, bars=bars,
                        epochs=epochs)

    # Set up audio stream
    music_gen = stream.Stream()

    # Generation loop
    curr_offset = 0.0
    loopEnd = len(giant_piece_chords)
    
    curr_chords_master = stream.Voice()
    for loopIndex in range(loopEnd):
        # Get chords from files
        curr_chords_master.insert((giant_piece_chords_offsets[loopIndex].offset 
                                   % 4), giant_piece_chords[loopIndex])
    
    # Set how long the output piece should be
    midi_loop_duration = 24
    random_chord_picker = random.sample(range(loopEnd), midi_loop_duration)
    counter = 0
    
    for loopIndex in range(loopEnd):
        # Get chords from file and insert into new music21 stream
        curr_chords = stream.Voice()
        curr_chords.insert((giant_piece_chords[loopIndex].offset % 4), giant_piece_chords[loopIndex])
    
    # Pick random chords to generate from
    for rand_chord in random_chord_picker:
        counter += 1
        print("Generating bar number", str(counter), "of", str(len(random_chord_picker)))
        curr_chords = stream.Voice()
        
        curr_chords.insert((curr_chords_master[rand_chord].offset % 4), curr_chords_master[rand_chord])
    
        # Generate rules (called below)
        print("Generating rules")
        curr_rules = __generate_rules(model=model, corpus=corpus, 
                                        theory_syntax=corpus, values=values, 
                                        val_indices=val_indices, 
                                        indices_val=indices_val, bars=bars, 
                                        max_tries=max_tries, 
                                        diversity=diversity)
        curr_rules = curr_rules.replace(' A',' C').replace(' X',' C')

        # Pruning #1: Smoothing measure
        curr_rules = prune_rules(curr_rules)

        # Get notes from grammar and chords
        curr_notes = unparse_rules(curr_rules, curr_chords)

        # Pruning #2: Remove notes that are too close together or repeated
        curr_notes = prune_notes(curr_notes)

        # Checking produced notes: clean up notes
        curr_notes = clean_up_notes(curr_notes)

        # Print number of note in curr_notes
        print('After pruning: %s notes' % (len([i for i in curr_notes
            if isinstance(i, note.Note)])))

        # Insert into the output stream
        for m in curr_notes:
            music_gen.insert(curr_offset + m.offset, m)
        for mc in curr_chords:
            music_gen.insert(curr_offset + mc.offset, mc)

        curr_offset += 4.0

    music_gen.insert(0.0, tempo.MetronomeMark(number=bpm))

    # Play the final stream through output (see 'play' lambda function below)
    #play = lambda x: midi.realtime.StreamPlayer(x).play()
    #play(music_gen)

    # Export music as MIDI
    mf = midi.translate.streamToMidiFile(music_gen)
    mf.open(out_fn, 'wb')
    mf.write()
    mf.close()
    
#----------------------------CALLED BY MAIN FUNCTION-------------------------#

''' 
Sub-function to generate a predicted value from a given matrix
'''
def __predict(model, x, indices_val, diversity):
    preds = model.predict(x, verbose=0)[0]
    next_index = __sample__(preds, diversity)
    next_val = indices_val[next_index]

    return next_val
 
# Sub-sub-function to sample an index from a probability array
def __sample__(preds, diversity=1.0):
    preds = np.log(preds) / diversity 
    dist = np.exp(preds)/np.sum(np.exp(preds)) 
    choices = range(len(preds)) 

    return np.random.choice(choices, p=dist)

''' 
Sub-function to use the musicological rules from the model to generate its own
rules given a set of pieces, indices_val (mapping), theory_syntax (list),
and diversity floating point value
'''
def __generate_rules(model, corpus, theory_syntax, values, val_indices,
                       indices_val, bars, max_tries, diversity):
    # Pre-establish variables
    curr_rules = ''
    start_index = np.random.randint(0, len(corpus) - bars)
    sentence = corpus[start_index: start_index + bars] 
    running_length = 0.0

    while running_length <= 4.1:

        # Transform musical sentence to matrix
        x = np.zeros((1, bars, len(values)))
        for t, val in enumerate(sentence):
            if (not val in val_indices): print(val)
            x[0, t, val_indices[val]] = 1.
            
        next_val = __predict(model, x, indices_val, diversity)

        # Fix first note: must not have < > and should not be a rest
        if (running_length < 0.00001):
            tries = 0
            while (next_val.split(',')[0] == 'R' or 
                len(next_val.split(',')) != 2):
                # Give up after 100 tries; random from input's first notes
                if tries >= max_tries:
                    rand = np.random.randint(0, len(theory_syntax))
                    next_val = theory_syntax[rand].split(' ')[0]
                else:
                    next_val = __predict(model, x, indices_val, diversity)

                tries += 1

        # Shift musical sentence over with new value
        sentence = sentence[1:] 
        sentence.append(next_val)

        # Add a ' ' separator (skip first one)
        if (running_length > 0.00001): curr_rules += ' '
        curr_rules += next_val
        
        length = float(next_val.split(',')[1])
        running_length += length
        print(length, next_val)

    return curr_rules

