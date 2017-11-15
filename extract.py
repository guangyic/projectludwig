'''
Author:     Chua Guang Yi
Project:    Ludwig S.T. Meinung (Ludwig for short)

'''

# Import necessary packages
from __future__ import print_function
from music21 import *

# Import from function called musicology
from musicology import *

#----------------------------MAIN FUNCTION-----------------------------------#

''' 
Extract musical data from MIDI files
Called by generate in main.py 
'''
def get_musical_data(training_set, midi_number, cwd):
    dictionary = __parse_midi(training_set, midi_number, cwd)
    theory_syntax = __get_theory_syntax(dictionary)

    return dictionary, theory_syntax

''' 
Get corpus data from theory syntax data
Called by generate in main.py 
'''
def get_corpus_data(theory_syntax):
    
    # This combines all the pieces into one array
    theory_list = [theory_syntax[a] for a in theory_syntax]
    theory_list_fin = [] 

    for a in range(len(theory_list)):
        theory_list_fin += theory_list[a]
    
    grouped_string = "".join(theory_list_fin)
    corpus = grouped_string.split(' ')
    values = set(corpus)
    val_indices = dict((v, i) for i, v in enumerate(values))
    indices_val = dict((i, v) for i, v in enumerate(values))
    
    return theory_list_fin, corpus, values, val_indices, indices_val

#----------------------------CALLED BY MAIN FUNCTION-------------------------#
    
''' 
Sub-function to parse a MIDI file for chords, key, time signature, voices
Called by get_musical_data above
'''
def __parse_midi(training_set, midi_number, cwd):
    
    dictionary = {}
    
    for n in range(midi_number) :
        
        # Read in each piece in the list
        print("Parsing piece number %s"%n)
        data = str(cwd) + '/' + str(training_set[n])
        
        # Parse the MIDI data for separate melody and accompaniment parts,
        # then flatten the parts into one line
        midi_data = converter.parse(data)
        midi_data_flat = midi_data.flat
        
        # Extracts the time signature and the metronome marking from the MIDI
        #timesig = midi_data_flat.getElementsByClass(meter.TimeSignature)[0]
        #mmark = midi_data_flat.getElementsByClass(tempo.MetronomeMark)[0]
        
        # Extracts the notes and chords from the MIDI  
        notes = midi_data_flat.getElementsByClass(note.Note).notes
        chords = midi_data_flat.getElementsByClass(chord.Chord)
        
        # Extracts the temporal offsets of the notes and chords from the MIDI  
        notes_offsets = [a.offset for a in notes]
        chords_offsets = [b.offset for b in chords]
        
        dictionary[str(training_set[n])] = {"notes":notes, "notes_offsets":notes_offsets, 
                  "chords":chords, "chords_offsets":chords_offsets}
                  #"timesig":timesig, "metronome":mmark}

    return dictionary

''' 
Sub-function to parse a MIDI file for musicology rules
Called by get_musical_data above
'''
def __get_theory_syntax(dictionary):
    
    voices = {} #m or measures equivalent in original
    chords = {} #c or chords equivalent in original
    theory_syntax = {} 
    
    
    for ix in dictionary :
        print("Getting theory rules for %s"%ix)
        n = stream.Voice()
        c = stream.Voice()
        for notes in range(len(dictionary[ix]["notes"])) :
            n.insert(dictionary[ix]["notes_offsets"][notes],dictionary[ix]["notes"][notes])
        voices[ix] = n 
        
        for corda in range(len(dictionary[ix]["chords"])) :
            c.insert(dictionary[ix]["chords_offsets"][corda],dictionary[ix]["chords"][corda])
        chords[ix] = c
        
        if len(voices[ix]) < 1 :
            parsed = None
        elif len(chords[ix]) < 1 :
            parsed = None
        else :
            parsed = parse_melody(voices[ix], chords[ix])
        
        theory_syntax[ix] = parsed
        
    # Remove null voices of "parsed" values
    keys = theory_syntax.keys()
    for key in keys :
        if theory_syntax[key] is None :
            del theory_syntax[key]

    return theory_syntax