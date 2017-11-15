'''
Author:     Chua Guang Yi
Project:    Ludwig S.T. Meinung (Ludwig for short)

'''

from collections import OrderedDict, defaultdict
from itertools import groupby
from music21 import *
import copy, random

#----------------------------MAIN FUNCTIONS-----------------------------------#

''' 
Main function that will transform the parsed MIDI data into the first machine-
readable format that we can fit into our model.
Called by __get_theory_syntax in extract.py
'''
def parse_melody(fullMeasureNotes, fullMeasureChords): #(voices, chords)
    
#    for streams in range(len(fullMeasureNotes_dict)) :
    # Remove extraneous elements.x
    measure = copy.deepcopy(fullMeasureNotes)
    chords_m = copy.deepcopy(fullMeasureChords)
    measure.removeByNotOfClass([note.Note, note.Rest])
    chords_m.removeByNotOfClass([chord.Chord])

    # Information for the start of the measure.
    # 1) measureStartTime: the offset for measure's start
    # 2) measureStartOffset: how long from the measure start to the first element.
    measureStartTime = measure[0].offset - (measure[0].offset % 4)
    measureStartOffset  = measure[0].offset - measureStartTime

    # Iterate over the notes and rests in measure, finding the rules for each
    # note in the measure and adding a  string for it. 

    fullRules = ""

    # Store previous note. Needed for intervals
    prevNote = None 

    # Number of non-rest notes/chords. Need for updating prevNote.
    numNonRests = 0 
    for ix, nr in enumerate(measure):
        # Get the last chord. If no last chord, then (assuming chords is
        # >0) shift first chord in chords to the beginning of the measure.
        try: 
            lastChord = [n for n in chords_m if n.offset <= nr.offset][-1]
        except IndexError:
            chords_m[0].offset = measureStartTime
            lastChord = [n for n in chords_m if n.offset <= nr.offset][-1]

        # Code the notes
        elementType = ' '
        # R: First, check if it's a rest. Clearly a rest --> only one possibility.
        if isinstance(nr, note.Rest):
            elementType = 'R'
        # C: Next, check to see if note pitch is in the last chord.
        elif nr.name in lastChord.pitchNames or isinstance(nr, chord.Chord):
            elementType = 'C'
        # S: Check if it's a scale tone.
        elif __is_scale_tone(lastChord, nr):
            elementType = 'S'
        # X: Otherwise, it's an arbitrary tone. Generate random chord.
        else:
            elementType = 'C'

        # SECOND, get the length for each element. e.g. 8th note = R8, but
        # to simplify things use the decimal (e.g. R,0.125)
        if (ix == (len(measure)-1)):
            # formula for a in "a - b": start of measure (e.g. 476) + 4
            diff = measureStartTime + 4.0 - nr.offset
        else:
            diff = measure[ix + 1].offset - nr.offset

        # Combine into the note info.
        noteInfo = "%s,%.3f" % (elementType, nr.quarterLength) 

        # THIRD, get the differences in range (max range up, max range down) based on where
        # the previous note was. Skips rests 
        intervalInfo = ""
        if isinstance(nr, note.Note):
            numNonRests += 1
            if numNonRests == 1:
                prevNote = nr
            else:
                noteDist = interval.Interval(noteStart=prevNote, noteEnd=nr)
                noteDistUpper = interval.add([noteDist, "m3"])
                noteDistLower = interval.subtract([noteDist, "m3"])
                intervalInfo = ",<%s,%s>" % (noteDistUpper.directedName, 
                    noteDistLower.directedName)
                prevNote = nr

        # Return. Do lazy evaluation for real-time performance.
        grammarTerm = noteInfo + intervalInfo 
        fullRules += (grammarTerm + " ")

    return fullRules.rstrip()

''' 
Main function that strips down a measure into individual chords and notes
given a string of notation
Called in generate in main.py 
'''
def unparse_rules(rules, chords):
    elements = stream.Voice()

    # To recalculate last chord.
    currOffset = 0.0 
    prevElement = None
    for ix, ruleElement in enumerate(rules.split(' ')):
        terms = ruleElement.split(',')
        currOffset += float(terms[1]) # works just fine

        # Case 1: it's a rest. Just append
        if terms[0] == 'R':
            rNote = note.Rest(quarterLength = float(terms[1]))
            elements.insert(currOffset, rNote)
            continue

        # Get the last chord first so you can find chord note, scale note, etc.
        try: 
            lastChord = [n for n in chords if n.offset <= currOffset][-1]
        except IndexError:
            chords[0].offset = 0.0
            lastChord = [n for n in chords if n.offset <= currOffset][-1]

        # If no < > to indicate next note range, use this. Usually from
        # first note or for rests
        if (len(terms) == 2): # If no < >.
            insertNote = note.Note() # default is C

            # Case C: chord 
            if terms[0] == 'C':
                insertNote = __generate_chord_tone(lastChord)

            # Case S: scale tone
            elif terms[0] == 'S':
                insertNote = __generate_scale_tone(lastChord)

            # Case X: Anything else
            # Insert a chord
            else:
                insertNote = __generate_chord_tone(lastChord)

            # Update the stream of generated notes
            insertNote.quarterLength = float(terms[1])
            if insertNote.octave < 4:
                insertNote.octave = 4
            elements.insert(currOffset, insertNote)
            prevElement = insertNote

        # If < > is present. Usually for notes after the first one.
        else:
            # Get lower, upper intervals and notes.
            interval1 = interval.Interval(terms[2].replace("<",''))
            interval2 = interval.Interval(terms[3].replace(">",''))
            if interval1.cents > interval2.cents:
                upperInterval, lowerInterval = interval1, interval2
            else:
                upperInterval, lowerInterval = interval2, interval1
            lowPitch = interval.transposePitch(prevElement.pitch, lowerInterval)
            highPitch = interval.transposePitch(prevElement.pitch, upperInterval)
            numNotes = int(highPitch.ps - lowPitch.ps + 1) 

            # Case C: chord must be within increment (terms[2]).
            # First, transpose note with lowerInterval to get note that is
            # the lower bound. Then iterate over, and find valid notes. Then
            # choose randomly from those.
            
            if terms[0] == 'C':
                relevantChordTones = []
                for i in xrange(0, numNotes):
                    currNote = note.Note(lowPitch.transpose(i).simplifyEnharmonic())
                    if __is_chord_tone(lastChord, currNote):
                        relevantChordTones.append(currNote)
                if len(relevantChordTones) > 1:
                    insertNote = random.choice([i for i in relevantChordTones
                        if i.nameWithOctave != prevElement.nameWithOctave])
                elif len(relevantChordTones) == 1:
                    insertNote = relevantChordTones[0]
                else: # if no choices, set to prev element +-1 whole step
                    insertNote = prevElement.transpose(random.choice([-2,2]))
                if insertNote.octave < 3:
                    insertNote.octave = 3
                insertNote.quarterLength = float(terms[1])
                elements.insert(currOffset, insertNote)

            # Case S: scale note, must be within increment.
            elif terms[0] == 'S':
                relevantScaleTones = []
                for i in xrange(0, numNotes):
                    currNote = note.Note(lowPitch.transpose(i).simplifyEnharmonic())
                    if __is_scale_tone(lastChord, currNote):
                        relevantScaleTones.append(currNote)
                if len(relevantScaleTones) > 1:
                    insertNote = random.choice([i for i in relevantScaleTones
                        if i.nameWithOctave != prevElement.nameWithOctave])
                elif len(relevantScaleTones) == 1:
                    insertNote = relevantScaleTones[0]
                else: # if no choices, set to prev element +-1 whole step
                    insertNote = prevElement.transpose(random.choice([-2,2]))
                if insertNote.octave < 3:
                    insertNote.octave = 3
                insertNote.quarterLength = float(terms[1])
                elements.insert(currOffset, insertNote)

            # Case X: Everything else.
            else:
                relevantChordTones = []
                for i in xrange(0, numNotes):
                    currNote = note.Note(lowPitch.transpose(i).simplifyEnharmonic())
                    if __is_chord_tone(lastChord, currNote):
                        relevantChordTones.append(currNote)
                if len(relevantChordTones) > 1:
                    insertNote = random.choice([i for i in relevantChordTones
                        if i.nameWithOctave != prevElement.nameWithOctave])
                elif len(relevantChordTones) == 1:
                    insertNote = relevantChordTones[0]
                else: # if no choices, set to prev element +-1 whole step
                    insertNote = prevElement.transpose(random.choice([-2,2]))
                if insertNote.octave < 3:
                    insertNote.octave = 3
                insertNote.quarterLength = float(terms[1])
                elements.insert(currOffset, insertNote)

            # Update the previous element.
            prevElement = insertNote

    return elements    

#----------------------------CALLED BY MAIN FUNCTIONS-------------------------#

''' 
Sub-function to determine is tone is a scale tone
'''
def __is_scale_tone(chord, note):
    # Method: generate all scales that have the chord notes then check if note is
    # in names

    # Derive major or minor scales (minor if 'other') based on the quality
    # of the chord.
    scaleType = scale.MinorScale() # i.e. minor 
    if chord.quality == 'major':
        scaleType = scale.MajorScale()
    # Can change later to deriveAll() for flexibility. If so then use list
    # comprehension of form [x for a in b for x in a].
    scales = scaleType.derive(chord) # use deriveAll() later for flexibility
    allPitches = list(set([pitch for pitch in scales.getPitches()]))
    allNoteNames = [i.name for i in allPitches] # octaves don't matter

    # Get note name. Return true if in the list of note names.
    noteName = note.name
    return (noteName in allNoteNames)

''' 
Sub-function to determine is tone is a chord tone
'''
def __is_chord_tone(lastChord, note):
    return (note.name in (p.name for p in lastChord.pitches))

''' 
Sub-function to generate a chord tone
'''
def __generate_chord_tone(lastChord):
    lastChordNoteNames = [p.nameWithOctave for p in lastChord.pitches]
    return note.Note(random.choice(lastChordNoteNames))

''' 
Sub-function to generate a scale tone
'''
def __generate_scale_tone(lastChord):
    # Derive major or minor scales (minor if 'other') based on the quality
    # of the lastChord.
    scaleType = scale.MinorScale() # minor pentatonic
    if lastChord.quality == 'major':
        scaleType = scale.MajorScale()
    # Can change later to deriveAll() for flexibility. If so then use list
    # comprehension of form [x for a in b for x in a].
    scales = scaleType.derive(lastChord) # use deriveAll() later for flexibility
    allPitches = list(set([pitch for pitch in scales.getPitches()]))
    allNoteNames = [i.name for i in allPitches] # octaves don't matter

    # Return a note (no octave here) in a scale that matches the lastChord.
    sNoteName = random.choice(allNoteNames)
    lastChordSort = lastChord.sortAscending()
    sNoteOctave = random.choice([i.octave for i in lastChordSort.pitches])
    sNote = note.Note(("%s%s" % (sNoteName, sNoteOctave)))
    return sNote

''' 
Sub-function to generate a random tone
'''
def __generate_arbitrary_tone(lastChord):
    return __generate_chord_tone(lastChord) 