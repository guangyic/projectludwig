# Project Ludwig: Music Generation through Deep Learning
---
```html
<center>
<img src="https://github.com/guangyic/projectludwig/blob/master/siteelements/beethoven.png?raw=true"style="width:200px;height:300px;">
</center>
```
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