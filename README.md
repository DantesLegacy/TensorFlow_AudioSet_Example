# TensorFlow_AudioSet_Example
This repository contains code that was used as an example of how to use Python to download part of the AudioSet dataset and use Tensorflow to create a neural network to predict the labels associated with the audio samples.

Firstly, I have to acknowledge and thank GitHub user aqibsaeed for contributing their 'Urban sound classification using Deep Learning' code. I essentially copied what they have done and applied it to the AudioSet dataset rather than the Urban8k dataset. Most of my code is essentially the same as what you will see in their repo. Speacking of which, you can find it here:

https://github.com/aqibsaeed/Urban-Sound-Classification

You can also find their very useful post explaining in detail how to use their code here:

http://www.kdnuggets.com/2016/09/urban-sound-classification-neural-networks-tensorflow.html

So all credit where credit is due.

Secondly, the results from my code are not great. I get an F-Score of .0342 (34.2% accuracy) and the learning graph is clearly wrong. This is because I do not know a huge amount about neural networks nor TensorFlow itself. So I kept the parameters from aqibsaeed's code when setting up the neural network. However, I am sure with time, I can improve on this, or someone who knows a good deal more about neural networks and Tensorflow will contribute some useful knowledge as to how to improve the results.

I must also mention that this code was completed as part of a module for a Masters of Science in Computing degree  in Dublin Institute of Technology. Namely, the Speech & Audio Processing module given by Dr. Andrew Hines.

I used an Ubuntu 16.04 operating system with PycharmEdu 3.5.1 IDE and Python 3.5.2 to get this code running. You will also need to have the following installed on your Linux OS:
youtube-dl
ffmpeg

You will require the following Python libraries as well:
csv
sys
os
wave
contextlib
glob
librosa
numpy
matplotlib
tensorflow
sklearn
itertools

I would recommend following aqibsaeed's post and getting their code set up first, then come back to this. That way, you should only have to install youtube-dl and ffmpeg on your Linux OS and everything *should* work... Maybe...

So first thing you need to do is get the relevant audio samples. I chose the balanced train dataset from the AudioSet dataset. You can see the details of this here:
https://research.google.com/audioset/download.html

Also, you can download the CSV required for this balanced train dataset here:
http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv

I've added this CSV file to my repo, but it wouldn't hurt to download and compare against my copy in case it has been updated. Also, you could just as easily get the unbalanced train or evaluation datasets if you wish and edit the code to download and process them instead. For now, I will presume you want to download the balanced train dataset and process that.

You will also need the 'class_labels_indices.csv' file for the code to proces the labels for each of the audio samples correctly. This can be found here:
http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv

Again, download and compare to the version in the repo as they may have updated it.

You may notice that you can't just simply download the audio samples from the YouTube videos in one handy download, that is why I created this code.

First things first, create a 'balanced_train' and 'error_files' directory in the 'AudioSet' directory. You will need them later.

You need to run the 'get_youtube_dataset_balanced.py' python script which should download all the required audio files into the 'balanced_train' directory. Be aware that some video have been taken down, so when the script tries to download those particular files, it will just create an empty file with '_ERROR' in the file name. I would recommend that after this code has finished running, move all the files that contain '_ERROR' into the 'error_files' directory.

Once the files have been downloaded, and the 'get_youtube_dataset_balanced.py' code has finished running, you can then run the 'neural_network_audioset.py' code to process all the audio samples in the 'balanced_train' directory and create a neural network for classification of the samples using TensorFlow.

Please be aware that it will take a long time to download over 22k samples from YouTube in the first Python code. then it will take quite awhile to process each of those files in the 'neural_network_audioset.py' code as well as create a neural network for classification.

For example, on my budget laptop with the following specs:
Memory: 3.8 GiB
Processor: Intel® Core™ i3-4030U CPU @ 1.90GHz × 4
(Please note that TensorFlow used the CPU and not the GPU on this machine, if you have a good GPU - it could make this go alot faster)
It took ~30 hours to download the 10-second audio samples (this is more based on network speed than CPU).
It took ~6 hours to extract all the features from all the audio files.
It took ~3 hours for the TensorFlow neural network to complete.

There were some occasions when running the code that processes the audio files and extract the features from them that the program would just hang like it was stuck in an infinite loop somewhere. In order to get around this, I had to remove certain files from my dataset. I still don't know why it did this, but (apart from the files that contained '_ERROR' in the name) I had to remove two files from the dataset in the 'balanced_dataset' directory.

Anyway, I hope this bit of work can help someone to get up and running to create a proper classification model with the AudioSet dataset. Also, again, thanks to aqibsaeed for his great work in getting me this far.
