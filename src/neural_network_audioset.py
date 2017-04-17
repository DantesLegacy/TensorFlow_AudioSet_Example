import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
#%matplotlib inline
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support

import csv
import sys
import itertools

plt.style.use('ggplot')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 13

audioset_csv = 'AudioSet/balanced_train_segments.csv'
audioset_indices_csv = 'AudioSet/class_labels_indices.csv'

def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X,sr = librosa.load(fp)
        raw_sounds.append(X)
    return raw_sounds

def plot_waves(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60), dpi = 900)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        librosa.display.waveplot(np.array(f),sr=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 1: Waveplot',x=0.5, y=0.915,fontsize=18)
    plt.show()

def plot_specgram(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60), dpi = 900)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        specgram(np.array(f), Fs=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 2: Spectrogram',x=0.5, y=0.915,fontsize=18)
    plt.show()

def plot_log_power_specgram(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60), dpi = 900)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        D = librosa.logamplitude(np.abs(librosa.stft(f))**2, ref_power=np.max)
        librosa.display.specshow(D,x_axis='time' ,y_axis='log')
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 3: Log power spectrogram',x=0.5, y=0.915,fontsize=18)
    plt.show()

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def get_file_name_labels_from_audioset_csv(row_num):
    str_labels = []
    int_labels = []
    # Open the CSV file
    with open(audioset_csv, 'r') as f:
        # Skip to the line we need.
        # I think this should stop it loading it up in memory for each item.
        # Also, we add 3 to the number to skip the header in the CSV file.
        # We could search through the CSV file using the file name, but this
        #  will probably take much longer.
        line = next(itertools.islice(csv.reader(f), int(row_num) + 3, None))
        # Now that we have the line we need, we need to grab the labels from it
        # This file may have multiple labels, so we need to account for that
        for element in line[3:]:
            if (element.startswith(' "')) and (element.endswith('"')):
                str_labels.append(element[2:-1])
            elif element.startswith(' "'):
                str_labels.append(element[2:])
            elif element.endswith('"'):
                str_labels.append(element[:-1])
            else:
                str_labels.append(element)

    # Now we have the string version of the labels.
    # Let's convert them to int versions
    for element in str_labels:
        with open(audioset_indices_csv, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[1] == element:
                    int_labels.append(int(row[0]))

    return int_labels

def parse_audio_files(parent_dir,sub_dirs,file_ext='*.wav'):
    # The reason this is set to 193 is because that is how big the features
    #  array needs to be once it has extracted all the necessary features
    #  from each audio file
    features, labels = np.empty((0,193)), np.empty(0)
    file_labels = []
    count  = 1
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            # The print() calls here are just for debugging. Feel free to remove them.
            print(fn)
            mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
            print('features for ' + fn + ' extracted.')
            ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            print('features horizontally stacked for numpy array')
            features = np.vstack([features, ext_features])
            print('features vertically stacked to pre-existing features array')
            print('features numpy array size = ' + str(features.size))

            # Need to get the labels for each clip from the CSV file
            rownum = fn.split(sub_dir)[1].split('/')[1].split('_')[0]
            print('rownum = ' + str(rownum))
            file_labels.append(get_file_name_labels_from_audioset_csv(rownum))
            print('Processing File #' + str(count))

            count += 1

    return np.array(features), np.array(file_labels)


def one_hot_encode(labels):
    n_labels = len(labels)
    print(n_labels)
    #n_unique_labels = len(np.unique(labels))
    # I'll set this to the actual number of labels for now
    #  Otherwise it gets screwed up and thinks multiple labelled files are a single label
    n_unique_labels = 527
    print(n_unique_labels)
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    # Mark the relevant values in the area as '1'
    #  This can be multiple elements in the array as there can be
    #  multiple labels to an audio sample
    for index in range(n_labels):
        for element in labels[index]:
            one_hot_encode[index, element] = 1

    return one_hot_encode

parent_dir = 'AudioSet'

# Mave to this when tests are complete
sub_dirs = ['balanced_train']
features, labels = parse_audio_files(parent_dir,sub_dirs)

labels = one_hot_encode(labels)

# I used the following code while debugging.
# It takes quite awhile to process all the files in the AudioSet balanced
#  dataset. So you can save the numpy arrays here and then load them up again
#  when you want to kick off the TensorFlow neural network session. The idea is
#  that you only need to proces the audio files once, but you may want to tweak
#  your TensorFlow settings to get better results. So no need to proces the
#  audio files again, just load them up from a file with this bit of code.

# Save numpy arrays for later use
#np.savetxt("labels.csv", labels, delimiter=",")
#np.savetxt("features.csv", features, delimiter=",")

# Load up numpy arrays to save time if they have already been procesed.
#labels = np.loadtxt("labels.csv", delimiter=",")
#features = np.loadtxt("features.csv", delimiter=",")

train_test_split = np.random.rand(len(features)) < 0.95
train_x = features[train_test_split]
train_y = labels[train_test_split]
test_x = features[~train_test_split]
test_y = labels[~train_test_split]

training_epochs = 5000
n_dim = features.shape[1]
# There are 527 labels in the AudioSet dataset
n_classes = 527
n_hidden_units_one = 280
n_hidden_units_two = 300
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.01

X = tf.placeholder(tf.float32,[None,n_dim])
Y = tf.placeholder(tf.float32,[None,n_classes])

W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_one], mean = 0, stddev=sd))
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd))
h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1)


W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two], mean = 0, stddev=sd))
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd))
h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2)


W = tf.Variable(tf.random_normal([n_hidden_units_two,n_classes], mean = 0, stddev=sd))
b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd))
y_ = tf.nn.softmax(tf.matmul(h_2,W) + b)

init = tf.initialize_all_variables()

cost_function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost_history = np.empty(shape=[1],dtype=float)
y_true, y_pred = None, None
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        _,cost = sess.run([optimizer,cost_function],feed_dict={X:train_x,Y:train_y})
        cost_history = np.append(cost_history,cost)

    y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: test_x})
    y_true = sess.run(tf.argmax(test_y,1))

fig = plt.figure(figsize=(10,8))
plt.plot(cost_history)
plt.ylabel("Cost")
plt.xlabel("Iterations")
plt.axis([0,training_epochs,0,np.max(cost_history)])
plt.show()

# Uncomment if you wish to save the graph
#plt.savefig('training.pdf')
#plt.savefig('training.png')

p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average='micro')
print("F-Score:")
print(round(f,3))
