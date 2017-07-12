#!/usr/bin/env python

"""
- Split sound into e.g. 128-sample windows
- Treat each window as one sample in 128-dimensional space
- Do PCAs to 2 dimensions on all the samples
- On each frame, draw the PCs of e.g. every 16th sample since the last frame
"""

from __future__ import print_function, division
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("wav_file")
parser.add_argument("--fps", type=int, default=60)
parser.add_argument("--sparsity", type=int, default=16)
parser.add_argument("--point_size", type=int, default=5)
args = parser.parse_args()

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from sklearn.decomposition import PCA, IncrementalPCA
import pickle
import os.path
import math

pca_filename = args.wav_file + ".pca.pickle"
output_filename = args.wav_file.replace('.wav', '') + '.mp4'

print("Loading sound file...")
(rate, data) = wavfile.read(args.wav_file)
samples = data[:, 0]

samples = np.array(samples)
window_len_samples = 128
# stride argument: stride for each dimension of the resulting array
# this is set up for stride=1 right now
windows = np.lib.stride_tricks.as_strided(
    samples,
    shape=(len(samples) - window_len_samples + 1, window_len_samples),
    strides=(samples.itemsize, samples.itemsize))

try:
    with open(pca_filename, 'rb') as f:
        print("Loading PCA...")
        pca = pickle.load(f)
except FileNotFoundError:
    print("Generating PCA...")
    pca = IncrementalPCA(n_components=2)
    batch_size = 8192
    n_batches = math.floor(len(windows) / batch_size)
    for i in range(n_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        pca.partial_fit(windows[start:end])
        percent_complete = i  / n_batches * 100
        print("%.1f%% complete" % percent_complete)
    print("Done!")
    with open(pca_filename, 'wb') as f:
        pickle.dump(pca, f)

fig = plt.figure(figsize=(4, 4))
plt.xlim([-150000, 150000])
plt.ylim([-150000, 150000])

scat = plt.scatter([0], [0], s=args.point_size, c='w')
# set black background
fig.axes[0].set_facecolor('black')
# remove borders
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

FFMpegWriter = manimation.writers['ffmpeg']
writer = FFMpegWriter(fps=args.fps)

writer.setup(fig, output_filename, dpi=100)
frame_n = 1
while True:
    print("Frame %d" % frame_n)
    t1 = int(((frame_n - 1) / args.fps) * rate)
    t2 = int((frame_n / args.fps) * rate)
    if t2 > len(windows):
        break
    w = windows[t1:t2:args.sparsity]
    transformed_windows = pca.transform(w)
    scat.set_offsets(transformed_windows)
    writer.grab_frame()
    frame_n += 1
writer.finish()

print("Now run:")
print("ffmpeg -i %s -i %s -c copy -y %s" %
        (output_filename,
         args.wav_file,
         output_filename.replace('.mp4', '.mkv')))
