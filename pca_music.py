#!/usr/bin/env python

"""
Make a music visualiser out of PCA applied to windows of samples.
- Split sound into e.g. 128-sample windows
- Treat each window as one sample in 128-dimensional space
- Do PCAs to 2 dimensions on all the samples
- Render a video, on each frame drawing all the samples since the last frame
  (or rather, with sparsity=16, every 16th sample since the last frame)
"""

from __future__ import print_function, division
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("wav_file")
parser.add_argument("--fps", type=int, default=60)
parser.add_argument("--sparsity", type=int, default=16)
parser.add_argument("--point_size", type=int, default=5)
parser.add_argument("--window-len", type=int, default=128)
args = parser.parse_args()

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from sklearn.decomposition import PCA, IncrementalPCA
import pickle
import os.path
import math

# Load data

pca_filename = args.wav_file + ".pca.pickle"
output_filename = args.wav_file.replace('.wav', '') + '.mp4'

print("Loading sound file...")
(rate, data) = wavfile.read(args.wav_file)
samples = data[:, 0]

# Generate windows

samples = np.array(samples)
# Get a view of array which looks like a bunch of strides taken
# across the original - so we can get windowed data without actually
# having to allocate memory for all the windows.
# (stride argument: stride for each dimension of the resulting array.
# This is set up for stride=1 right now.)
windows = np.lib.stride_tricks.as_strided(
    samples,
    shape=(len(samples) - args.window_len + 1, args.window_len),
    strides=(samples.itemsize, samples.itemsize))

# Do PCA
# (Or load precomputed PCs if available)

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

# Draw the initial frame

fig = plt.figure(figsize=(4, 4))
lims = [0, 0]

scat = plt.scatter([0], [0], s=args.point_size, c='w')
# set black background
fig.axes[0].set_facecolor('black')
# remove borders
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Render the video

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

    # make sure the axis limits are always enough to show all the points
    dlims = (np.min(transformed_windows), np.max(transformed_windows))
    if dlims[0] < lims[0]:
        lims[0] = dlims[0]
        plt.xlim(xmin=lims[0])
        plt.ylim(ymin=lims[0])
    if dlims[1] > lims[1]:
        lims[1] = dlims[1]
        plt.xlim(xmax=lims[1])
        plt.ylim(ymax=lims[1])

    scat.set_offsets(transformed_windows)
    writer.grab_frame()
    frame_n += 1
writer.finish()

print("Wrote video to %s" % output_filename)
print("To combine audio and video, run something like:")
print("ffmpeg -i %s -i %s -c copy -y %s" %
        (output_filename,
         args.wav_file,
         output_filename.replace('.mp4', '.mkv')))
