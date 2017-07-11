from __future__ import print_function, division
import numpy as np
from scipy.io import wavfile
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import struct
from sklearn.decomposition import  PCA, IncrementalPCA
from matplotlib.animation import FuncAnimation
import pyaudio
import math
import time
import pickle

(rate, data) = wavfile.read('closer.wav')
samples = data[:, 0]

samples = np.array(samples)
window_len_samples = 128
# stride argument: stride for each dimension of the resulting array
# this is set up for stride=1 right now
windows = np.lib.stride_tricks.as_strided(samples,
        shape=(len(samples)-window_len_samples+1,window_len_samples),
        strides=(samples.itemsize, samples.itemsize))

"""
if True:
    pca = IncrementalPCA(n_components=2)
    batch_size = 10000
    n_batches = math.floor(len(windows) / batch_size)
    for i in range(n_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        pca.partial_fit(windows[start:end])
        percent_complete = i  / n_batches * 100
        print("%.1f" % percent_complete)
else:
    pca = PCA(n_components=2)
    pca.fit(windows)
with open('pca', 'wb') as f:
    pickle.dump(pca, f)
"""


with open('pca', 'rb') as f:
    pca = pickle.load(f)



fig = plt.figure()
plt.xlim([-150000, 150000])
plt.ylim([-150000, 150000])
#plt.xlim([np.amin(transformed_windows[:, 0]), np.amax(transformed_windows[:, 0])])
#plt.ylim([np.amin(transformed_windows[:, 1]), np.amax(transformed_windows[:, 1])])
scat = plt.scatter([0], [0], s=10)

p = pyaudio.PyAudio()
cur = 0
pyaudio_t = 0
def callback(in_data, frame_count, time_info, status):
    global cur
    global pyaudio_t
    d = samples[cur:cur+frame_count]
    samples_packed = \
        struct.pack('%dh' % len(d), *d)
    cur += frame_count
    pyaudio_t = cur / rate
    return (samples_packed, pyaudio.paContinue)
    
fps = 20
interval = 1000 / fps
last = None
def update(frame_n):
    global rate
    global cur
    global last
    global pyaudio_t
    global stream
    this_t = frame_n / fps
    delta = pyaudio_t - this_t
    if delta > 0.1:
        stream.stop_stream()
    else:
        stream.start_stream()
    if delta < -0.1:
        cur += 0.1 * rate
    print("animation: %.2f" % (pyaudio_t - this_t))
    t = int(frame_n / fps * rate)
    if last is None:
        last = t
        return []
    sparsity = 16
    prop = 0.0
    last2 = math.floor(last + (t - last) * prop)
    w = windows[last2:t:sparsity]
    transformed_windows = pca.transform(w)
    last = t
    scat.set_offsets(transformed_windows)
    return []

stream = p.open(rate=rate, channels=1, format=p.get_format_from_width(2), output=True,
                stream_callback=callback)
animation = FuncAnimation(fig, update, interval=interval, blit=True)
plt.show()
