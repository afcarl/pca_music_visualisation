from __future__ import print_function, division
import numpy as np
import wave
import matplotlib.pyplot as plt
import struct
from sklearn.decomposition import  PCA

volume = 1     # range [0.0, 1.0]
fs = 44100       # sampling rate, Hz, must be integer
duration = 1.0   # in seconds, may be float
f = 440.0        # sine frequency, Hz, may be float

samples = []
for f in [220, 440, 220, 440, 220, 440, 220, 440]:
    # generate samples, note conversion to float32 array
    samples1 = (np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)
    samples.extend(samples1)
print(len(samples))

samples = np.array(samples)
window_len_samples = 128
# stride argument: stride for each dimension of the resulting array
# this is set up for stride=1 right now
windows = np.lib.stride_tricks.as_strided(samples,
        shape=(len(samples)-window_len_samples+1,window_len_samples),
        strides=(samples.itemsize, samples.itemsize))

print(len(windows))
pca = PCA(n_components=2)
pca.fit(windows)
transformed_windows = pca.transform(windows)
plt.scatter(transformed_windows[:, 0], transformed_windows[:, 1])
plt.show()

plt.plot(samples)
#plt.show()

# samples = [int(32767 * x) for x in samples]
# sample_words = \
#     struct.pack('%dh' % len(samples), *samples)
# w = wave.open('/tmp/foo.wav', 'wb')
# w.setsampwidth(2)
# w.setnchannels(1)
# w.setframerate(44100)
# w.writeframes(sample_words)