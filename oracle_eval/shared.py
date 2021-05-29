import gc
import museval
import numpy as np
import random
from warnings import warn
try:
    import cupy
except ImportError:
    cupy = None

import torch
import librosa

import scipy
from scipy.signal import stft, istft

# use CQT based on nonstationary gabor transform
from nsgt import NSGT_sliced, MelScale, LogScale, BarkScale, VQLogScale
from nsgt.reblock import reblock

# small epsilon to avoid dividing by zero
eps = np.finfo(np.float32).eps


class TFTransform:
    def __init__(self, fs, transform_type="stft", window=4096, fscale="bark", fmin=78.0, fmax=None, fbins=125, fgamma=25.0, sllen=None, trlen=None):
        self.transform_type = transform_type
        use_nsgt = (transform_type == "nsgt")

        self.nperseg = window
        self.noverlap = self.nperseg // 4
        self.fbins = fbins

        self.nsgt = None

        if not fmax:
            fmax = fs/2

        if use_nsgt:
            scl = None
            if fscale == 'mel':
                scl = MelScale(fmin, fmax, fbins)
            elif fscale == 'bark':
                scl = BarkScale(fmin, fmax, fbins)
            elif fscale == 'cqlog':
                scl = LogScale(fmin, fmax, fbins)
            elif fscale == 'vqlog':
                scl = VQLogScale(fmin, fmax, fbins, gamma=fgamma)
            else:
                raise ValueError(f"unsupported scale {fscale}")

            if sllen is None or trlen is None:
                # use slice length required to support desired frequency scale/q factors
                sllen = scl.suggested_sllen(fs)
                trlen = sllen//4
                trlen = trlen + -trlen % 2 # make trlen divisible by 2

                print(f'auto-selected {sllen=}, {trlen=}')
            else:
                print(f'using supplied {sllen=}, {trlen=}')

            self.sllen = sllen
            self.nsgt = NSGT_sliced(scl, sllen, trlen, fs, real=True, matrixform=True, multichannel=True, reducedform=True, device="cpu")
            self.name = f'n{fscale}-{fbins}-{fmin:.2f}-{sllen}'
        else:
            self.name = f's{window}'

    def forward(self, audio):
        if not self.nsgt:
            return torch.tensor(stft(audio.T, nperseg=self.nperseg, noverlap=self.noverlap)[-1].astype(np.complex64), device="cpu")
        else:
            return self.nsgt.forward((torch.tensor(audio.T.copy(), device="cpu"), ))

    def backward(self, X, len_x):
        if not self.nsgt:
            return torch.tensor(istft(X, nperseg=self.nperseg, noverlap=self.noverlap)[1].T.astype(np.float32), device="cpu")
        else:
            return self.nsgt.backward(X, len_x).T


'''
from https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021
    nb_sources, nb_samples, nb_channels = 4, 100000, 2
    references = np.random.rand(nb_sources, nb_samples, nb_channels)
    estimates = np.random.rand(nb_sources, nb_samples, nb_channels)
'''
def fast_sdr(track, estimates_dct):
    references = torch.cat([torch.unsqueeze(torch.tensor(source.audio, device="cpu"), dim=0) for source in track.sources.values()])
    estimates = torch.cat([torch.unsqueeze(est, dim=0) for est_name, est in estimates_dct.items() if est_name != 'accompaniment'])

    # compute SDR for one song
    num = torch.sum(torch.square(references), dim=(1, 2)) + eps
    den = torch.sum(torch.square(references - estimates), dim=(1, 2)) + eps
    sdr_instr = 10.0 * torch.log10(num / den)
    sdr_song = torch.mean(sdr_instr)
    return sdr_song


def dimensionality_cmp(track, tfs):
    N = track.audio.shape[0]
    track_dur = N/track.rate

    for tf in tfs:
        print(f'track duration: {N} samples, {track_dur:.2f} s')
        if tf.transform_type == "stft":
            track_multiples = N//(tf.nperseg-tf.nperseg//4) + 2
            print(f'track in multiples of stft window size {tf.nperseg}: {track_multiples}')
            print(f'expected f bins: {tf.nperseg//2+1}')
        elif tf.transform_type == "nsgt":
            track_multiples = N//(tf.sllen-tf.sllen//4) + 2
            print(f'nsgt coef factor: {tf.nsgt.coef_factor}')
            print(f'total ncoefs: {tf.nsgt.coef_factor*tf.sllen}')
            print(f'track in multiples of nsgt slice size {tf.sllen}: {track_multiples}')
            print(f'expected f bins: {tf.fbins+1}')

            track_real_multiples = N//X.shape[0]
            print(f'nsgt actual track multiples: {track_real_multiples}')

        X = tf.forward(track.audio)
        print(f'forward transform: {X.dtype=}, {X.shape=}')
        Xmag = np.abs(X)
        print(f'abs forward transform: {Xmag.dtype=}, {Xmag.shape=}')
        #(I, F, T) = X.shape
        print('\n')
