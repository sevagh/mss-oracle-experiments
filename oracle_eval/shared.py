import gc
import museval
import numpy as np
try:
    import cupy
except ImportError:
    cupy = None

import librosa

import scipy
from scipy.signal import stft, istft

# use CQT based on nonstationary gabor transform
from nsgt import NSGT_sliced, MelScale, LogScale, BarkScale, VQLogScale
from nsgt.reblock import reblock

# small epsilon to avoid dividing by zero
eps = np.finfo(np.float32).eps


class TFTransform:
    def __init__(self, fs, transform_type="stft", window=4096, fscale="cqlog", fmin=20.0, fbins=48, fgamma=25.0):
        use_nsgt = (transform_type == "nsgt")

        self.nperseg = window
        self.noverlap = self.nperseg // 4

        self.nsgt = None
        self.nsgt = None

        if use_nsgt:
            scl = None
            if fscale == 'mel':
                scl = MelScale(fmin, fs/2, fbins)
            elif fscale == 'bark':
                scl = BarkScale(fmin, fs/2, fbins)
            elif fscale == 'cqlog':
                scl = LogScale(fmin, fs/2, fbins)
            elif fscale == 'vqlog':
                scl = VQLogScale(fmin, fs/2, fbins, gamma=fgamma)
            else:
                raise ValueError(f"unsupported scale {fscale}")

            # use slice length required to support desired frequency scale/q factors
            sllen = scl.suggested_sllen(fs)
            trlen = sllen//4
            trlen = trlen + -trlen % 2 # make trlen divisible by 2

            print(f'{sllen=}, {trlen=}')
            self.nsgt = NSGT_sliced(scl, sllen, trlen, fs, real=True, matrixform=True, multichannel=True)
            self.name = fscale
        else:
            self.name = f's{window}'

    def forward(self, audio):
        if not self.nsgt:
            return stft(audio.T, nperseg=self.nperseg, noverlap=self.noverlap)[-1].astype(np.complex64)
        else:
            print('slicq forward!')
            return np.asarray(list(self.nsgt.forward((audio.T,)))).astype(np.complex64)

    def backward(self, X, len_x):
        print(len_x)
        if not self.nsgt:
            return istft(X, nperseg=self.nperseg, noverlap=self.noverlap)[1].T.astype(np.float32)
        else:
            print('slicq backward!')
            return next(reblock(self.nsgt.backward(X), len_x, fulllast=False, multichannel=True)).real.astype(np.float32).T


def ideal_mask(track, tf, alpha=2, binary_mask=False, theta=0.5, eval_dir=None):
    """
    if theta=None:
        Ideal Ratio Mask:
        processing all channels inpependently with the ideal ratio mask.
        this is the ratio of spectrograms, where alpha is the exponent to take for
        spectrograms. usual values are 1 (magnitude) and 2 (power)

    if theta=float:
        Ideal Binary Mask:
        processing all channels inpependently with the ideal binary mask.

        the mix is send to some source if the spectrogram of that source over that
        of the mix is greater than theta, when the spectrograms are take as
        magnitude of STFT raised to the power alpha. Typical parameters involve a
        ratio of magnitudes (alpha=1) and a majority vote (theta = 0.5)
    """

    N = track.audio.shape[0]

    X = tf.forward(track.audio)

    print(X.shape)

    #(I, F, T) = X.shape

    # soft mask stuff
    if not binary_mask:
        # Compute sources spectrograms
        P = {}
        # compute model as the sum of spectrograms
        model = eps

        # parallelize this
        for name, source in track.sources.items():
            # compute spectrogram of target source:
            # magnitude of STFT to the power alpha
            P[name] = np.abs(tf.forward(source.audio))**alpha
            model += P[name]

    # now performs separation
    estimates = {}
    accompaniment_source = 0
    for name, source in track.sources.items():
        if binary_mask:
            # compute STFT of target source
            Yj = tf.forward(source.audio)

            # Create Binary Mask
            Mask = np.divide(np.abs(Yj)**alpha, (eps + np.abs(X)**alpha))
            Mask[np.where(Mask >= theta)] = 1
            Mask[np.where(Mask < theta)] = 0
        else:
            # compute soft mask as the ratio between source spectrogram and total
            Mask = np.divide(np.abs(P[name]), model)

        # multiply the mix by the mask
        Yj = np.multiply(X, Mask)

        print(Yj.shape)

        # invert to time domain
        target_estimate = tf.backward(Yj, N)

        # set this as the source estimate
        estimates[name] = target_estimate

        # accumulate to the accompaniment if this is not vocals
        if name != 'vocals':
            accompaniment_source += target_estimate

    estimates['accompaniment'] = accompaniment_source

    gc.collect()

    if cupy:
        # cupy disable fft caching to free blocks
        fft_cache = cupy.fft.config.get_plan_cache()
        fft_cache.set_size(0)

        cupy.get_default_memory_pool().free_all_blocks()

        # cupy reenable fft caching
        fft_cache.set_size(16)
        fft_cache.set_memsize(-1)

    bss_scores = museval.eval_mus_track(
        track,
        estimates,
        output_dir=eval_dir,
    )

    return estimates, bss_scores


def ideal_mixphase(track, tf, eval_dir=None):
    """
    ideal performance of magnitude from estimated source + phase of mix
    which is the default umx strategy for separation
    """
    N = track.audio.shape[0]

    X = tf.forward(track.audio)

    #(I, F, T) = X.shape
    print(X.shape)

    # Compute sources spectrograms
    P = {}
    # compute model as the sum of spectrograms
    model = eps

    # parallelize this
    for name, source in track.sources.items():
        # compute spectrogram of target source:
        # magnitude of STFT
        src_coef = tf.forward(source.audio)

        P[name] = np.abs(src_coef)

        # store the original, not magnitude, in the mix
        model += src_coef

    # now performs separation
    estimates = {}
    accompaniment_source = 0
    for name, source in track.sources.items():
        source_mag = P[name]

        _, mix_phase = librosa.magphase(model)
        Yj = source_mag * mix_phase

        # invert to time domain
        target_estimate = tf.backward(Yj, N)

        # set this as the source estimate
        estimates[name] = target_estimate

        # accumulate to the accompaniment if this is not vocals
        if name != 'vocals':
            accompaniment_source += target_estimate

    estimates['accompaniment'] = accompaniment_source

    gc.collect()

    if cupy:
        # cupy disable fft caching to free blocks
        fft_cache = cupy.fft.config.get_plan_cache()
        fft_cache.set_size(0)

        cupy.get_default_memory_pool().free_all_blocks()

        # cupy reenable fft caching
        fft_cache.set_size(16)
        fft_cache.set_memsize(-1)

    bss_scores = museval.eval_mus_track(
        track,
        estimates,
        output_dir=eval_dir,
    )

    return estimates, bss_scores
