import gc
import museval
import numpy as np
from warnings import warn
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
    def __init__(self, fs, transform_type="stft", window=4096, fscale="bark", fmin=78.0, fbins=125, fgamma=25.0, sllen=32768, trlen=8192):
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

            self.nsgt = NSGT_sliced(scl, sllen, trlen, fs, real=True, matrixform=True, multichannel=True)
            self.name = f'n{fscale}-{fbins}-{fmin:.2f}'
        else:
            self.name = f's{window}'

    def forward(self, audio):
        if not self.nsgt:
            return stft(audio.T, nperseg=self.nperseg, noverlap=self.noverlap)[-1].astype(np.complex64)
        else:
            return np.asarray(list(self.nsgt.forward((audio.T,)))).astype(np.complex64)

    def backward(self, X, len_x):
        if not self.nsgt:
            return istft(X, nperseg=self.nperseg, noverlap=self.noverlap)[1].T.astype(np.float32)
        else:
            return next(reblock(self.nsgt.backward(X), len_x, fulllast=False, multichannel=True)).real.astype(np.float32).T


def ideal_mask(track, tf, alpha=2, binary_mask=False, theta=0.5, eval_dir=None, fast_eval=False):
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

    bss_scores = None
    if not fast_eval:
        bss_scores = museval.eval_mus_track(
            track,
            estimates,
            output_dir=eval_dir,
        )
    else:
        bss_scores = fast_sdr(track, estimates)

    return estimates, bss_scores


def ideal_mixphase(track, tf, eval_dir=None, fast_eval=False):
    """
    ideal performance of magnitude from estimated source + phase of mix
    which is the default umx strategy for separation
    """
    N = track.audio.shape[0]

    X = tf.forward(track.audio)

    #(I, F, T) = X.shape

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

    bss_scores = None
    if not fast_eval:
        bss_scores = museval.eval_mus_track(
            track,
            estimates,
            output_dir=eval_dir,
        )
    else:
        bss_scores = fast_sdr(track, estimates)

    return estimates, bss_scores


'''
from https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021
    nb_sources, nb_samples, nb_channels = 4, 100000, 2
    references = np.random.rand(nb_sources, nb_samples, nb_channels)
    estimates = np.random.rand(nb_sources, nb_samples, nb_channels)
'''
def fast_sdr(track, estimates_dct):
    references = np.asarray([source.audio for source in track.sources.values()])
    estimates = np.asarray([est for est_name, est in estimates_dct.items() if est_name != 'accompaniment'])
    # compute SDR for one song
    num = np.sum(np.square(references), axis=(1, 2)) + eps
    den = np.sum(np.square(references - estimates), axis=(1, 2)) + eps
    sdr_instr = 10.0 * np.log10(num  / den)
    sdr_song = np.mean(sdr_instr)
    return sdr_song
