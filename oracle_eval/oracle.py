import gc
import museval
import numpy as np
import random
from sklearn.decomposition import TruncatedSVD
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

from shared import fast_sdr


def ideal_mask_mixphase_per_coef(track, tf, eval_dir=None, dur=None, start=None, fast_eval=None):
    if dur:
        track.chunk_duration = dur
    if start:
        track.chunk_start = start

    _, bss_scores1 = ideal_mask(track, tf, binary_mask=False, alpha=1, fast_eval=True)
    _, bss_scores2 = ideal_mixphase(track, tf, fast_eval=True)

    X = tf.forward(track.audio)

    #(I, F, T) = X.shape
    coef = int(tf.nsgt.coef_factor*tf.sllen)*tf.fbins+1

    # return score divided by coefficient count
    ret = (bss_scores1+bss_scores2)/coef
    return None, ret



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


def ideal_mixphase(track, tf, eval_dir=None, fast_eval=False, dur=None, start=None):
    """
    ideal performance of magnitude from estimated source + phase of mix
    which is the default umx strategy for separation
    """
    if dur:
        track.chunk_duration = dur
    if start:
        track.chunk_start = start

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

        #print('{0} {1}: {2}'.format(tf.name, name, P[name]))
        #input()

        # store the original, not magnitude, in the mix
        model += src_coef

    # now performs separation
    estimates = {}
    accompaniment_source = 0
    for name, source in track.sources.items():
        source_mag = P[name]
        #print(source_mag)

        print('inverting phase')
        _, mix_phase = torch.tensor(librosa.magphase(model.cpu().numpy()))

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

    print('computing bss')
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


def ideal_mask_fbin(track, tf, dur=None, start=None, eval_dir=None, fast_eval=False, mbin=False):
    if dur:
        track.chunk_duration = dur
    if start:
        track.chunk_start = start

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
        # magnitude of STFT to the power alpha

        nsgt = tf.forward(source.audio)
        print('nsgt shape: {0}'.format(nsgt.shape))

        if mbin:
            nsgt = torch.linalg.norm(nsgt, dim=-2, keepdim=False, ord=3)
        elif svd:
            nsgt = torch.linalg.svd(nsgt, dim=-1, keepdim=False)
            print('after svd: {0}'.format(nsgt.shape))
        else:
            nsgt = torch.linalg.norm(nsgt, dim=-1, keepdim=False, ord=3)

        P[name] = torch.abs(nsgt)
        model += P[name]

    # now performs separation
    estimates = {}
    accompaniment_source = 0
    for name, source in track.sources.items():
        # compute soft mask as the ratio between source spectrogram and total
        Mask = torch.divide(torch.abs(P[name]), model)

        # multiply the mix by the mask across f bins

        if mbin:
            Yj = torch.multiply(X, Mask[..., None, :])
        else:
            Yj = torch.multiply(X, Mask[..., None])

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


def slicq_svd(track, tf, dur=None, start=None, eval_dir=None, fast_eval=False):
    if dur:
        track.chunk_duration = dur
    if start:
        track.chunk_start = start

    N = track.audio.shape[0]

    X = tf.forward(track.audio)

    svd = TruncatedSVD(n_components=50, n_iter=7, random_state=42)
    #X = svd.fit_transform(X)

    #(I, F, T) = X.shape

    # Compute sources spectrograms
    P = {}
    # compute model as the sum of spectrograms
    model = eps

    # parallelize this
    for name, source in track.sources.items():
        # compute spectrogram of target source:
        # magnitude of STFT to the power alpha

        nsgt = tf.forward(source.audio)
        print('PRE-SVD nsgt shape: {0}'.format(nsgt.shape))

        # svd across the last two slicq channels, f x m
        for frame in range(nsgt.shape[0]):
            for chan in range(nsgt.shape[1]):
                tmp = torch.tensor(svd.fit_transform(torch.abs(nsgt[frame, chan, ...])), device="cpu")
                print('tmp.shape: {0}'.format(tmp.shape))

        print('POST-SVD nsgt shape: {0}'.format(nsgt.shape))

        P[name] = nsgt
        model += P[name]

    # now performs separation
    estimates = {}
    accompaniment_source = 0
    for name, source in track.sources.items():
        # compute soft mask as the ratio between source spectrogram and total
        Mask = torch.divide(
            svd.inverse_transform(P[name]),
            svd.inverse_transform(model),
        )

        # multiply the mix by the mask across f bins

        Yj = torch.multiply(X, Mask)

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
