import gc
import numpy as np
import argparse
import sys
import itertools
import librosa
import scipy
from scipy.signal import stft, istft

# use CQT based on nonstationary gabor transform
from nsgt import NSGT_sliced, MelScale, LogScale, BarkScale, VQLogScale


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='check slice length support for nsgt config'
    )
    parser.add_argument(
        '--control',
        action='store_true',
        help='evaluate control (stft)'
    )
    parser.add_argument(
        '--bins',
        type=str,
        default='12,2000,10',
        help='comma-separated range of bins to evaluate, step is last element'
    )
    parser.add_argument(
        '--fmins',
        type=str,
        default='10,130,5',
        help='comma-separated range of fmin to evaluate, step is last element'
    )
    parser.add_argument(
        '--fmaxes',
        type=str,
        default='14000,22050,5',
        help='comma-separated range of fmax to evaluate, step is last element'
    )
    parser.add_argument(
        '--gammas',
        type=str,
        default='0,100',
        help='comma-separated range of gamma to evaluate'
    )
    parser.add_argument(
        '--fscale',
        type=str,
        default='vqlog',
        help='nsgt frequency scale (choices: vqlog, cqlog, mel, bark)'
    )
    parser.add_argument(
        '--sllen-test',
        type=int,
        default=16384*2,
        help='sllen to test'
    )

    args = parser.parse_args()

    fs = 44100

    sldur = args.sllen_test/fs

    bmin, bmax, bstep = [int(x) for x in args.bins.split(',')]
    fmin, fmax, fstep = [float(x) for x in args.fmins.split(',')]
    fmaxmin, fmaxmax, fmaxstep = [float(x) for x in args.fmaxes.split(',')]

    bins = np.arange(bmin,bmax,bstep)
    fmins = np.arange(fmin,fmax,fstep)
    fmaxes = np.arange(fmaxmin,fmaxmax,fmaxstep)

    for (fmin, fmax) in itertools.product(fmins, fmaxes):
        for fbins in bins:
            scl = None
            if args.fscale == 'mel':
                scl = MelScale(fmin, fs/2, fbins)
            elif args.fscale == 'bark':
                scl = BarkScale(fmin, fs/2, fbins)
            elif args.fscale == 'vqlog':
                scl = VQLogScale(fmin, fs/2, fbins, gamma=25)
            elif args.fscale == 'cqlog':
                scl = LogScale(fmin, fs/2, fbins)

            # use slice length required to support desired frequency scale/q factors
            sllen_suggested = scl.suggested_sllen(fs)

            if sllen_suggested > args.sllen_test:
                print(f"testing nsgt param combination:\n\t{args.fscale=} {fbins=} {fmin=} {fmax=}")
                print(f'sllen too big to be supported by slice duration: {sldur:.2f} s')
                break
