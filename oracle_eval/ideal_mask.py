import sys
import os
import musdb
import gc
import itertools
import museval
import numpy as np
import functools
import argparse
import tqdm

import scipy
from scipy.signal import stft, istft

import json

from shared import TFTransform
from oracle import ideal_mask, ideal_mixphase, ideal_mask_fbin


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate Ideal Ratio Mask'
    )
    parser.add_argument(
        '--audio_dir',
        nargs='?',
        help='Folder where audio results are saved',
        default=None,
    )

    parser.add_argument(
        '--eval_dir',
        nargs='?',
        help='Folder where evaluation results are saved'
    )
    parser.add_argument(
        'config_file',
        help='json file with time-frequency (stft, cqt) evaluation configs',
    )

    args = parser.parse_args()

    max_tracks = int(os.getenv('MUSDB_MAX_TRACKS', sys.maxsize))

    # initiate musdb with validation tracks
    mus = musdb.DB(subsets='train', split='valid', is_wav=True)

    # accumulate all time-frequency configs to compare
    tfs = []

    with open(args.config_file) as jf:
        config = json.load(jf)
        tmp = None
        for stft_win in config.get('stft_configs', {}).get('window_sizes', []):
            tfs.append(
                TFTransform(
                    44100,
                    transform_type="stft",
                    window=stft_win
                )
            )
        for nsgt_conf in config.get('nsgt_configs', []):
            tfs.append(
                TFTransform(
                    44100,
                    transform_type="nsgt",
                    fscale=nsgt_conf['scale'],
                    fmin=nsgt_conf['fmin'],
                    fbins=nsgt_conf['bins'],
                    fgamma=nsgt_conf.get('gamma', 0.0),
                    sllen=nsgt_conf.get('sllen', 32768),
                    trlen=nsgt_conf.get('trlen', 8192),
                )
            )

    masks = [
            {'power': 1, 'binary': False},
            {'power': 1, 'binary': False, 'phasemix': True},
            #{'power': 1, 'binary': False, 'fbin': True},
            #{'power': 2, 'binary': False}, #-- why test these since no NNs actually use them irl
            #{'power': 1, 'binary': True},
            #{'power': 2, 'binary': True},
    ]

    mss_evaluations = list(itertools.product(mus.tracks[:max_tracks], tfs, masks))

    for (track, tf, mask) in tqdm.tqdm(mss_evaluations):
        N = track.audio.shape[0]  # remember number of samples for future use
        # construct mask name e.g. irm1, ibm2
        mask_name = 'i'
        if mask['binary']:
            mask_name += 'b'
        else:
            mask_name += 'r'
        mask_name += f"m{str(mask['power'])}"

        if mask.get('phasemix', False):
            mask_name = 'mpi'

        if mask.get('fbin', False):
            mask_name = 'fbin'

        name = mask_name
        if tf.name != '':
            name += f'-{tf.name}'

        est = None

        est_path = os.path.join(args.eval_dir, f'{name}') if args.eval_dir else None
        aud_path = os.path.join(args.audio_dir, f'{name}') if args.audio_dir else None

        if not mask.get('phasemix', False) and not mask.get('fbin', False):
            # ideal mask
            est, _ = ideal_mask(
                track,
                tf,
                mask['power'],
                mask['binary'],
                0.5,
                eval_dir=est_path)
        elif mask.get('phasemix', False):
            est, _ = ideal_mixphase(
                track,
                tf,
                eval_dir=est_path,
                dur=7.4,
                start=46.5
                )
        elif mask.get('fbin', False):
            est, _ = ideal_mask_fbin(
                track,
                tf,
                eval_dir=est_path)

        gc.collect()

        if args.audio_dir:
            mus.save_estimates(est, track, aud_path)
