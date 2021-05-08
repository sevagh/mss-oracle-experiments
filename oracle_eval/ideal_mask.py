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

# use CQT based on nonstationary gabor transform
from nsgt import NSGT, MelScale, LogScale, BarkScale, VQLogScale

import json
from types import SimpleNamespace

from shared import ideal_mask, ideal_mixphase, TFTransform


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

    # initiate musdb
    mus = musdb.DB(subsets='test', is_wav=True)

    # accumulate all time-frequency configs to compare
    tfs = []

    with open(args.config_file) as jf:
        config = json.load(jf)
        tmp = None
        for stft_win in config.get('stft_configs', {}).get('window_sizes', []):
            tmp = {'type': 'stft', 'window': stft_win}
            tmp['name'] = f's{str(stft_win)}'

            tf_transform = SimpleNamespace(**tmp)
            tfs.append(tf_transform)
        for nsgt_conf in config.get('nsgt_configs', []):
            tmp = {'type': 'nsgt', 'scale': nsgt_conf['scale'], 'fmin': nsgt_conf['fmin'], 'fmax': 22050, 'bins': nsgt_conf['bins']}
            fmin_str = f"{nsgt_conf['fmin']}"
            fmin_str = fmin_str.replace('.', '')
            tmp['name'] = f"{nsgt_conf['scale']}"

            if nsgt_conf.get('gamma', None):
                tmp['gamma'] = nsgt_conf['gamma']

            tf_transform = SimpleNamespace(**tmp)
            tfs.append(tf_transform)

    masks = [
            {'power': 1, 'binary': False},
            {'power': 2, 'binary': False},
            {'power': 1, 'binary': True},
            {'power': 2, 'binary': True},
            {'power': 1, 'binary': False, 'phasemix': True},
    ]

    mss_evaluations = list(itertools.product(mus.tracks[:max_tracks], tfs, masks))

    for (track, tf_transform, mask) in tqdm.tqdm(mss_evaluations):
        N = track.audio.shape[0]  # remember number of samples for future use
        tf = TFTransform(N, track.rate, tf_transform)

        # construct mask name e.g. irm1, ibm2
        mask_name = 'i'
        if mask['binary']:
            mask_name += 'b'
        else:
            mask_name += 'r'
        mask_name += f"m{str(mask['power'])}"

        if mask.get('phasemix', False):
            mask_name = 'mpi'

        name = mask_name
        if tf_transform.name != '':
            name += f'-{tf_transform.name}'

        est = None
        if not mask.get('phasemix', False):
            # ideal mask
            est, _ = ideal_mask(
                track,
                tf,
                mask['power'],
                mask['binary'],
                0.5,
                os.path.join(args.eval_dir, f'{name}'))
        else:
            print('doing phasemix!')
            est, _ = ideal_mixphase(
                track,
                tf,
                os.path.join(args.eval_dir, f'{name}'))

        gc.collect()

        if args.audio_dir:
            mus.save_estimates(est, track, os.path.join(args.eval_dir, f'{name}'))
