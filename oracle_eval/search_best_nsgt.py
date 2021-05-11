import sys
import os
import musdb
import itertools
import museval
from functools import partial
import numpy as np
import random
import argparse
from bayes_opt import BayesianOptimization
from bayes_opt.util import load_logs
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

from shared import ideal_mask, ideal_mixphase, TFTransform

import scipy
from scipy.signal import stft, istft

import json
from types import SimpleNamespace


class TrackEvaluator:
    def __init__(self, tracks, oracle='irm1'):
        self.tracks = tracks
        self.oracle_func = None
        if oracle == 'irm1':
            self.oracle_func = partial(ideal_mask, binary_mask=False, alpha=1)
        elif oracle == 'irm2':
            self.oracle_func = partial(ideal_mask, binary_mask=False, alpha=2)
        elif oracle == 'ibm1':
            self.oracle_func = partial(ideal_mask, binary_mask=True, alpha=1)
        elif oracle == 'ibm2':
            self.oracle_func = partial(ideal_mask, binary_mask=True, alpha=2)
        elif oracle == 'mpi':
            self.oracle_func = partial(ideal_mixphase)
        else:
            raise ValueError(f'unsupported oracle {oracle}')

    def eval_control(self, window_size=4096):
        return self.oracle(control=True, stft_window=window_size)

    def eval_vqlog(self, fmin=20.0, bins=12, gamma=25):
        return self.oracle(scale='vqlog', fmin=fmin, bins=bins, gamma=gamma, control=False)

    def eval_cqlog(self, fmin=20.0, bins=12):
        return self.oracle(scale='cqlog', fmin=fmin, bins=bins, control=False)

    def eval_mel(self, fmin=20.0, bins=12):
        return self.oracle(scale='mel', fmin=fmin, bins=bins, control=False)

    def eval_bark(self, fmin=20.0, bins=12):
        return self.oracle(scale='bark', fmin=fmin, bins=bins, control=False)

    def oracle(self, scale='cqlog', fmin=20.0, bins=12, gamma=25, control=False, stft_window=4096):
        bins = int(bins)

        med_sdrs = []

        transform_type = "nsgt"
        if control:
            transform_type = "stft"

        tf = TFTransform(track.rate, transform_type, stft_window, scale, fmin, bins, gamma, slicq_sllen, slicq_trlen)

        for track in self.tracks:
            #print(f'track: {track.name}')
            N = track.audio.shape[0]

            _, bss_scores_obj = self.oracle_func(track, tf, eval_dir=None)
            print(bss_scores_obj)
            bss_scores = bss_scores_obj.scores

            scores = np.zeros((4, 1), dtype=np.float32)
            for target_idx, t in enumerate(bss_scores['targets']):
                if t['name'] == 'accompaniment':
                    continue
                for metric_idx, metric in enumerate(['SDR']):
                    agg = np.nanmedian([np.float32(f['metrics'][metric]) for f in t['frames']])
                    scores[target_idx, metric_idx] = agg

            median_sdr = np.median(scores)
            med_sdrs.append(median_sdr)

        return np.median(med_sdrs)


def optimize(f, bounds, name, n_iter, n_random, logdir=None):
    optimizer = BayesianOptimization(
        f=f,
        pbounds=bounds,
        verbose=2,
        random_state=1,
    )
    if logdir:
        logpath = os.path.join(logdir, f"./{name}_logs.json")
        try:
            load_logs(optimizer, logs=[logpath])
            print('loaded previous log')
        except FileNotFoundError:
            print('no log found, re-optimizing')
            pass

        logger = JSONLogger(path=logpath)
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    print(f'optimizing {name} scale')
    optimizer.maximize(
        init_points=n_random,
        n_iter=n_iter,
    )
    print(f'max {name}: {optimizer.max}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Search NSGT configs for best ideal mask'
    )
    parser.add_argument(
        '--control',
        action='store_true',
        help='evaluate control (stft)'
    )
    parser.add_argument(
        '--control-window-sizes',
        type=str,
        default='1024,2048,4096,8192,16384',
        help='comma-separated window sizes of stft to evaluate'
    )
    parser.add_argument(
        '--oracle',
        type=str,
        default='irm1',
        help='type of oracle to compute (choices: irm1, irm2, ibm1, ibm2, mpi)'
    )
    parser.add_argument(
        '--n-random-tracks',
        type=int,
        default=None,
        help='use N random tracks instead of MUSDB_MAX_TRACKS'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='rng seed to pick the same random 5 songs'
    )
    parser.add_argument(
        '--optimization-iter',
        type=int,
        default=2,
        help='bayesian optimization iterations',
    )
    parser.add_argument(
        '--optimization-random',
        type=int,
        default=1,
        help='bayesian optimization random iterations',
    )
    parser.add_argument(
        '--logdir',
        default=None,
        type=str,
        help='directory to store optimization logs',
    )

    args = parser.parse_args()

    random.seed(args.random_seed)

    # initiate musdb
    mus = musdb.DB(subsets='test', is_wav=True)

    max_tracks = min(int(os.getenv('MUSDB_MAX_TRACKS', sys.maxsize)), len(mus.tracks))

    tracks = None
    if args.n_random_tracks:
        print(f'using {args.n_random_tracks} random tracks from MUSDB18-HQ test set')
        tracks = random.sample(mus.tracks, args.n_random_tracks)
    else:
        print(f'using tracks 0-{max_tracks} from MUSDB18-HQ test set')
        tracks = mus.tracks[:max_tracks]

    t = TrackEvaluator(tracks, oracle=args.oracle)

    bins = (12,348)
    fmins = (15.0,60.0)
    gammas = (0.0,100.0)

    pbounds_vqlog = {
        'bins': bins,
        'fmin': fmins,
        'gamma': gammas,
    }

    pbounds_other = {
        'bins': bins,
        'fmin': fmins
    }

    print('oracle: {0}'.format(args.oracle))

    if args.control:
        for window_size in [int(x) for x in args.control_window_sizes.split(',')]:
            print(f'evaluating control stft {window_size}')
            print('median SDR: {0}'.format(t.eval_control(window_size=window_size)))
        sys.exit(0)

    optimize(t.eval_vqlog, pbounds_vqlog, "vqlog", args.optimization_iter, args.optimization_random, logdir=args.logdir)
    optimize(t.eval_cqlog, pbounds_other, "cqlog", args.optimization_iter, args.optimization_random, logdir=args.logdir)
    optimize(t.eval_mel, pbounds_other, "mel", args.optimization_iter, args.optimization_random, logdir=args.logdir)
    optimize(t.eval_bark, pbounds_other, "bark", args.optimization_iter, args.optimization_random, logdir=args.logdir)
