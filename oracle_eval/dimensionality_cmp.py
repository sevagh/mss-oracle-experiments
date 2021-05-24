import sys
import os
import musdb
import itertools
import numpy as np
import random
import argparse
from shared import TFTransform, dimensionality_cmp


class TrackEvaluator:
    def __init__(self, tracks):
        self.tracks = tracks

    def dimcmp(self, tfs):
        for track in self.tracks:
            print(f'{track=}')
            dimensionality_cmp(track, tfs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare NSGT and STFT dimensionality'
    )
    parser.add_argument(
        '--window',
        type=int,
        default=4096,
        help='stft window size',
    )
    parser.add_argument(
        '--sllen',
        type=int,
        default=4096,
        help='nsgt slice length',
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

    args = parser.parse_args()

    random.seed(args.random_seed)

    # initiate musdb
    mus = musdb.DB(subsets='train', split='valid', is_wav=True)

    max_tracks = min(int(os.getenv('MUSDB_MAX_TRACKS', sys.maxsize)), len(mus.tracks))

    tracks = None
    if args.n_random_tracks:
        print(f'using {args.n_random_tracks} random tracks from MUSDB18-HQ train set validation split')
        tracks = random.sample(mus.tracks, args.n_random_tracks)
    else:
        print(f'using tracks 0-{max_tracks} from MUSDB18-HQ train set validation split')
        tracks = mus.tracks[:max_tracks]

    # stft
    tf_stft = TFTransform(44100, window=9216)

    # nsgt sliced
    tf_nsgt = TFTransform(44100, transform_type="nsgt", sllen=65536, trlen=16384, fscale="bark", fbins=800, fmin=78.0)

    t = TrackEvaluator(tracks)
    t.dimcmp([tf_stft, tf_nsgt])
