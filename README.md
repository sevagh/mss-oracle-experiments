**:warning: PROJECT ARCHIVAL NOTICE 2021-05-29 :warning:**

These scripts and findings will be released as part of a bigger project soon.

# mss-oracle-experiments

This repo contains experiments in music source separation, mostly related to oracle mask performance.

These 3 ideas are explored:
* Estimating the oracle performance of the "source estimate magnitude spectrogram" + "mix phase" strategy, which is sometimes used in place of soft/hard masking - discussion for Open-Unmix [here](https://github.com/sigsep/open-unmix-pytorch/issues/83)
* How different window sizes of the STFT affect separation BSS scores in the oracles
* Whether the [NSGT](https://github.com/grrrr/nsgt) (Nonstationary Gabor Transform) can achieve good results in music source separation compared to the STFT

This project follows the general practices of oracle mask estimation from [SigSep](https://github.com/sigsep/) and their tools ecosystem, including [musdb](https://github.com/sigsep/sigsep-mus-db) to interact with the [MUSDB18-HQ dataset](https://sigsep.github.io/datasets/musdb.html) and [museval](https://github.com/sigsep/sigsep-mus-eval) to compute BSS scores. The source code of this project derives from the oracle code of [sigsep-mus-oracle](https://github.com/sigsep/sigsep-mus-oracle/) and the evaluation + boxplot code from [sigsep-mus-2018-analysis](https://github.com/sigsep/sigsep-mus-2018-analysis) to generate results similar to the [SiSec 2018 evaluation paper](https://arxiv.org/abs/1804.06267). MUSDB18-HQ is downloadable from [Zenodo](https://zenodo.org/record/3338373).

## Install

* To run on host CPU + memory, use the `requirements.txt` file, which installs most of the dependencies from the latest pip version, except for nsgt, which installs from the [master branch of my fork](https://github.com/sevagh/nsgt) which contains two additional frequency scales, BarkScale and VQLogScale. **NB for CPU!** Install `numpy` first in the Conda env, or virtualenv, since the setup.py of nsgt depends on numpy.
* To perform faster NSGTs + BSS on NVIDIA GPUs, use the `requirements-cupy.txt` file to install from the CuPy-accelerated branches of my forks of [museval](https://github.com/sevagh/sigsep-mus-eval) and [nsgt](https://github.com/sevagh/nsgt/tree/feat/cupy-accel). This drops BSS evaluation time from ~3:20 minutes per track, to ~1 minute. **NB for GPU!** Follow the [CuPy install instructions](https://github.com/cupy/cupy#installation) with your appropriate OS and CUDA version - you may need to adjust the `cupy-cuda112` line in `requirements-cupy.txt`.

Example with virtualenv + Python 3.8 (Fedora 32) on CPU, no NVIDIA GPU code:

```
sevagh:mss-oracle-experiments $ virtualenv --python=python3.8 ~/venvs/mss
sevagh:mss-oracle-experiments $ source ~/venvs/mss/bin/active
(mss) sevagh:mss-oracle-experiments $ pip install numpy
Collecting numpy
  Using cached numpy-1.20.2-cp38-cp38-manylinux2010_x86_64.whl (15.4 MB)
Installing collected packages: numpy
Successfully installed numpy-1.20.2
(mss) sevagh:mss-oracle-experiments $
(mss) sevagh:mss-oracle-experiments $ pip install -r ./requirements.txt
Collecting nsgt
  Cloning git://github.com/sevagh/nsgt to /tmp/pip-install-s27kffaa/nsgt_a000bf1c71fb4ec3b60681c4308f33bb
  Running command git clone -q git://github.com/sevagh/nsgt /tmp/pip-install-s27kffaa/nsgt_a000bf1c71fb4ec3b60681c4308f33bb
...
```

## Run

There are a variety of scripts in oracle_eval, each of which runs a different experiment.

### Oracle performance of "MPI" (mix phase inversion)

Open-Unmix by default uses the estimated source magnitude spectrogram + phase of the original mix to pass to `istft` and get back the waveform of the estimated source.

Let's create a new type of oracle, called "Mix Phase Inversion" to refer to and benchmark the above strategy. The theoretical best possible performance of the MPI oracle is when we have access to the ground truth training signals, which is standard when computing oracles.

The pseudocode of the MPI oracle is:
```python
mix = <load mix>                          # mixed track
vocals_gt = <load vocals stem>   # ground truth

mix_phase = phase(stft(mix))
vocals_gt_magnitude = abs(stft(vocals_gt))

vocals_stft = pol2cart(vocals_gt_magnitude, mix_phase)

vocals_est = istft(vocals_stft)  # estimate after "round trip" through phase inversion
```

Compared to the pseudocode of the IRM1 (ideal ratio mask with magnitude spectrogram, i.e. magnitude raised to the 1th power):

```python
mix = <load mix>                          # mixed track
vocals_gt = <load vocals stem>   # ground truth

vocals_irm1 = abs(stft(vocals_gt)) / abs(stft(mix))

vocals_est = istft(vocals_irm1 * stft(mix)) # estimate after "round trip" through soft mask
```

The source code for the MPI oracle is in [oracle_eval/shared.py](https://github.com/sevagh/mss-oracle-experiments/blob/main/oracle_eval/shared.py#L186). Here are the results, comparing two different window sizes of STFT - the flag is `--control` since in relation to the NSGT, the STFT is the control/default configuration.

Evaluation of MPI oracle vs. IRM1 oracle on 1 random track from MUSDB18-HQ:
```
$ export MUSDB_PATH=~/TRAINING-MUSIC/MUSDB18-HQ/
$ 
$ python oracle_eval/search_best_nsgt.py --control \
    --control-window-sizes='2048,4096' \
    --oracle='mpi' \
    --n-random-tracks=1
using 1 random tracks from MUSDB18-HQ train set validation split
oracle: mpi
evaluating control stft 2048
drums           ==> SDR:   9.728  SIR:  20.598  ISR:  22.516  SAR:   9.841
bass            ==> SDR:   4.692  SIR:  13.206  ISR:  18.199  SAR:   5.351
other           ==> SDR:   5.151  SIR:  10.215  ISR:  17.343  SAR:   5.470
vocals          ==> SDR:   5.639  SIR:  15.814  ISR:  21.559  SAR:   6.320
accompaniment   ==> SDR:   7.501  SIR:  17.219  ISR:  19.734  SAR:   9.005

median SDR: 5.3947601318359375
evaluating control stft 4096
drums           ==> SDR:   8.581  SIR:  19.429  ISR:  21.557  SAR:   8.883
bass            ==> SDR:   4.716  SIR:  13.182  ISR:  18.502  SAR:   5.357
other           ==> SDR:   5.475  SIR:  10.596  ISR:  17.952  SAR:   5.905
vocals          ==> SDR:   5.882  SIR:  16.298  ISR:  21.445  SAR:   6.400
accompaniment   ==> SDR:   7.686  SIR:  17.225  ISR:  19.857  SAR:   9.250

median SDR: 5.678299903869629
```

IRM1:
```
$ python oracle_eval/search_best_nsgt.py --control \
    --control-window-sizes='2048,4096' \
    --oracle='irm1' \
    --n-random-tracks=1
using 1 random tracks from MUSDB18-HQ train set validation split
oracle: irm1
evaluating control stft 2048
drums           ==> SDR:   9.670  SIR:  20.991  ISR:  15.276  SAR:  10.493
bass            ==> SDR:   6.136  SIR:  13.512  ISR:  12.106  SAR:   7.218
other           ==> SDR:   6.563  SIR:  11.156  ISR:  12.556  SAR:   6.830
vocals          ==> SDR:   6.712  SIR:  16.301  ISR:  13.098  SAR:   7.721
accompaniment   ==> SDR:  13.364  SIR:  18.284  ISR:  23.180  SAR:  14.984

median SDR: 6.637419700622559
evaluating control stft 4096
drums           ==> SDR:   9.362  SIR:  20.382  ISR:  15.063  SAR:  10.503
bass            ==> SDR:   6.109  SIR:  13.620  ISR:  12.247  SAR:   7.165
other           ==> SDR:   6.807  SIR:  11.556  ISR:  13.046  SAR:   7.133
vocals          ==> SDR:   6.824  SIR:  16.848  ISR:  13.171  SAR:   7.881
accompaniment   ==> SDR:  13.517  SIR:  18.312  ISR:  23.784  SAR:  15.127

median SDR: 6.815374851226807
```

Note that the IRM1 oracle performance is higher than the MPI, which leads to the question why the mixed phase approach is preferred.
