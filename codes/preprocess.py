import os
from scipy.io import loadmat, matlab
from scipy.signal import spectrogram, find_peaks, medfilt
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# remove these exercises from targets since there are no data samples
ignore_exercises = ['<Initial Activity>', 'Arm straight up', 'Invalid', 'Non-Exercise',
                    'Note', 'Tap IMU Device', 'Tap Right Device']

# load in the 75 different exercises available in the full data
exercises = pd.read_csv('codes/exercises.txt', header=None, names=['exercise'])  # assumes the cwd is highest level
exercises = exercises.where(~exercises.exercise.isin(ignore_exercises)).dropna()

# 5 samples of exercise to classify for the simple model
targets_idx = {
    'Crunch': 10,
    'Jumping Jacks': 23,
    'Running (treadmill)': 42,
    'Squat': 50,
    'Walk': 70
    }

# Spectrogram Parameters for 'spec_gram'
spec_params = dict(fs=50, window='hamming',
                   nperseg=256, nfft=500,
                   noverlap=256 - 50,  # half a second time difference
                   detrend='constant',
                   scaling='spectrum',
                   )

pad_size = ((spec_params['nperseg'] - spec_params['noverlap'],), (0,))


def load_mat(filename):
    """
    This function should be called instead of direct scipy.io.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """

    def _check_vars(d):
        """
        Checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            if isinstance(d[key], matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
            elif isinstance(d[key], np.ndarray):
                d[key] = _toarray(d[key])
        return d

    def _todict(matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _toarray(elem)
            else:
                d[strg] = elem
        return d

    def _toarray(ndarray):
        """
        A recursive function which constructs ndarray from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        """
        if ndarray.dtype != 'float64':
            elem_list = []
            for sub_elem in ndarray:
                if isinstance(sub_elem, matlab.mio5_params.mat_struct):
                    elem_list.append(_todict(sub_elem))
                elif isinstance(sub_elem, np.ndarray):
                    elem_list.append(_toarray(sub_elem))
                else:
                    elem_list.append(sub_elem)
            return np.array(elem_list)
        else:
            return ndarray

    data = loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_vars(data)


def spec_gram(subj, spec_params=spec_params, pad_size=pad_size):
    """

    Args:
        subj (numpy.ndarray):  nd
        spec_params:
        pad_size:

    Returns:

    """
    spec3d = []
    pad_sig = pd.DataFrame(np.pad(subj, pad_size, 'symmetric'), columns=['time', 'x', 'y', 'z'])
    for d in ['x', 'y', 'z']:
        f, t, Sxx = spectrogram(pad_sig[d], **spec_params)
        spec3d.append(Sxx)
    return f, t, spec3d


def log_freqs(nfft=None, fmin=1e-1, fs=50, bpo=10):
    """
    Calculate a log-frequency spectrogram
    %   S is spectrogram to log-transform;
    %   nfft is parent FFT window; fs is the source samplerate.
    %    Optional FMIN is the lowest frequency to display (1mHz);
    %    BPO is the number of bins per octave (10).
    %    MX returns the nlogbin x nfftbin mapping matrix;
    %    sqrt(MX'*(Y.^2)) is an approximation to the original FFT
    %    spectrogram that Y is based on, suitably blurred by going
    %    through the log-F domain.
    """

    # Ratio between adjacent frequencies in log-f axis
    fratio = 2 ** (1 / bpo)
    # How many bins in log-f axis
    nbins = int(np.log2((fs / 2) / fmin) // np.log2(fratio)) + 1

    # nfft is parent FFT window
    if nfft is None:
        nfft = 500
    # Freqs corresponding to each bin in FFT
    fftfrqs = np.arange(0, nfft / 2 + 1).reshape(1, -1) * (fs / nfft)
    nfftbins = int(nfft / 2 + 1)

    # Freqs corresponding to each bin in log F output
    logffrqs = fmin * np.exp(np.log(2) * np.arange(0, nbins).reshape(1, -1) / bpo)

    # Bandwidths of each bin in log F
    logfbws = logffrqs * (fratio - 1)
    # .. but bandwidth cannot be less than FFT binwidth
    logfbws = np.maximum(logfbws, fs / nfft)

    # Controls how much overlap there is between adjacent bands
    ovfctr = 0.54  # Adjusted by hand to make sum(mx'*mx) close to 1.0

    # Weighting matrix mapping energy in FFT bins to logF bins
    # is a set of Gaussian profiles depending on the difference in
    # frequencies, scaled by the bandwidth of that bin
    freqdiff = ((np.tile(logffrqs.T, (1, nfftbins)) - np.tile(fftfrqs, (nbins, 1)))
                / np.tile(ovfctr * logfbws.T, (1, nfftbins)))
    mx = np.exp(-0.5 * freqdiff ** 2)
    # Normalize rows by sqrt(E), so multiplying by mx' gets approx orig spec back
    mx = mx / np.tile(np.sqrt(2 * (mx ** 2).sum(1).reshape(-1, 1)), (1, nfftbins))
    return logffrqs, mx


def log_specgram(S, mx=None):
    if mx is None:
        logffrqs, mx = log_freqs()
    # Perform mapping in magnitude-squared (energy) domain
    return np.sqrt(mx @ (np.abs(S) ** 2))


def signal_feats(s):
    peaks, p_dict = find_peaks(medfilt(s, 5), prominence=0.1, width=5)

    avg_peak_dist = np.diff(peaks).mean()

    print('avg peak dist: ', round(avg_peak_dist, 2),
          '\tavg prom: ', round(p_dict['prominences'].mean(), 2),
          '\tavg width: ', round(p_dict['width_heights'].mean(), 2))

    return peaks, p_dict


def segment_signal(data, targets=None, win=250, stride=50):
    """
    Segment signal of an exercise recording into chunks of designated window size.

    Args:

        data (dict):    signal data in dict of a recording of exercise from a subject with keys structure of
                            {'data':{accelDataMatrix':[...], 'gyroDataMatrix':[...]}}
        targets (dict): exercises with their index in the 'exercises.txt' and in 'singleonly.mat' file
        win:              window of the signal to look at (default: 250 samples = 5 seconds)
        stride:           step-size of samples to skip for the next window (default: 50 samples = 1 seconds)
        *** Assumes the sampling raate of the motion data is 50 Hz.

    Returns:
        A generator yielding windowed signal.
    """
    if data['activityName'] not in targets:  # only segment signal if there the activity is in the targets
        return

    signal = np.hstack((data['data']['accelDataMatrix'][:, 1:], data['data']['gyroDataMatrix'][:, 1:]))
    # Data sanity check: only process the data for both accel & gyro that at least has the same samples as the window
    # size
    if signal.shape[0] > win:
        if targets is None:
            targets = targets_idx
        ex = np.array(
                [1 if k == data['activityName'] else 0 for k in targets])  # convert the exercise name to interger to
        # pass into CNN model
        steps = (signal.shape[0] - win) // stride if signal.shape[0] > 300 else 1
        segments = np.vstack([[signal[s * stride:s * stride + win], ex] for s in range(steps)])
        if isinstance(segments, np.ndarray):
            return segments
