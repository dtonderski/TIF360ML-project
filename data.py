import numpy as np
import librosa
import cv2


def specs_from_file(path, seconds = 5, overlap = 1, sr=None, **kwargs):
    sig, rate = librosa.load(path, sr=sr, offset=None, duration=None)
    for spec in specs_from_signal(sig, rate, seconds, overlap, **kwargs):
        yield spec


def specs_from_signal(sig, rate, seconds, overlap, **kwargs):
    splits = split_signal(sig, rate, seconds, overlap)
    for split in splits:
        yield mel_spec(split, rate, **kwargs)


def split_signal(sig, rate, seconds, overlap):
    splits = []
    for sig_start in range(0, len(sig), int((seconds - overlap) * rate)):
        sig_end = sig_start + int(seconds * rate)
        if sig_end > len(sig):
            break
        else:
            split = sig[sig_start:sig_end]
        splits.append(split)
    return splits


def mel_spec(sig, rate, shape=(64, 256), fmin=500, fmax=12500, normalize=True, n_fft = 0):
    N_MELS = shape[0]
    if n_fft > 0:
        N_FFT = n_fft
    else:
        N_FFT = shape[0] * 16
    HOP_LENGTH = len(sig) // (shape[1] - 1)

    spec = librosa.feature.melspectrogram(y=sig,
                                          sr=rate,
                                          n_fft=N_FFT,
                                          hop_length=HOP_LENGTH,
                                          n_mels=N_MELS,
                                          fmin=fmin,
                                          fmax=fmax)

    spec = librosa.amplitude_to_db(spec, ref=np.max, top_db=80)

    if normalize:
        spec -= np.min(spec)
        if np.max(spec) > 0:
            spec /= np.max(spec)

    return spec


def signal2noise(spec):
    # Get working copy
    spec = spec.copy()

    # Calculate median for columns and rows
    col_median = np.median(spec, axis=0, keepdims=True)
    row_median = np.median(spec, axis=1, keepdims=True)

    # Binary threshold
    spec[spec < row_median * 1.25] = 0.0
    spec[spec < col_median * 1.15] = 0.0
    spec[spec > 0] = 1.0

    # Median blur
    spec = cv2.medianBlur(spec, 3)

    # Morphology
    spec = cv2.morphologyEx(spec, cv2.MORPH_CLOSE, np.ones((3, 3), np.float32))

    # Sum of all values
    spec_sum = spec.sum()

    # Signal to noise ratio (higher is better)
    try:
        s2n = spec_sum / (spec.shape[0] * spec.shape[1] * spec.shape[2])
    except IndexError:
        s2n = spec_sum / (spec.shape[0] * spec.shape[1])

    return s2n
