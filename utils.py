import numpy as np
import pandas as pd
import librosa
import pywt

NAMES = ['LL', 'LP', 'RP', 'RR']
FEATS = [['Fp1', 'F7', 'T3', 'T5', 'O1'],
         ['Fp1', 'F3', 'C3', 'P3', 'O1'],
         ['Fp2', 'F8', 'T4', 'T6', 'O2'],
         ['Fp2', 'F4', 'C4', 'P4', 'O2']]

def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def denoise(x, wavelet='haar', level=1):
    """ Denoising with wavelet transform. """
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = [pywt.threshold(c, value=uthresh, mode='hard') for c in coeff[1:]]
    ret = pywt.waverec(coeff, wavelet, mode='per')
    return ret

def eeg_to_spectrogram(df, use_wavelet=False, channels=FEATS, sample_length=10_000):
    """ Convert EEG data to spectrograms. """
    middle = (len(df) - sample_length) // 2
    df = df.iloc[middle:middle + sample_length]

    img = np.zeros((128, 256, 4), dtype='float32')
    for k in range(4):  # Processing each region of interest
        COLS = channels[k]
        for kk in range(4):  # Loop through each feature channel
            x = df[COLS[kk]].values - df[COLS[kk + 1]].values
            m = np.nanmean(x)
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m)
            else:
                x[:] = 0
            if use_wavelet:
                x = denoise(x)

            mel_spec = librosa.feature.melspectrogram(y=x, sr=200, hop_length=len(x) // 256,
                                                      n_fft=1024, n_mels=128, fmin=0, fmax=20, win_length=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)
            width = (mel_spec_db.shape[1] // 32) * 32
            mel_spec_db = mel_spec_db[:, :width]
            mel_spec_db = (mel_spec_db + 40) / 40  # Normalize
            img[:, :, k] += mel_spec_db
        img[:, :, k] /= 4.0  # Averaging over all channels

    return img
