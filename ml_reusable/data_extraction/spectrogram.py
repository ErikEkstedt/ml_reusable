import numpy as np
import librosa
from os.path import join, split


def wav2Spectrogram(wav_path, frame_duration=10e-3, fmax=8000, n_mels=128):
    '''Extracts melspectrogram with fft-frame=hop_length=frame_ms'''
    y, sr = librosa.core.load(wav_path, sr=None)
    
    # MelSpectrogram
    frame = librosa.core.time_to_samples(frame_duration, sr)
    spectrogram = librosa.feature.melspectrogram(y, sr=sr,
            n_fft=frame,
            hop_length=frame,
            fmax=fmax,
            n_mels=n_mels).astype(np.float32)
    return spectrogram


def wav2npy(wav_path, filename=None, frame_duration=10e-3, verbose=False):
    if not filename:
        dpath, fname = split(wav_path)
        filename = join(dpath, fname.replace('.wav', '.npy'))

    mels = wav2Spectrogram(wav_path, frame_duration)
    if verbose: print(f'Saving spectrogram -> {filename}')
    np.save(filename, mels)
    return filename
