from os.path import join, split
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


#----------- MelSpectrogram ----------
def plot_melspectrogram(spec):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
            librosa.power_to_db(spec, ref=np.max),
            y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()


def wav2MelSpectrogram(wav_path, frame_duration=10, n_mels=128, fmax=8000):
    ''' extract melspectrogram with fft-frame=hop_length=frame_ms '''
    frame_duration = frame_duration * 1e-3
    y, sr = librosa.core.load(wav_path, sr=None)
    
    # MelSpectrogram
    frame = librosa.core.time_to_samples(frame_duration, sr)
    spectrogram = librosa.feature.melspectrogram(y, sr=sr, n_fft=frame,
            hop_length=frame, fmax=fmax, n_mels=n_mels).astype(np.float32).T
    return spectrogram


def wav2mel2npy(wavpath, filename, frame_length=10, verbose=False):
    mels = wav2MelSpectrogram(wavpath, frame_length)

    if not filename.endswith('.npy'):
        filename += '.npy'
    np.save(filename, mels)
    if verbose: print(f'Saved spectrogram to disk -> {filename}')


def test_spectrogram():
    wpath = '/home/erik/Data/hej.wav'
    wav2mel2npy(wpath, 'test.npy')
    mel = np.load('test.npy')
    plot_melspectrogram(mel.T)

#----------- MFCC Features -----------

def extract_mfcc(wavpath, n_mfcc=40, dct_type=2):
    ''' MFCC
    Arguments:
        wavpath:        path to wavfile
        n_mfcc:         Int, number of mfcc features
        dct_type:       Int, one of 1,2,3. (default: 2: Slaney-RASTAMAT,
                        (3: m_htk) )
    Returns:
        mfccs:          np.ndarray
    '''
    y, sr = librosa.core.load(wavpath, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, dct_type=dct_type)
    return mfccs

def plot_mfccs(mfccs):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()
