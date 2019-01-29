from os.path import join
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


def extract_mel_spectrogram(
        y, sr, n_fft, hop_length,
        n_mels=128, fmin=0, fmax=8000, power=2, center=False):
    '''
    center=False breaks down for some audio:
        "librosa.util.exceptions.ParameterError: 
            Buffer is too short (n=480) for frame_length=800"

    Custom mel_spectrogram based on librosa with the ability to use the center
    variable. The amount of frames differs based on center is true/false.
    '''

    # from librosa  _spectrogram()
    S = np.abs(librosa.stft(
        y, n_fft=n_fft, hop_length=hop_length, center=center))**power
    # Build a Mel filter
    # mel_basis = librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax)
    # return np.dot(mel_basis, S)

    mel = librosa.feature.melspectrogram(
            S=S, n_fft=n_fft, hop_length=hop_length, power=power)
    return S, mel


def wav2MelSpectrogram(
        wav_path, frame_duration=50, n_mels=128, fmax=8000, default=False):
    ''' extract melspectrogram with fft-frame=hop_length=frame_ms '''

    frame_duration = frame_duration * 1e-3
    y, sr = librosa.core.load(wav_path, sr=None)

    # MelSpectrogram
    frame = librosa.core.time_to_samples(frame_duration, sr)
    if default:
        spectrogram = librosa.feature.melspectrogram(
                y, sr=sr, n_fft=frame, hop_length=frame,
                fmax=fmax, n_mels=n_mels).astype(np.float32).T
    else:
        try:
            spectrogram = extract_mel_spectrogram(
                    y, sr=sr, n_fft=frame, hop_length=frame,
                    fmax=fmax, n_mels=n_mels).astype(np.float32).T
        except:
            print('Failed: ', wav_path)
            spectrogram = None

    return spectrogram


def wav2mel2npy(wavpath, filename, frame_length=50, verbose=False):
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


# ----------- MFCC Features -----------

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
