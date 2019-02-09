'''
Dataset for the kaggle competition:
    https://www.kaggle.com/c/tensorflow-speech-recognition-challenge

Download data at:
    https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data
'''
import os
from os.path import join, isdir
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy import signal
from scipy.io import wavfile

import torch
from torch.utils.data import Dataset, DataLoader

# Resnet settings
def norm(x):
    x = x - x.min()
    x /= x.max()
    return x


def plot_log_spectrogram(log_spec, dataclass=None, sample_rate=16000, title='Log Spectrogram'):
    fig = plt.figure(figsize=(14, 8))
    plt.title('Log Spectrogram ({})'.format(dataclass))
    plt.imshow(log_spec.T, aspect='auto', origin='lower')
    plt.xticks([])
    plt.xlabel('Seconds')
    plt.ylabel('Freqs in Hz')
    plt.yticks([])
    plt.show()


# PyTorch Dataset 
class WordClassificationDataset(Dataset):
    def __init__(self,
                 audio_path="data/train/audio",
                 window_size=20,
                 hop_len=10,
                 samples=False):
        super(WordClassificationDataset, self).__init__()

        self.audio_path = audio_path
        self.classes = [f for f in os.listdir(audio_path) 
                        if isdir(os.path.join(audio_path, f))]
        self.classes.sort() 

        # self.classes[1:] do not include background noise (broken sounds)
        if len(self.classes) > 1:
            self.classes = self.classes[1:]

        self.num_classes = len(self.classes)

        self.class2idx = {}
        self.idx2class = {}

        self.window_size = window_size
        self.hop_len = hop_len

        self.use_one_hot = False
        if samples:
            self.datapoints = self.samples()
        else:
            self.datapoints = self._datapoints()  # list of (dpath, label)

    def samples(self, nfiles=3):
        paths = []
        exceptions = 0
        for i, c in enumerate(self.classes):
            self.class2idx[c] = i
            self.idx2class[i] = c
            n = 0
            folder_path = join(self.audio_path, c)
            for f in os.listdir(folder_path):
                if f.endswith('.wav'):
                    fpath = join(folder_path, f)

                    # Expensive check.
                    sample_rate, samples = wavfile.read(fpath)
                    if samples.shape[0] != 16000:
                        exceptions += 1
                        print('wrong sample rate: ', exceptions)
                        continue
                    elif samples.mean() == 0:
                        exceptions += 1
                        print('zero values')
                        continue
                    paths.append((fpath, i))
                    n += 1
                    if n == nfiles:
                        break
        print('Total samples: {} ({} unused)'.format(len(paths), exceptions))
        return paths

    def _datapoints(self):
        paths = []
        exceptions = 0
        for i, c in enumerate(self.classes):
            self.class2idx[c] = i
            self.idx2class[i] = c
            folder_path = join(self.audio_path, c)
            for f in os.listdir(folder_path):
                if f.endswith('.wav'):
                    fpath = join(folder_path, f)

                    # Expensive check.
                    sample_rate, samples = wavfile.read(fpath)
                    if samples.shape[0] != 16000:
                        exceptions += 1
                        # print('wrong sample rate: ', exceptions)
                        continue
                    elif samples.mean() == 0:
                        exceptions += 1
                        # print('zero values')
                        continue
                    paths.append((fpath, i))
        print('Total samples: {} ({} unused)'.format(len(paths), exceptions))
        return paths

    def get_random(self):
        idx = int(torch.randint(0, len(self), (1,)).item())
        return self[idx]

    def log_spectrogram(self, samples, sample_rate, eps=1e-10):
        nperseg = int(round(self.window_size * sample_rate / 1e3))
        noverlap = int(round(self.hop_len * sample_rate / 1e3))
        freqs, times, spec = signal.spectrogram(samples,
                                                fs=sample_rate,
                                                window='hann',
                                                nperseg=nperseg,
                                                noverlap=noverlap,
                                                detrend=False)
        return freqs, times, np.log(spec.T.astype(np.float32) + eps)

    def __len__(self):
        return len(self.datapoints)

    def onehot(self, idx):
        onehot = np.zeros((self.num_classes,)).astype(np.float32)
        onehot[idx] = 1
        return onehot

    def standardize(self, x):
        try:
            x = (x-x.mean()) / x.std()
        except:
            print('mean: ', x.mean())
            print('std: ', x.std())
        return x

    def normalize(self, x):
        x = x - x.min()
        x /= x.max()
        return x

    def __getitem__(self, idx):
        path, label = self.datapoints[idx]
        filename = os.path.abspath(path)
        sample_rate, samples = wavfile.read(filename)
        _ , _, log_spec = self.log_spectrogram(samples, sample_rate)

        # log_spec = self.standardize(log_spec)
        log_spec = self.normalize(log_spec)  # in range [0, 1]

        samples = samples.astype(np.float32)
        if self.use_one_hot:
            label = self.onehot(label)
        # return {'samples': samples, 'log_spec': log_spec, 'label': label}
        return [samples, log_spec, label]


def collate_fn(batch):
    samples, log_specs, labels = zip(*batch)
    samples = torch.tensor(samples)
    log_specs = torch.tensor(log_specs)
    log_specs = log_specs.unsqueeze(1)  # add channel dimension
    labels = torch.LongTensor(labels)
    return {'samples': samples, 'log_specs': log_specs, 'labels': labels}



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Word Dataset')
    parser.add_argument('-d', '--dir', type=str, default='train/audio')
    args = parser.parse_args()

    print('\nCreate Dataset and DataLoader')
    print('-----------------------------')
    dset = WordClassificationDataset(audio_path=args.dir)

    samples, log_spec, label = dset.get_random()
    dloader = DataLoader(dset, collate_fn=collate_fn, batch_size=16, shuffle=True)

    print('Number of datapoints: ', len(dset))
    print('Number of classes: ', dset.num_classes)

    print('\nSample Batch')
    print('------------')
    for d in dloader:
        samples = d['samples']
        log_specs = d['log_specs']
        labels = d['labels']
        print('\nSamples {}, {}'.format(samples.shape, samples.dtype))
        print('Samples max: {}, min: {}'.format(samples.max(), samples.min()))
        print('Log specs {}, {}'.format(log_specs.shape, log_specs.dtype))
        print('Log specs max: {}, min: {}'.format(log_specs.max(), log_specs.min()))
        print('Log specs mean: {}, std: {}'.format(log_specs.mean(), log_specs.std()))
        print('Labels {},{}'.format(labels.shape, labels.dtype))
        # Samples torch.Size([16, 16000]), torch.float32
        # Log specs torch.Size([16, 99, 161]), torch.float32
        # Labels torch.Size([16]),torch.int64
        break

    print('\nPlot data sample')
    print('----------------')
    sample , log_spec, label = dset.get_random()
    print(log_spec.shape)
    plot_log_spectrogram(log_spec, dataclass=dset.idx2class[label])


