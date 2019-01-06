'''
Code that extracts GeMap-Features used in:
    https://arxiv.org/pdf/1808.10785.pdf
    https://arxiv.org/pdf/1806.11461.pdf

Assumes that Opensmile binaries are installed along with the correct
configuration files.

0. Install Opensmile using `scripts/install_opensmile.sh`

```bash
python gemaps.py -s wavfile.wav -t gemaps.csv -o /path/to/opensmile-3.2.0 -f 10
```

'''
from os import system, makedirs
from os.path import join, exists, split
import numpy as np
import csv
import matplotlib.pyplot as plt


INDEX2GEMAPS = { 
        0:'Loudness',
        1:'alphaRatio',
        2:'hammarbergIndex',
        3:'slope0-500',
        4:'slope500-1500',
        5:'spectralFlux',
        6:'mfcc1',
        7:'mfcc2',
        8:'mfcc3',
        9:'mfcc4',
        10:'F0semitoneFrom27.5Hz',
        11:'jitterLocal',
        12:'shimmerLocaldB',
        13:'HNRdBACF',
        14:'logRelF0-H1-H2',
        15:'logRelF0-H1-A3',
        16:'F1frequency',
        17:'F1bandwidth',
        18:'F1amplitudeLogRelF0',
        19:'F2frequency',
        20:'F2amplitudeLogRelF0',
        21:'F3frequency',
        22:'F3amplitudeLogRelF0'
        }

GEMAPS2INDEX = {}
for k, v in INDEX2GEMAPS.items():
    GEMAPS2INDEX[v] = k


def load_gemaps(path):
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)  # skip header
        gemaps = []
        for frame in reader:
            gemaps.append([float(f) for f in frame[2:]])  # skip column ('unknown'), ('frameTime')
    return np.array(gemaps, dtype=np.float32)


def plot_gemaps(gemaps):
    ''' plots all the gemaps features
    Argument:
        gemaps:     str: path to csv OR np.ndarray of gemap features
    '''
    if isinstance(gemaps, str):
        gemaps = load_gemaps(gemaps)

    if not isinstance(gemaps, np.ndarray):
        print('Argument needs to be a string: path to csv-file or a numpy array')
        return None


    plt.figure()
    j = 0
    for name, i in GEMAPS2INDEX.items():
        print(i, name)
        plt.subplot(5,5, i+1)

        # plt.plot(gemaps[:, j], label=f"{GeMaps2Index.index2gemaps[i]}")
        plt.plot(gemaps[:, j])
        plt.xlabel(name)
        j += 1
    plt.tight_layout()
    plt.show()


def extract_gemaps_from_wav(
        wavpath,
        filepath='gemaps.csv',
        frame_length=10,
        opensmile = 'opensmile/opensmile-2.3.0',
        verbose=True):
    '''
    Arguments:
        wavpath:        path to wav-file
        filepath:       filepath to output gemaps (.csv) file
        frame_length:   int, length of frames in milliseconds
        opensmile:      path to opensmile-2.3.0
        verbose:        Boolean, True if display success print 
    '''

    BIN_PATH = join(opensmile, 'bin/linux_x64_standalone_static/SMILExtract')
    CONF_50 = join(opensmile, 'config/gemaps_50ms/eGeMAPSv01a.conf')
    CONF_10 = join(opensmile, 'config/gemaps_10ms/eGeMAPSv01a.conf')

    if frame_length == 10:
        opensmile_cmd = BIN_PATH + ' -C '+ CONF_10 + ' -l 0'
    elif frame_length == 50:
        opensmile_cmd = BIN_PATH + ' -C '+ CONF_50 + ' -l 0'
    else:
        print('No configuration file for frame length != 50, 10 ms')
        exit(0)

    # Check if output filepath already exists
    if exists(filepath):
        print(f'Filepath: {filepath} already exists!')
        ans = input('Do you wish to overwrite? (Y/N)\n> ')
        if not ans.lower() == 'y':
            exit(0)

    # Create folder if neccessary
    output_directory, fname = split(filepath)
    if not exists(output_directory):
        makedirs(output_directory)

    cmd = " ".join([opensmile_cmd, '-I', wavpath, '-D', filepath])
    ret = system(cmd)

    if verbose:
        print(f'Extracted GeMaps from {wavpath} and saved into {filepath}')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Gemaps')
    parser.add_argument('-s', '--source', type=str, help='path to wavfile')
    parser.add_argument('-t', '--target', type=str, help='path to output csv')
    parser.add_argument('-f', '--frame_length', type=int, default=10,
            help='frame length in milliseconds')
    parser.add_argument('-o', '--opensmile', type=str,
            help='Path to opensmile directory')
    args = parser.parse_args()


    extract_gemaps_from_wav(
            wavpath=args.source,
            filepath=args.target,
            opensmile=args.opensmile,
            frame_length=args.frame_length)
