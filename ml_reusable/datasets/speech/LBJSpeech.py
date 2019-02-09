import subprocess
from os.path import join, expanduser
from os import makedirs
import sys


dataset = 'LJSpeech'
name = 'LJSpeech-1.1.tar.bz2'
url = 'https://data.keithito.com/data/speech/' + name


if len(sys.argv) > 1:
    target = sys.argv[1]
else:
    target = join(expanduser('~'), 'Data/speech')


makedirs(target, exist_ok=True)
target = join(target, name)

print(f'Downloading {dataset} ==> {target}')
print()
cmd = ['wget', url, '-O', target]
subprocess.run(cmd)

print('Download Complete!')
print('Extracting files')
untar = ['tar', '-jxvf', target]
subprocess.run(cmd)
