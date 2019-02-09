from os.path import join, expanduser
from os import makedirs
import subprocess
import sys

from ml_reusable.utils.gdrive_downloader import download_file_from_google_drive

# Needs to be downloaded from goodle drive
dataset = 'CelebA'
fname = 'img_align_celeba.zip'
id = '0B7EVK8r0v71pZjFTYXZWM3FlRnM'

if len(sys.argv) > 1:
    destination = sys.argv[1]
else:
    destination = join(expanduser('~'), 'Data/vision')

makedirs(destination, exist_ok=True)
destination = join(destination, fname)

print(f'Downloading {dataset} ==> {destination}')
print('This takes a while...')
print()
download_file_from_google_drive(id, destination)

print('Download Done!')
cmd = ['tar', '-jxvf', fname]
subprocess.run(cmd)

