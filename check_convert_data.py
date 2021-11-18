import random
from abc import ABCMeta
from pathlib import Path
from torch.utils.data import Dataset
from helper_functions import get_trim_length
import math
import librosa

import musdb
import numpy as np
import soundfile
import torch
import time


MUSDBROOT = "musdb18/musdb18"
musdb_reference = musdb.DB(root=MUSDBROOT, subsets='train', split='train', is_wav=False)

target_names = source_names = ['vocals', 'drums', 'bass', 'other']
stem_list = ['mixture', 'drums', 'bass', 'other', 'vocals']
num_stem = 5

num_tracks = len(musdb_reference)
    
save_path = "musdb18_train/"

print(time.time())
audio_from_musdb = musdb_reference.tracks[0].stems[4].astype(np.float32)
print(time.time())
audio_from_npy = np.load(save_path + musdb_reference.tracks[0].name + " - " + stem_list[4] + ".npy")
print(time.time())

print("Original Audio: ")
print(audio_from_musdb.shape)
print(type(audio_from_musdb[0, 0]))
print(audio_from_musdb[0, 0])

print("npy audio: ")
print(audio_from_npy.shape)
print(type(audio_from_npy[0, 0]))
print(audio_from_npy[0, 0])