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


MUSDBROOT = "musdb18/musdb18"
musdb_reference = musdb.DB(root=MUSDBROOT, subsets='train', split='valid', is_wav=False)

target_names = source_names = ['vocals', 'drums', 'bass', 'other']
stem_list = ['linear_mixture', 'drums', 'bass', 'other', 'vocals']
num_stem = 5

num_tracks = len(musdb_reference)
    
save_path = "musdb18_valid/"

for i in range(num_tracks):
    for stem_id in range(1, num_stem):
        stem_audio = musdb_reference.tracks[i].stems[stem_id].astype(np.float32)
        np.save(save_path + musdb_reference.tracks[i].name + " - " + stem_list[stem_id], stem_audio)
        print(musdb_reference.tracks[i].name + " - " + stem_list[stem_id] + ".npy saved")
        
    linear_mixture = musdb_reference.tracks[i].targets["linear_mixture"].audio.astype(np.float32)
    np.save(save_path + musdb_reference.tracks[i].name + " - " + "linear_mixture", linear_mixture)
    print(musdb_reference.tracks[i].name + " - " + "linear_mixture" + ".npy saved")


'''
ta1 = musdb_reference.tracks[0].stems[1]
ta2 = musdb_reference.tracks[0].stems[2]
ta3 = musdb_reference.tracks[0].stems[3]
ta4 = musdb_reference.tracks[0].stems[4]

test_audio_add_mixture = ta1 + ta2 + ta3 + ta4
print(test_audio_add_mixture.shape)
print(test_audio_add_mixture)

test_audio_linear_mixture = musdb_reference.tracks[0].targets["linear_mixture"].audio
print(test_audio_linear_mixture.shape)
print(test_audio_linear_mixture)
'''