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
musdb_reference = musdb.DB(root=MUSDBROOT, subsets='train', split='train', is_wav=False)

target_names = source_names = ['vocals', 'drums', 'bass', 'other']
stem_list = ['mixture', 'drums', 'bass', 'other', 'vocals']
num_stem = 5

num_tracks = len(musdb_reference)
    
save_path = "musdb18_train/"

for i in range(num_tracks):
    for stem_id in range(1, num_stem):
        audio = musdb_reference.tracks[i].stems[stem_id].astype(np.float32)
        np.save(save_path + musdb_reference.tracks[i].name + " - " + stem_list[stem_id], audio)
        print(musdb_reference.tracks[i].name + " - " + stem_list[stem_id] + ".npy saved")


# print(musdb_reference.tracks[0].name)
# print(musdb_reference.tracks[0].stems[0].shape)
# print(musdb_reference.tracks[0].stems[4].shape)

'''
test_audio = musdb_reference.tracks[0].stems[4].astype(np.float32)
print(test_audio.shape)
print(type(test_audio[500][0]))
print(test_audio[500][0])

test_audio_1 = musdb_reference.tracks[0].stems[4]
print(test_audio_1.shape)
print(type(test_audio_1[500][0]))
print(test_audio_1[500][0])

test_audio_2 = (librosa.load(musdb_reference.tracks[0].sources["vocals"].path, sr=44100, mono=False))[0]
test_audio_2 = test_audio_2.T
print(test_audio_2.shape)
print(type(test_audio_2[500][0]))
print(test_audio_2[500][0])
'''

'''
test_audio = musdb_reference.tracks[0].stems[0].astype(np.float32)
print(test_audio.shape)
print(type(test_audio[500][0]))
print(test_audio[500][0])

test_audio_1 = musdb_reference.tracks[0].stems[0]
print(test_audio_1.shape)
print(type(test_audio_1[500][0]))
print(test_audio_1[500][0])
'''



# print(musdb_reference.tracks[0].sources["vocals"].path)
'''
source_names = ['vocals', 'drums', 'bass', 'other']
wav_dict = {i: {s: musdb_reference.tracks[i].sources[s].path for s in source_names}
                         for i in range(num_tracks)}

print(wav_dict[0])
'''