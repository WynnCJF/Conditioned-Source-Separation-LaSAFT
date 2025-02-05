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


def check_musdb_valid(musdb_train):
    if len(musdb_train) > 0:
        pass
    else:
        print('It seems like you used wrong path for musdb18 dataset')
        raise NotImplemented  # TODO: Exception handling


class MusdbWrapperDataset(Dataset):
    __metaclass__ = ABCMeta

    def __init__(self, musdb_root, subset, n_fft, hop_length, num_frame):

        # musdb_root = Path(musdb_root)
        # self.root = musdb_root.joinpath(subset)

        if subset == 'test':
            self.musdb_reference = musdb.DB(root=musdb_root, subsets='test', is_wav=False)
        elif subset == 'train':
            self.musdb_reference = musdb.DB(root=musdb_root, subsets='train', split='train', is_wav=False)
        elif subset == 'valid':
            self.musdb_reference = musdb.DB(root=musdb_root, subsets='train', split='valid', is_wav=False)
        else:
            raise ModuleNotFoundError
        # Note: specify the data to use

        check_musdb_valid(self.musdb_reference)

        self.target_names = self.source_names = ['vocals', 'drums', 'bass', 'other']
        
        # Note: created by wy
        self.target_lookup = {'mixture': 0, 'drums': 1, 'bass': 2, 'other': 3, 'vocals': 4}
        self.train_audio_path = "musdb18_train/"

        self.num_tracks = len(self.musdb_reference)
        self.num_targets = len(self.target_names)
        # cache wav file dictionary
        self.wav_dict = {i: {s: self.musdb_reference.tracks[i].sources[s].path for s in self.source_names}
                         for i in range(self.num_tracks)}
        # Note: create a dictionary where keys are track index, and values are another dictionary, where
        #       keys are source name and values are the file path for that specific source wav file

        for i in range(self.num_tracks):
            self.wav_dict[i]['mixture'] = self.musdb_reference[i].path
        # Note: add another key 'mixture' to the secondary dictionary, with the value of mixture wav file path

        self.lengths = [self.get_audio(i, 'vocals').shape[0] for i in range(self.num_tracks)]
        # Note: self.lengths is a list of all track's lengths
        self.window_length = hop_length * (num_frame - 1)
        # Note: Guess the window_length here is the total length of a track that is used to generate data

    # Note: (below) the length of dataset is the number of stems?
    def __len__(self): # -> int
        return self.num_tracks * self.num_targets

    def get_audio(self, idx, target_name, pos=0, length=None):
        '''
        arg_dicts = {
            'file': self.wav_dict[idx][target_name],
            'start': pos,
            'dtype': 'float32'
        }

        if length is not None:
            arg_dicts['stop'] = pos + length

        return soundfile.read(**arg_dicts)[0]
        '''
        
        # Read in the audio
        # Note: librosa is different from soundfile in that, librosa counts the offset and duration in seconds, while
        #       soundfile as well as this implementation counts them in samples, so conversion is needed here
        '''
        sr = 44100
        start_pos_sec = pos / sr
        
        if (length is not None):
            duration_sec = length / sr
            audio = (librosa.load(self.wav_dict[idx][target_name], sr=44100, mono=False, offset=start_pos_sec, duration=duration_sec))[0]
        else:
            audio = (librosa.load(self.wav_dict[idx][target_name], sr=44100, mono=False, offset=start_pos_sec))[0]
        
        audio = audio.T
        '''
        
        # Crop the audio to the desired part
        '''
        if (length is not None):
            audio = audio[pos : pos + length, :]
        else:
            audio = audio[pos:, :]
        '''
        
        '''
        target_index = self.target_lookup[target_name]
        if (length is not None):
            audio = (self.musdb_reference.tracks[idx].stems[target_index])[pos : pos + length, :]
        else:
            audio = (self.musdb_reference.tracks[idx].stems[target_index])[pos : , :]
        
        print("Audio length: ", audio.shape)
        '''
        
        full_len_audio = np.load(self.train_audio_path + self.musdb_reference.tracks[idx].name + " - " + target_name + ".npy")
        
        if (length is not None):
            audio = full_len_audio[pos : pos + length, :]
        else:
            audio = full_len_audio[pos : , :]
            
        # print("Audio length: ", audio.shape)
        
        return audio


class MusdbTrainSet(MusdbWrapperDataset):
    
    def __init__(self,
                 musdb_root='etc/musdb18_samples_wav/',
                 n_fft=2048,
                 hop_length=1024,
                 num_frame=64):
        super(MusdbTrainSet, self).__init__(musdb_root, 'train',  n_fft, hop_length, num_frame)

    def __len__(self):
        return sum([length // self.window_length for length in self.lengths]) * len(self.target_names)
    # Note: what does this length mean? 
    #       Judging from the evaluation set, the length is (total number of chuncks x number of targets)

    def __getitem__(self, whatever):
        source_sample = {target: self.get_random_audio_sample(target) for target in self.source_names}
        # Note: randomly mix a track with the four stems (not necessarily from the same track)
        rand_target = np.random.choice(self.target_names)
        # Note: randomly choose a target

        mixture = sum(source_sample.values())
        # Note: sum up the stems to get the mixture
        target = source_sample[rand_target]
        # Note: pick out the target stem

        input_condition = np.array(self.source_names.index(rand_target), dtype=np.long)
        # Note: the input condition is an np array with source index

        return torch.from_numpy(mixture), torch.from_numpy(target), torch.tensor(input_condition, dtype=torch.long)
        # Note: the output of this dataset is mixture audio (not spectrogram), target audio (not spectrogram), and 
        #       a single scale tensor specifying the mixture index

    def get_random_audio_sample(self, target_name):
        return self.get_audio_sample(random.randint(0, self.num_tracks - 1), target_name)

    def get_audio_sample(self, idx, target_name):
        length = self.lengths[idx] - self.window_length
        # Note: ensure enough length for at least one self.window_length
        start_position = random.randint(0, length - 1)
        
        # print("Fetching track #", idx, " starting from", start_position, " and length", self.window_length)
        return self.get_audio(idx, target_name, start_position, self.window_length)


class MusdbEvalSet(MusdbWrapperDataset):
    
    def __init__(self, musdb_root, eval_type, n_fft, hop_length, num_frame):

        super(MusdbEvalSet, self).__init__(musdb_root, eval_type, n_fft, hop_length, num_frame)

        self.hop_length = hop_length
        self.trim_length = get_trim_length(self.hop_length)
        # Note: what is get_trim_length or trim_length?
        self.true_samples = self.window_length - 2 * self.trim_length
        # Note: Guess this is the true sample number in the part of the track used as data

        num_chunks = [math.ceil(length / self.true_samples) for length in self.lengths]
        self.acc_chunk_final_ids = [sum(num_chunks[:i + 1]) for i in range(self.num_tracks)]
        # Note: accumulate the chunk number of different tracks and store as the id of the final chunk of a track

        file_name = 'linear_mixture.wav'
        for i in range(self.num_tracks):
            self.wav_dict[i]['linear_mixture'] = self.wav_dict[i]['vocals'][:-10] + file_name
            # Note: what is this path? It adds a new .wav file to each original track

    def __len__(self):
        return self.acc_chunk_final_ids[-1] * len(self.target_names)

    def __getitem__(self, idx):
        target_offset = idx % len(self.target_names)
        idx = idx // len(self.target_names)
        # Note: seems that one track is laid out as different stems in the dataset
        # Note: what is idx now? From the following functions it seems that idx is still an id of chunk, and
        #       the dataset is layed out as:
        #       track1.chunk1.stem1, track1.chunk1.stem2, track1.chunk1.stem3, track1.chunk1.stem4, 
        #       track1.chunk2.stem1, ...
        #       ...
        #       track2.chunk1.stem1, track2.chunk1.stem2, track2.chunk1.stem3, track2.chunk1.stem4, 
        #       ...

        target_name = self.target_names[target_offset]
        # Note: find the exact target stem

        mixture, mixture_idx, offset = self.get_mixture_sample(idx)

        input_condition = np.array(target_offset, dtype=np.long)

        mixture = torch.from_numpy(mixture)
        input_condition = torch.tensor(input_condition, dtype=torch.long)
        window_offset = offset // self.true_samples
        # Note: what is this window_offset?

        return mixture, mixture_idx, window_offset, input_condition, target_name

    def get_mixture_sample(self, idx):
        mixture_idx, start_pos = self.idx_to_track_offset(idx)
        length = self.true_samples
        left_padding_num = right_padding_num = self.trim_length
        mixture_length = self.lengths[mixture_idx]
        if start_pos + length > mixture_length:  # last
            right_padding_num += self.true_samples - (mixture_length - start_pos)
            # Note: what is this?
            length = None

        mixture = self.get_audio(mixture_idx, 'linear_mixture', start_pos, length)

        mixture = np.concatenate((np.zeros((left_padding_num, 2), dtype=np.float32), mixture,
                                  np.zeros((right_padding_num, 2), dtype=np.float32)), 0)
        # Conduct 2D zero padding to the left and right of the mixture
        return mixture, mixture_idx, start_pos

    def idx_to_track_offset(self, idx):
        for i, last_chunk in enumerate(self.acc_chunk_final_ids):
            if idx < last_chunk:
                if i != 0:
                    offset = (idx - self.acc_chunk_final_ids[i - 1]) * self.true_samples
                else:
                    offset = idx * self.true_samples
                return i, offset
            # Note: obtain the specific mixture index as i, and the specific sample that starts the 
            #       chunk as offset

        return None, None
    
    
def MusdbValidSet(musdb_root='etc/musdb18_samples_wav/',
                  n_fft=2048,
                  hop_length=1024,
                  num_frame=64):
    return MusdbEvalSet(musdb_root, 'valid', n_fft, hop_length, num_frame)


def MusdbTestSet(musdb_root='etc/musdb18_samples_wav/',
                 n_fft=2048,
                 hop_length=1024,
                 num_frame=64):
    return MusdbEvalSet(musdb_root, 'test', n_fft, hop_length, num_frame)


class MusdbEvalSetWithGT(MusdbEvalSet):
    
    def __init__(self, musdb_root, eval_type, n_fft, hop_length, num_frame):
        super(MusdbEvalSetWithGT, self).__init__(musdb_root, eval_type, n_fft, hop_length, num_frame)

    def __getitem__(self, idx):
        target_offset = idx % len(self.target_names)
        idx = idx // len(self.target_names)

        target_name = self.target_names[target_offset]

        mixture, target, mixture_idx, offset = self.get_mixture_sample_with_GT(idx, target_name)

        input_condition = np.array(target_offset, dtype=np.long)

        mixture = torch.from_numpy(mixture)
        input_condition = torch.tensor(input_condition, dtype=torch.long)
        window_offset = offset // self.true_samples

        return mixture, target, mixture_idx, window_offset, input_condition, target_name
        # Note: compared with MusdbEvalSet, this class returns both the mixture and the target

    def get_mixture_sample_with_GT(self, idx, target_name):
        mixture_idx, start_pos = self.idx_to_track_offset(idx)
        length = self.true_samples
        left_padding_num = right_padding_num = self.trim_length
        mixture_length = self.lengths[mixture_idx]
        if start_pos + length > mixture_length:  # last
            right_padding_num += self.true_samples - (mixture_length - start_pos)
            length = None

        mixture = self.get_audio(mixture_idx, 'linear_mixture', start_pos, length)
        target = self.get_audio(mixture_idx, target_name, start_pos, length)

        mixture = np.concatenate((np.zeros((left_padding_num, 2), dtype=np.float32), mixture,
                                  np.zeros((right_padding_num, 2), dtype=np.float32)), 0)
        target = np.concatenate((np.zeros((left_padding_num, 2), dtype=np.float32), target,
                                 np.zeros((right_padding_num, 2), dtype=np.float32)), 0)

        return mixture, target, mixture_idx, start_pos


def MusdbValidSetWithGT(musdb_root='etc/musdb18_samples_wav/',
                        n_fft=2048,
                        hop_length=1024,
                        num_frame=64):
    return MusdbEvalSetWithGT(musdb_root, 'valid', n_fft, hop_length, num_frame)


def MusdbTestSetWithGT(musdb_root='etc/musdb18_samples_wav/',
                       target_names=None,
                       n_fft=2048,
                       hop_length=1024,
                       num_frame=64):
    return MusdbEvalSetWithGT(musdb_root, 'test', target_names, n_fft, hop_length, num_frame)


class SingleTrackSet(Dataset):
    
    def __init__(self, track, hop_length, num_frame):

        assert len(track.shape) == 2
        assert track.shape[1] == 2  # check stereo audio

        self.hop_length = hop_length
        self.window_length = hop_length * (num_frame - 1)
        self.trim_length = get_trim_length(self.hop_length)

        self.true_samples = self.window_length - 2 * self.trim_length

        self.lengths = [track.shape[0]]
        self.num_tracks = 1

        num_chunks = [math.ceil(length / self.true_samples) for length in self.lengths]
        self.acc_chunk_final_ids = [sum(num_chunks[:i + 1]) for i in range(self.num_tracks)]

        self.cached = track.astype(np.float32) if track.dtype is not np.float32 else track

    def __len__(self):
        return self.acc_chunk_final_ids[-1]

    def __getitem__(self, idx):

        track_idx, start_pos = self.idx_to_track_offset(idx)

        length = self.true_samples
        left_padding_num = right_padding_num = self.trim_length
        if track_idx is None:
            raise StopIteration
        mixture_length = self.lengths[track_idx]
        if start_pos + length > mixture_length:  # last
            right_padding_num += self.true_samples - (mixture_length - start_pos)
            length = None

        mixture = self.get_audio(start_pos, length)

        mixture = np.concatenate((np.zeros((left_padding_num, 2), dtype=np.float32), mixture,
                                  np.zeros((right_padding_num, 2), dtype=np.float32)), 0)

        mixture = torch.from_numpy(mixture)

        return mixture

    def idx_to_track_offset(self, idx):

        for i, last_chunk in enumerate(self.acc_chunk_final_ids):
            if idx < last_chunk:
                if i != 0:
                    offset = (idx - self.acc_chunk_final_ids[i - 1]) * self.true_samples
                else:
                    offset = idx * self.true_samples
                return i, offset

        return None, None

    def get_audio(self, pos=0, length=None):

        track = self.cached

        return track[pos:pos + length] if length is not None else track[pos:]
    

'''
training_set = MusdbTrainSet(self.musdb_root, self.n_fft, self.hop_length, self.num_frame)

loader = DataLoader(training_set, shuffle=True, batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory)
                    
validation_set = MusdbValidSetWithGT(self.musdb_root, self.n_fft, self.hop_length, self.num_frame)

loader = DataLoader(validation_set, shuffle=False, batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory)
'''