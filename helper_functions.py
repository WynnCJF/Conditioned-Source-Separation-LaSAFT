import os
from warnings import warn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import math


def get_activation_by_name(activation):
    if activation == "leaky_relu":
        return nn.LeakyReLU
    elif activation == "relu":
        return nn.ReLU
    elif activation == "sigmoid":
        return nn.Sigmoid
    elif activation == "tanh":
        return nn.Tanh
    elif activation == "softmax":
        return nn.Softmax
    elif activation == "identity":
        return nn.Identity
    else:
        return None


def get_optimizer_by_name(optimizer):
    if optimizer == "adam":
        return torch.optim.Adam
    elif optimizer == "adagrad":
        return torch.optim.Adagrad
    elif optimizer == "sgd":
        return torch.optim.SGD
    elif optimizer == "rmsprop":
        return torch.optim.RMSprop
    else:
        return torch.optim.Adam
    

def string_to_tuple(kernel_size):
    kernel_size_ = kernel_size.strip().replace('(', '').replace(')', '').split(',')
    kernel_size_ = [int(kernel) for kernel in kernel_size_]
    return kernel_size_


def string_to_list(int_list):
    int_list_ = int_list.strip().replace('[', '').replace(']', '').split(',')
    int_list_ = [int(v) for v in int_list_]
    return int_list_


# ---------------------------- PoCM ---------------------------- #
def Pocm_naive(x, gammas, betas):
    """
    :param x: an output feature map of a CNN layer [*, ch, T, F]
    :param gamma: [*, ch, ch]
    :param beta: [*, ch]
    :return: gamma * x + beta
    """
    x = x.unsqueeze(-4)
    gammas = gammas.unsqueeze(-1).unsqueeze(-1)

    pocm = [f.conv2d(x_, gamma_, beta_) for x_, gamma_, beta_ in zip(x, gammas, betas)]

    return torch.cat(pocm, dim=0)


def Pocm_Matmul(x, gammas, betas):
    """
    :param x: an output feature map of a CNN layer [*, ch, T, F]
    :param gamma: [*, ch, ch]
    :param beta: [*, ch]
    :return: gamma * x + beta
    """
    x = x.transpose(-1, -3)  # [*, F, T, ch]
    gammas = gammas.unsqueeze(-3)  # [*, 1, ch, ch]

    pocm = torch.matmul(x, gammas) + betas.unsqueeze(-2).unsqueeze(-3)

    return pocm.transpose(-1, -3)

# ---------------------------- Weight Initialization ---------------------------- #
def init_weights_functional(module, activation='default'):
    if isinstance(activation, nn.ReLU):
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.kaiming_normal_(param, nonlinearity='relu')

    elif activation == 'relu':
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.kaiming_normal_(param, nonlinearity='relu')

    elif isinstance(activation, nn.LeakyReLU):
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.kaiming_normal_(param, nonlinearity='leaky_relu')

    elif activation == 'leaky_relu':
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.kaiming_normal_(param, nonlinearity='leaky_relu')

    elif isinstance(activation, nn.Sigmoid):
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)

    elif activation == 'sigmoid':
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)

    elif isinstance(activation, nn.Tanh):
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)

    elif activation == 'tanh':
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)

    else:
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)
                
# ---------------------------- Fourier ---------------------------- #
def get_trim_length(hop_length, min_trim=5000):
    trim_per_hop = math.ceil(min_trim / hop_length)
    trim_length = trim_per_hop * hop_length
    assert trim_per_hop > 1
    return trim_length

def complex_norm(spec_complex, power=1.0):
    return spec_complex.pow(2.).sum(-1).pow(0.5 * power)


def complex_angle(spec_complex):
    return torch.atan2(spec_complex[..., 1], spec_complex[..., 0])


def mag_phase_to_complex(mag, phase, power=1.0):
    """
    input_signal: mag(*, N, T) , phase(*, N, T), power is optional
    output: *, N, T, 2
    """
    mag_power_1 = mag.pow(1 / power)
    spec_real = mag_power_1 * torch.cos(phase)
    spec_imag = mag_power_1 * torch.sin(phase)
    spec_complex = torch.stack([spec_real, spec_imag], dim=-1)
    return spec_complex


class STFT(nn.Module):
    
    def __init__(self, n_fft, hop_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.nn.Parameter(torch.hann_window(n_fft))
        # self.freeze()

    def forward(self, input_signal):
        return self.to_spec_complex(input_signal)

    def to_spec_complex(self, input_signal: torch.Tensor):
        """
        input_signal: *, signal
        output: *, N, T, 2
        """
        # if input_signal.dtype != self.window.dtype or input_signal.device != self.window.device :
        #     self.window = torch.as_tensor(self.window, dtype=input_signal.dtype, device=input_signal.device)
        # else:
        #     window = self.window

        return torch.stft(input_signal, self.n_fft, self.hop_length, window=self.window)

    def to_mag(self, input_signal, power=1.0):
        """
        input_signal: input signal (*, signal), power is optional
        output: *, N, T
        """
        spec_complex = self.to_spec_complex(input_signal)
        return complex_norm(spec_complex, power)

    def to_phase(self, input_signal):
        """
        input_signal: *, signal
        output: *, N, T
        """
        spec_complex = self.to_spec_complex(input_signal)
        return complex_angle(spec_complex)

    def to_mag_phase(self, input_signal, power=1.0):
        """
        input_signal: input signal (*, signal), power is optional
        output: tuple (mag(*, N, T) , phase(*, N, T))
        """
        spec_complex = self.to_spec_complex(input_signal)
        return complex_norm(spec_complex, power), complex_angle(spec_complex)

    def restore_complex(self, spec_complex):
        """
        input_signal:  *, N, T, 2
        output: *, signal
        """
        if spec_complex.dtype != self.window.dtype:
            window = torch.as_tensor(self.window, dtype=spec_complex.dtype)
        else:
            window = self.window

        if spec_complex.device != self.window.device:
            window = window.to(spec_complex.device)
        else:
            window = self.window

        return torch.istft(spec_complex, self.n_fft, self.hop_length, window=window)

    def restore_mag_phase(self, mag, phase, power=1.):
        """
        input_signal: mag(*, N, T), phase(*, N, T), power is optional
        output: *, signal
        """
        spec_complex = mag_phase_to_complex(mag, phase, power)
        return self.restore_complex(spec_complex)
    

class multi_channeled_STFT(nn.Module):
    def __init__(self, n_fft, hop_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.stft = STFT(n_fft, hop_length)

    def forward(self, input_signal):
        return self.to_spec_complex(input_signal)

    def to_spec_complex(self, input_signal): # -> Tensor
        """
        input_signal: *, signal, ch
        output: *, N, T, 2, ch
        """
        num_channels = input_signal.shape[-1]
        spec_complex_ch = [self.stft.to_spec_complex(input_signal[..., ch_idx])
                           for ch_idx in range(num_channels)]
        return torch.stack(spec_complex_ch, dim=-1)

    def to_mag(self, input_signal, power=1.0):
        """
        input_signal: input signal (*, signal, ch), power is optional
        output: *, N, T, ch
        """
        num_channels = input_signal.shape[-1]
        mag_ch = [self.stft.to_mag(input_signal[..., ch_idx], power)
                  for ch_idx in range(num_channels)]
        return torch.stack(mag_ch, dim=-1)

    def to_phase(self, input_signal):
        """
        input_signal: *, signal, ch
        output: *, N, T, ch
        """
        num_channels = input_signal.shape[-1]
        phase_ch = [self.stft.to_phase(input_signal[..., ch_idx])
                    for ch_idx in range(num_channels)]
        return torch.stack(phase_ch, dim=-1)

    def to_mag_phase(self, input_signal, power=1.0):
        """
        input_signal: input signal (*, signal, ch), power is optional
        output: tuple (mag(*, N, T, ch) , phase(*, N, T, ch))
        """
        num_channels = input_signal.shape[-1]
        mag_ch = [self.stft.to_mag(input_signal[..., ch_idx], power)
                  for ch_idx in range(num_channels)]
        phase_ch = [self.stft.to_phase(input_signal[..., ch_idx])
                    for ch_idx in range(num_channels)]
        return torch.stack(mag_ch, dim=-1), torch.stack(phase_ch, dim=-1)

    def restore_complex(self, spec_complex):
        """
        input_signal:  *, N, T, 2, ch
        output: *, signal, ch
        """
        num_channels = spec_complex.shape[-1]
        signal_ch = [self.stft.restore_complex(spec_complex[..., ch_idx])
                     for ch_idx in range(num_channels)]
        return torch.stack(signal_ch, dim=-1)

    def restore_mag_phase(self, mag, phase, power=1.):
        """
        input_signal: mag(*, N, T, ch), phase(*, N, T, ch), power is optional
        output: *, signal
        """
        num_channels = mag.shape[-1]
        signal_ch = [self.stft.restore_mag_phase(mag[..., ch_idx], phase[..., ch_idx], power)
                     for ch_idx in range(num_channels)]
        return torch.stack(signal_ch, dim=-1)