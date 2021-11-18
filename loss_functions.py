import math
from abc import ABC, abstractmethod
from argparse import ArgumentParser

import torch
import torch.nn.functional as f
import torch.nn as nn


class Conditional_Loss(ABC):
    
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def compute(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)


class Conditional_Spectrogram_Loss(Conditional_Loss):
    
    def __init__(self, mode, **kwargs):
        super().__init__()
        assert mode in ['l1', 'l2', 'mse']
        self.criterion = f.l1_loss if mode == 'l1' else f.mse_loss

    def compute(self, model, mixture_signal, condition, target_signal):
        target = model.to_spec(target_signal)
        target_hat = model(mixture_signal, condition)
        return self.criterion(target_hat, target)

    def compute_with_ca(self, model, mixture_signal, condition, target_signal, disable_ca):
        target = model.to_spec(target_signal)
        target_hat, mu, log_var = model.forward(mixture_signal, condition, disable_ca)
        return self.criterion(target_hat, target), mu, log_var
    
    
class SpecMSELoss(nn.Module):
    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.criterion = f.mse_loss
    
    def forward(self, pred_target, target):
        return self.criterion(pred_target, target)
    