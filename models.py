from argparse import ArgumentParser
from typing import Tuple
from warnings import warn

import torch
import torch.nn as nn
from torch import Tensor

# from lasaft.data.musdb_wrapper import SingleTrackSet
# from lasaft.source_separation.conditioned.separation_framework import Spectrogram_based
from helper_functions import get_activation_by_name, string_to_list

from control_module import pocm_control_model, dense_control_block
from helper_functions import Pocm_Matmul, Pocm_naive
from building_blocks import TFC_LaSAFT
# from lasaft.source_separation.conditioned.loss_functions import get_conditional_loss
# from lasaft.utils import functions


class Dense_CUNet(nn.Module):

    def __init__(self,
                 n_fft,
                 input_channels, internal_channels,
                 n_blocks, n_internal_layers,
                 mk_block_f, mk_ds_f, mk_us_f,
                 first_conv_activation, last_activation,
                 t_down_layers, f_down_layers,
                 control_vector_type, control_input_dim, embedding_dim, condition_to
                 ):

        first_conv_activation = get_activation_by_name(first_conv_activation)
        last_activation = get_activation_by_name(last_activation)

        super(Dense_CUNet, self).__init__()

        '''num_block should be an odd integer'''
        assert n_blocks % 2 == 1

        dim_f, t_down_layers, f_down_layers = self.mk_overall_structure(n_fft, internal_channels, input_channels,
                                                                        n_blocks,
                                                                        n_internal_layers, last_activation,
                                                                        first_conv_activation,
                                                                        t_down_layers, f_down_layers)

        self.mk_blocks(dim_f, internal_channels, mk_block_f, mk_ds_f, mk_us_f, t_down_layers)

        #########################
        # Conditional Mechanism #
        #########################
        assert control_vector_type in ['one_hot_mode', 'embedding']
        if control_vector_type == 'one_hot_mode':
            if control_input_dim != embedding_dim:
                warn('in one_hot_mode, embedding_dim should be the same as num_targets. auto correction')
                embedding_dim = control_input_dim

                with torch.no_grad():
                    one_hot_weight = torch.zeros((control_input_dim, embedding_dim))
                    for i in range(control_input_dim):
                        one_hot_weight[i, i] = 1.

                    self.embedding = nn.Embedding(control_input_dim, embedding_dim, _weight=one_hot_weight)
                    self.embedding.weight.requires_grad = True
                    # Note: does it mean that the embedding is also going to be learned?
        elif control_vector_type == 'embedding':
            self.embedding = nn.Embedding(control_input_dim, embedding_dim)

        # Where to condition
        assert condition_to in ['encoder', 'decoder', 'full']
        self.is_encoder_conditioned = self.is_middle_conditioned = self.is_decoder_conditioned = False
        if condition_to == 'encoder':
            self.is_encoder_conditioned = True
        elif condition_to == 'decoder':
            self.is_decoder_conditioned = True
        elif condition_to == 'full':
            self.is_encoder_conditioned = self.is_middle_conditioned = self.is_decoder_conditioned = True
        else:
            raise NotImplementedError

        self.activation = self.last_conv[-1]

    def mk_blocks(self, dim_f, internal_channels, mk_block_f, mk_ds_f, mk_us_f, t_down_layers):
        # Note: A function called once in the class constructor. Seems to make encoder/decoder blocks
        f = dim_f
        for i in range(self.n):
            self.encoders.append(mk_block_f(internal_channels, internal_channels, f))
            # Note: mk_block_f is a parameter function making encoder block? Which block? 
            ds_layer, f = mk_ds_f(internal_channels, i, f, t_down_layers)
            # Note: make downsampling layer?
            self.downsamplings.append(ds_layer)
        self.mid_block = mk_block_f(internal_channels, internal_channels, f)
        for i in range(self.n):
            us_layer, f = mk_us_f(internal_channels, i, f, self.n, t_down_layers)
            self.upsamplings.append(us_layer)
            self.decoders.append(mk_block_f(2 * internal_channels, internal_channels, f))
            # Here concatenation of downsampling and upsampling layers is conducted
        
        # Note: seems that the internal channel doesn't change throughout the network

    def mk_overall_structure(self, n_fft, internal_channels, input_channels, n_blocks, n_internal_layers,
                             last_activation, first_conv_activation, t_down_layers, f_down_layers):
        # Note: a function called in the constructor *before mk_blocks*
        dim_f = n_fft // 2
        input_channels = input_channels
        self.first_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=internal_channels,
                kernel_size=(1, 2),
                stride=1
            ),
            nn.BatchNorm2d(internal_channels),
            first_conv_activation(),
        )
        # Note: first 1x2 convolution before encoder. No paddings?
        self.encoders = nn.ModuleList()
        self.downsamplings = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upsamplings = nn.ModuleList()
        self.last_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=internal_channels,
                out_channels=input_channels,
                kernel_size=(1, 2),
                stride=1,
                padding=(0, 1)
            ),
            last_activation()
        )
        # Note: internal channel number doesn't change over the network. In the output, the input channel number is 
        # restored
        self.n = n_blocks // 2
        # Note: n_blocks should be the total blocks in encoder and decoder?
        if t_down_layers is None:
            t_down_layers = list(range(self.n))
        elif n_internal_layers == 'None':
            t_down_layers = list(range(self.n))
        else:
            t_down_layers = string_to_list(t_down_layers)
        if f_down_layers is None:
            f_down_layers = list(range(self.n))
        elif n_internal_layers == 'None':
            f_down_layers = list(range(self.n))
        else:
            f_down_layers = string_to_list(f_down_layers)
        return dim_f, t_down_layers, f_down_layers
    
    # Note: what are t_down_layers and f_down_layers in actual implementation?

    def forward(self, input_spec, input_condition):

        condition_embedding = self.embedding(input_condition)
        gammas, betas = self.condition_generator(condition_embedding)
        # Note: where is condition_generator?

        x = self.first_conv(input_spec)
        encoding_outputs = []

        for encoder, downsampling, gamma, beta in zip(self.encoders, self.downsamplings, gammas, betas):
        # Note: here zip() implies that gammas and betas have the same dimension (n) as encoders and downsamplings
            x = encoder(x)
            x = self.film(x, gamma, beta)
            # Note: where is film?
            encoding_outputs.append(x)
            # Note: Output of each layer is kept so that it can be concatenated with decoder layer's output
            # later on
            x = downsampling(x)

        x = self.mid_block(x)

        for i in range(self.n):
            x = self.upsamplings[i](x)
            x = torch.cat((x, encoding_outputs[-i - 1]), 1)
            x = self.decoders[i](x)

        return self.last_conv(x)
    

class DenseCUNet_GPoCM(Dense_CUNet):
    def __init__(self,
                 n_fft,
                 input_channels, internal_channels,
                 n_blocks, n_internal_layers,
                 mk_block_f, mk_ds_f, mk_us_f,
                 first_conv_activation, last_activation,
                 t_down_layers, f_down_layers,
                 # Conditional Mechanism #
                 control_vector_type, control_input_dim, embedding_dim, condition_to,
                 control_type, control_n_layer, pocm_type, pocm_norm
                 ):
        # Note: similar to DenseCUNet_FiLM, the parameters are the same as Dense_CUNet except for the last five

        super(DenseCUNet_GPoCM, self).__init__(
            n_fft,
            input_channels, internal_channels,
            n_blocks, n_internal_layers,
            mk_block_f, mk_ds_f, mk_us_f,
            first_conv_activation, last_activation,
            t_down_layers, f_down_layers,
            control_vector_type, control_input_dim, embedding_dim, condition_to
        )

        # select PoCM implementation:
        # both yield the same outputs, but 'matmul' is faster with gpus since it does not use loops.
        assert pocm_type in ['naive', 'matmul']
        self.pocm = Pocm_naive if pocm_type == 'naive' else Pocm_Matmul

        # Select normalization methods for PoCM
        assert pocm_norm in [None, 'batch_norm']

        # Make condition generator
        if control_type == "dense":
            self.condition_generator = pocm_control_model(
                dense_control_block(embedding_dim, control_n_layer),
                n_blocks, internal_channels,
                pocm_to=condition_to,
                pocm_norm=pocm_norm
            )
        else:
            raise NotImplementedError

        self.activation = self.last_conv[-1]

    def forward(self, input_spec, input_condition):

        condition_embedding = self.embedding(input_condition)
        gammas, betas = self.condition_generator(condition_embedding)

        x = self.first_conv(input_spec)
        encoding_outputs = []

        gammas_encoder, gammas_middle, gammas_decoder = gammas
        betas_encoder, betas_middle, betas_decoder = betas

        for i in range(self.n):
            x = self.encoders[i](x)
            if self.is_encoder_conditioned:
                g = self.pocm(x, gammas_encoder[i], betas_encoder[i]).sigmoid()
                x = g * x
            # Note: the implementation of control for pocm is different from that for film.
            encoding_outputs.append(x)
            x = self.downsamplings[i](x)

        x = self.mid_block(x)
        if self.is_middle_conditioned:
            g = self.pocm(x, gammas_middle, betas_middle).sigmoid()
            x = g * x

        for i in range(self.n):
            x = self.upsamplings[i](x)
            x = torch.cat((x, encoding_outputs[-i - 1]), 1)
            x = self.decoders[i](x)
            if self.is_decoder_conditioned:
                g = self.pocm(x, gammas_decoder[i], betas_decoder[i]).sigmoid()
                x = g * x
        return self.last_conv(x)
    
    
class DCUN_TFC_GPoCM_LaSAFT(DenseCUNet_GPoCM):
    
    def __init__(self,
                 n_fft,
                 n_blocks, input_channels, internal_channels, n_internal_layers,
                 first_conv_activation, last_activation,
                 t_down_layers, f_down_layers,
                 kernel_size_t, kernel_size_f,
                 bn_factor, min_bn_units,
                 tfc_tdf_bias,
                 tfc_tdf_activation,
                 num_tdfs, dk,
                 control_vector_type, control_input_dim, embedding_dim, condition_to,
                 control_type, control_n_layer, pocm_type, pocm_norm
                 ):

        tfc_tdf_activation = get_activation_by_name(tfc_tdf_activation)

        def mk_tfc_lasaft(in_channels, internal_channels, f):
            return TFC_LaSAFT(in_channels, n_internal_layers, internal_channels,
                              kernel_size_t, kernel_size_f, f,
                              bn_factor, min_bn_units,
                              tfc_tdf_bias,
                              tfc_tdf_activation,
                              embedding_dim, num_tdfs, dk)

        def mk_ds(internal_channels, i, f, t_down_layers):
            # Note: i is the current encoder block
            if t_down_layers is None:
                scale = (2, 2)
                # Note: if t_down_layers is not defined, each block would conduct downsampling
            else:
                scale = (2, 2) if i in t_down_layers else (1, 2)
                # (t, f)
            ds = nn.Sequential(
                nn.Conv2d(in_channels=internal_channels, out_channels=internal_channels,
                          kernel_size=scale, stride=scale),
                nn.BatchNorm2d(internal_channels)
            )
            return ds, f // scale[-1]

        def mk_us(internal_channels, i, f, n, t_down_layers):
            if t_down_layers is None:
                scale = (2, 2)
            else:
                scale = (2, 2) if i in [n - 1 - s for s in t_down_layers] else (1, 2)

            us = nn.Sequential(
                nn.ConvTranspose2d(in_channels=internal_channels, out_channels=internal_channels,
                                   kernel_size=scale, stride=scale),
                nn.BatchNorm2d(internal_channels)
            )
            return us, f * scale[-1]

        super(DCUN_TFC_GPoCM_LaSAFT, self).__init__(
            n_fft,
            input_channels, internal_channels,
            n_blocks, n_internal_layers,
            mk_tfc_lasaft, mk_ds, mk_us,
            first_conv_activation, last_activation,
            t_down_layers, f_down_layers,
            # Conditional Mechanism #
            control_vector_type, control_input_dim, embedding_dim, condition_to,
            control_type, control_n_layer, pocm_type, pocm_norm
        )

    def forward(self, input_spec, input_condition):

        condition_embedding = self.embedding(input_condition)
        gammas, betas = self.condition_generator(condition_embedding)

        x = self.first_conv(input_spec)
        encoding_outputs = []

        gammas_encoder, gammas_middle, gammas_decoder = gammas
        betas_encoder, betas_middle, betas_decoder = betas

        for i in range(self.n):
            x = self.encoders[i].tfc(x)
            # Note: here the channel number of x is changed to gr
            if self.is_encoder_conditioned:
                g = self.pocm(x, gammas_encoder[i], betas_encoder[i]).sigmoid()
                x = g * x
            x = x + self.encoders[i].lasaft(x, condition_embedding)
            encoding_outputs.append(x)
            x = self.downsamplings[i](x)

        x = self.mid_block.tfc(x)
        if self.is_middle_conditioned:
            g = self.pocm(x, gammas_middle, betas_middle).sigmoid()
            x = g * x
        x = x + self.mid_block.lasaft(x, condition_embedding)

        for i in range(self.n):
            x = self.upsamplings[i](x)
            x = torch.cat((x, encoding_outputs[-i - 1]), 1)
            x = self.decoders[i].tfc(x)
            if self.is_decoder_conditioned:
                g = self.pocm(x, gammas_decoder[i], betas_decoder[i]).sigmoid()
                x = g * x
            x = x + self.decoders[i].lasaft(x, condition_embedding)
        return self.last_conv(x)