import torch.nn as nn

class dense_control_block(nn.Module):

    def __init__(self, input_dim, num_layer, activation=nn.ReLU, scale=2, scale_type="exp"):
        # Note: here input_dim should be the dimension of input control embedding. Not sure though.
        super(dense_control_block, self).__init__()
        self.input_dim = input_dim
        self.num_layer = num_layer
        self.activation = activation

        linear_list = []
        if scale_type == 'exp':
            dims = [input_dim * (scale ** i) for i in range(num_layer)]
        elif scale_type == 'mul':
            dims = [input_dim + input_dim * (scale * i) for i in range(num_layer)]
        # Note: what is this dim for?
        
        print(num_layer)

        for i, (in_features, out_features) in enumerate(zip(dims[:-1], dims[1:])):
            # Note: in_features and out_features are respectively index 0&1, 1&2, ..., n-2&n-1 of dims
            extra = i != 0
            linear_list.append(nn.Linear(in_features, out_features))
            linear_list.append(activation())

            if extra:
                linear_list.append(nn.Dropout())
                linear_list.append(nn.BatchNorm1d(out_features))
                
            print(i, ": in_features:", in_features, " out_features:", out_features)

        self.linear = nn.Sequential(*linear_list)
        # Note: a series of linear layer that convert the constrol embedding from input_dim to output_dim
        self.last_dim = dims[-1]

    def forward(self, x_condition):
        # print(self.linear[-1].s)
        # print(x_condition.shape)
        return self.linear(x_condition)


class pocm_control_model(nn.Module):
    def __init__(self, dense_control_block, n_blocks, internal_channels, gamma_activation=nn.Identity,
                 beta_activation=nn.Identity, pocm_to='full', pocm_norm=None):
        super(pocm_control_model, self).__init__()

        self.dense_control_block = dense_control_block
        self.n_blocks = n_blocks
        self.c = internal_channels
        self.gamma_activation = gamma_activation()
        self.beta_activation = beta_activation()

        if pocm_to == 'full':
            self.full, self.encoder_only, self.decoder_only = True, False, False
            num_target_block = n_blocks
        elif pocm_to == 'encoder':
            self.full, self.encoder_only, self.decoder_only = False, True, False
            num_target_block = n_blocks // 2
        elif pocm_to == 'decoder':
            self.full, self.encoder_only, self.decoder_only = False, False, True
            num_target_block = n_blocks // 2
        else:
            raise NotImplementedError
        # Note: still need to confirm what is the controlling region of LaSAFT+GPoCM

        assert pocm_norm in [None, 'batch_norm', 'full_batch_norm', 'lstm_like']

        def mk_norm(pocm_norm, type):
            if pocm_norm is None:
                return nn.Identity()
            # Note: seems pocm_norm is default to None. Is it the case for LaSAFT?
            elif 'batch_norm' in pocm_norm:
                if type == 'gamma':
                    return nn.BatchNorm1d(num_target_block * (internal_channels ** 2), affine=False)
                elif type == 'beta':
                    return nn.BatchNorm1d(num_target_block * internal_channels, affine=False)
            elif 'lstm_like' == pocm_norm:
                if type == 'gamma':
                    return nn.BatchNorm1d(num_target_block * (internal_channels ** 2), affine=False)
                elif type == 'beta':
                    return nn.BatchNorm1d(num_target_block * internal_channels, affine=False)
            # Note: why nn.BatchNorm1d is used here instead of Batchnorm2d? Guess because here batch normalization
            #       is applied to control vectors. But why the parameter seems to represent the dimension of vector
            #       instead of channel numeber as in 2d?
                
            # Note: each target block needs a set of gamma and beta, which counts for num_target_block;
            #       for gamma (omega in the paper), since each channel needs to be convolved c times, 
            #       this sums up to c^2; correspondingly, beta needs cs

        self.linear_gamma = nn.Sequential(
            nn.Linear(dense_control_block.last_dim, num_target_block * (internal_channels ** 2)),
            # Convert the output of dense_control_block to the dimension of gamma
            self.gamma_activation,
            mk_norm(pocm_norm, 'gamma')
        )
        self.linear_beta = nn.Sequential(
            nn.Linear(dense_control_block.last_dim, num_target_block * internal_channels),
            self.beta_activation,
            mk_norm(pocm_norm, 'beta')
        )

    def forward(self, x):
        # Note: what is x? one-hot vector or embedding vector? Should be the embedding vector
        x = self.dense_control_block(x)
        m = self.n_blocks // 2

        if self.full:
            gammas = self.gamma_split(self.linear_gamma(x), 0, self.n_blocks)
            betas = self.beta_split(self.linear_beta(x), 0, self.n_blocks)

            g_encoder, g_middle, g_decoder = gammas[:m], gammas[m], gammas[m + 1:]
            gammas = [g_encoder, g_middle, g_decoder]

            b_encoder, b_middle, b_decoder = betas[:m], betas[m], betas[m + 1:]
            betas = [b_encoder, b_middle, b_decoder]

        elif self.encoder_only or self.decoder_only:
            gammas = self.gamma_split(self.linear_gamma(x), 0, m)
            betas = self.beta_split(self.linear_beta(x), 0, m)

            if self.encoder_only:
                gammas = [gammas, None, None]
                betas = [betas, None, None]
            else:
                gammas = [None, None, gammas]
                betas = [None, None, betas]

        else:
            raise NotImplementedError

        return gammas, betas

    def gamma_split(self, tensor, start_idx, end_idx):
        # Note: here tensor is a control vector with length as number of gammas
        tensor_shape = list(tensor.shape[:-1]) + [self.c, self.c]
        return [tensor[..., layer * self.c * self.c: (layer + 1) * self.c * self.c].view(tensor_shape)
                for layer in range(start_idx, end_idx)]
        # Note: change the 1D gamma vector to 2D (c x c) for each single block

    def beta_split(self, tensor, start_idx, end_idx):
        return [tensor[..., layer * self.c: (layer + 1) * self.c]
                for layer in range(start_idx, end_idx)]