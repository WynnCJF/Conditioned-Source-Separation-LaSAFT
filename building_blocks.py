import torch
import torch.nn as nn
import math

class TFC(nn.Module):
    """ [B, in_channels, T, F] => [B, gr, T, F] """

    def __init__(self, in_channels, num_layers, gr, kt, kf, activation):
        """
        in_channels: number of input channels
        num_layers: number of densely connected conv layers
        gr: growth rate
        kt: kernel size of the temporal axis.
        kf: kernel size of the freq. axis
        activation: activation function
        """
        super(TFC, self).__init__()

        c = in_channels
        self.H = nn.ModuleList()
        for i in range(num_layers):
            self.H.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=gr, kernel_size=(kf, kt), stride=1,
                              padding=(kt // 2, kf // 2)),
                    nn.BatchNorm2d(gr),
                    activation(),
                )
            )
            c += gr
            # Note: the value of c is changing in the layer, which corresponds to the concatenation
            #       of x and x_ in forward() 

        self.activation = self.H[-1][-1]

    def forward(self, x):
        """ [B, in_channels, T, F] => [B, gr, T, F] """
        x_ = self.H[0](x)
        # x_: gr
        for h in self.H[1:]:
            x = torch.cat((x_, x), 1)
            # x: c + n*gr
            x_ = h(x)
            # x_: gr

        return x_


class TDF(nn.Module):
    """ [B, in_channels, T, F] => [B, gr, T, F] """

    def __init__(self, channels, f, bn_factor=16, bias=False, min_bn_units=16, activation=nn.ReLU):

        """
        channels: # channels
        f: num of frequency bins
        bn_factor: bottleneck factor. if None: single layer. else: MLP that maps f => f//bn_factor => f
        bias: bias setting of linear layers
        activation: activation function
        """

        super(TDF, self).__init__()
        if bn_factor is None:
            self.tdf = nn.Sequential(
                nn.Linear(f, f, bias),
                nn.BatchNorm2d(channels),
                activation()
            )

        else:
            bn_units = max(f // bn_factor, min_bn_units)
            self.bn_units = bn_units
            self.tdf = nn.Sequential(
                nn.Linear(f, bn_units, bias),
                nn.BatchNorm2d(channels),
                activation(),
                nn.Linear(bn_units, f, bias),
                nn.BatchNorm2d(channels),
                activation()
            )
            # Note: what's the point of shrinking and expanding the frequency bins?
            # Note: there is no specification of axis in nn.Linear. Is it because the default is set to
            #       last dimension?

    def forward(self, x):
        return self.tdf(x)
    
    
class TFC_TDF(nn.Module):
    def __init__(self, in_channels, num_layers, gr, kt, kf, f, bn_factor=16, min_bn_units=16, bias=False,
                 activation=nn.ReLU):
        """
        in_channels: number of input channels
        num_layers: number of densely connected conv layers
        gr: growth rate
        kt: kernel size of the temporal axis.
        kf: kernel size of the freq. axis
        f: num of frequency bins

        below are params for TDF
        bn_factor: bottleneck factor. if None: single layer. else: MLP that maps f => f//bn_factor => f
        bias: bias setting of linear layers

        activation: activation function
        """

        super(TFC_TDF, self).__init__()
        self.tfc = TFC(in_channels, num_layers, gr, kt, kf, activation)
        self.tdf = TDF(gr, f, bn_factor, bias, min_bn_units, activation)
        # Note: the input channel number of tdf is gr. When will the channel number be set back to 
        #       internal_channel?
        self.activation = self.tdf.tdf[-1]

    def forward(self, x):
        x = self.tfc(x)
        return x + self.tdf(x)


class TDF_f1_to_f2(nn.Module):
    """ [B, in_channels, T, F] => [B, gr, T, F] """

    def __init__(self, channels, f1, f2, bn_factor=16, bias=False, min_bn_units=16, activation=nn.ReLU):

        """
        channels:  # channels
        f1: num of frequency bins (input)
        f2: num of frequency bins (output)
        bn_factor: bottleneck factor. if None: single layer. else: MLP that maps f => f//bn_factor => f
        bias: bias setting of linear layers
        activation: activation function
        """

        super(TDF_f1_to_f2, self).__init__()

        self.num_target_f = f2

        if bn_factor is None:
            self.tdf = nn.Sequential(
                nn.Linear(f1, f2, bias),
                nn.BatchNorm2d(channels),
                activation()
            )

        else:
            bn_units = max(f2 // bn_factor, min_bn_units)
            self.tdf = nn.Sequential(
                nn.Linear(f1, bn_units, bias),
                nn.BatchNorm2d(channels),
                activation(),
                nn.Linear(bn_units, f2, bias),
                nn.BatchNorm2d(channels),
                activation()
            )
            # Note: the number of frequency bins are changed


    def forward(self, x):
        return self.tdf(x)
    
    
class TFC_LaSAFT(nn.Module):
    def __init__(self, in_channels, num_layers, gr, kt, kf, f, bn_factor, min_bn_units, bias,
                 activation, condition_dim, num_tdfs, dk):
        super(TFC_LaSAFT, self).__init__()
        self.dk_sqrt = math.sqrt(dk)
        # Note: dk is the scaling factor in attention
        self.num_tdfs = num_tdfs
        self.tfc = TFC(in_channels, num_layers, gr, kt, kf, activation)
        self.tdfs = TDF_f1_to_f2(gr, f, f * num_tdfs, bn_factor, bias, min_bn_units, activation)
        # Note: duplicate I_L copies of V. Seems that the frequency bins are not simply copied, but
        #       expanded by the linear layer from f to f * I_L
        self.keys = nn.Parameter(torch.randn(dk, num_tdfs), requires_grad=True)
        # Note: what does this dimension (dk, num_tdfs) mean?
        #       num_tdfs seems to correspond with I_L in the paper, indicating the latent instruments
        self.linear_query = nn.Linear(condition_dim, dk)
        # Note: is the condition_dim here related to the control module?
        #       condition_dim should correspond to E in paper
        self.activation = self.tdfs.tdf[-1]

    def forward(self, x, c):
        x = self.tfc(x)
        return x + self.lasaft(x, c)

    def lasaft(self, x, c):
        query = self.linear_query(c)
        # Note: what is c? guess it is a matrix, and the final dimension is equivalent to condition_dim
        #       keys: (dk, I_L), query: (N, dk)
        qk = torch.matmul(query, self.keys) / self.dk_sqrt
        # Note: qk: (N, I_L) not sure
        value = (self.tdfs(x)).view(list(x.shape)[:-1] + [-1, self.num_tdfs])
        # Note: V: (..., F, I_L)
        #       x: (N, gr, T, F) -> tdfs(x): (N, gr, T, F*I_L) -> value: (N, gr, T, F, I_L)
        att = qk.softmax(-1)
        # Note: what is the dimension of att? Guess (N, I_L)
        return torch.matmul(value, att.unsqueeze(-2).unsqueeze(-3).unsqueeze(-1)).squeeze(-1)
        # Note: att: (N, I_L) -> (N, 1, I_L) -> (N, 1, 1, I_L) -> (N, 1, 1, I_L, 1)
        #       value: (N, gr, T, F, I_L)
        #       result: (N, gr, T, F, 1) -> (N, gr, T, F)
        #       In the implementation of LaSAFT, gr is defined directly as internal_channel in function parameters