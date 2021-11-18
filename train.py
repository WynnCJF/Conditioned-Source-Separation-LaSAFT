import torch
# Dataset
from dataset import MusdbTrainSet, MusdbValidSetWithGT, MusdbTestSetWithGT
from torch.utils.data import DataLoader
# Model
from models import DCUN_TFC_GPoCM_LaSAFT

# Helper function
from helper_functions import *
import matplotlib.pyplot as plt
import time

# Loss function
from loss_functions import SpecMSELoss


# -------------------------------- Hyperparameters -------------------------------- #
MUSDBROOT = "musdb18/musdb18"
TRAIN_BATCH_SIZE = 4
LR = 0.001
EPOCH = 10
TRAIN_RECORD = 100
DEVICE_NO = 0

SAVE_MODEL_PREFIX = "train_net_params_3_27"
DATE = " - 3.27"

N_FFT = 2048
HOP_LENGTH = 1024

N_BLOCKS = 7
INPUT_CHANNELS = 4
INTERNAL_CHANNELS = 24
DENSE_CONTROL_INTERNAL_LAYERS = 5
FIRST_CONV_ACTIVATION = "relu"
LAST_ACTIVATION = "identity"
T_DOWN_LAYERS = None
F_DOWN_LAYERS = None
TFC_T_KERNEL = 3
TFC_F_KERNEL = 3
BN_FACTOR = 16
MIN_BN_UNITS = 16
TFC_TDF_BIAS = False
TFC_TDF_ACTIVATION = "relu"
NUM_IL = 6
DK = 32
CONTROL_VECTOR_TYPE = "embedding"
CONTROL_INPUT_DIM = 4
EMBEDDING_DIM = 32
CONDITION_TO = "decoder"
POCM_CONTROL_TYPE = "dense"
POCM_CONTROL_N_LAYER = 4
POCM_TYPE = "matmul"
POCM_NORM = "batch_norm"


device = torch.device("cuda:" + str(DEVICE_NO) if torch.cuda.is_available() else "cpu")

# -------------------------------- Data -------------------------------- #
train_dataset = MusdbTrainSet(musdb_root=MUSDBROOT)
train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False)

print("Training dataset loaded. Data size:", len(train_dataset))

# Delete This
# print(train_dataset[0])


# -------------------------------- Train Step -------------------------------- #
def train():
    running_loss = 0.0
    losses = []
    batches = []
    
    for i, (mixture, target, control_condition) in enumerate(train_dataloader, 1):
        # Turn the mixture signal to complex spectrogram
        mixture_spec = stft.to_spec_complex(mixture)
        mixture_spec = torch.flatten(mixture_spec, start_dim=-2)
        mixture_spec = mixture_spec.transpose(-1, -3)
        mixture_spec = mixture_spec.to(device)

        # Turn the target stem signal to complex spectrogram
        target_spec = stft.to_spec_complex(target)
        target_spec = torch.flatten(target_spec, start_dim=-2)
        target_spec = target_spec.transpose(-1, -3)
        target_spec = target_spec.to(device)
        
        control_condition = control_condition.to(device)
        
        print("data loaded. ", time.time())
        
        pred_target = model(mixture_spec, control_condition)
        
        optimizer.zero_grad()
        
        # label = label.squeeze()
        
        loss = criterion(pred_target, target_spec)
        
        loss.backward()
        optimizer.step()
        
        print("iteration complete ", time.time())
        
        running_loss = running_loss + loss.item()
        
        if i % TRAIN_RECORD == 0:
            print('[' + str(i * TRAIN_BATCH_SIZE) + '/' + str(len(train_dataset)) + "]")
            loss_stat = running_loss / (TRAIN_BATCH_SIZE * TRAIN_RECORD)
            running_loss = 0.0
            print('Loss: ' + str(loss_stat))
            losses.append(loss_stat)
            batches.append(i * TRAIN_BATCH_SIZE)
        
    return batches, losses


# -------------------------------- Main Function -------------------------------- #
if __name__ == '__main__':
    stft = multi_channeled_STFT(N_FFT, HOP_LENGTH)
    model = DCUN_TFC_GPoCM_LaSAFT(
        n_fft=N_FFT,
        n_blocks=N_BLOCKS,
        input_channels=INPUT_CHANNELS,
        internal_channels=INPUT_CHANNELS,
        n_internal_layers=DENSE_CONTROL_INTERNAL_LAYERS,
        first_conv_activation=FIRST_CONV_ACTIVATION,
        last_activation=LAST_ACTIVATION,
        t_down_layers=T_DOWN_LAYERS,
        f_down_layers=F_DOWN_LAYERS,
        kernel_size_t=TFC_T_KERNEL,
        kernel_size_f=TFC_F_KERNEL,
        bn_factor=BN_FACTOR,
        min_bn_units=MIN_BN_UNITS,
        tfc_tdf_bias=TFC_TDF_BIAS,
        tfc_tdf_activation=TFC_TDF_ACTIVATION,
        num_tdfs=NUM_IL,
        dk=DK,
        control_vector_type=CONTROL_VECTOR_TYPE,
        control_input_dim=CONTROL_INPUT_DIM,
        embedding_dim=EMBEDDING_DIM,
        condition_to=CONDITION_TO,
        control_type=POCM_CONTROL_TYPE,
        control_n_layer=POCM_CONTROL_N_LAYER,
        pocm_type=POCM_TYPE,
        pocm_norm=POCM_NORM
    )
    
    # model.load_state_dict(torch.load('.pkl'))
    model.to(device)
    model.train()
    
    criterion = SpecMSELoss()
    criterion.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    checkpoint_count = []
    loss_count = []
    for epoch in range(1,EPOCH+1):
        print("Epoch " + str(epoch) + " starts: ---------------------------------------------------")
        (train_batches, train_losses) = train()
        accumulate_train_batchs = [train_batch + len(train_dataset) * (epoch - 1) for train_batch in train_batches]
        checkpoint_count = checkpoint_count + accumulate_train_batchs
        loss_count = loss_count + train_losses
        
        epoch_stats = [accumulate_train_batchs, train_losses]
        epoch_stats_arr = np.array(epoch_stats)
        np.savetxt('loss_stats/epoch' + str(epoch) + DATE + '.csv', epoch_stats_arr, delimiter=',', fmt='%s')
        
        #if epoch == 5:
        torch.save(model.state_dict(), SAVE_MODEL_PREFIX + '_' + str(epoch) + 'epoch' + '.pkl')
    
    # torch.save(model,'net.pkl')
    # torch.save(model.state_dict(), 'train_adam_net_params_1_24(25 epoch).pkl')
    
    # Create loss plot
    plt.figure()
    plt.plot(checkpoint_count, loss_count)
    plt.xlabel("breakpoints")
    plt.ylabel("loss")
    plt.savefig("loss_fig.png")




'''
def to_spec(self, input_signal) -> torch.Tensor:
    if self.magnitude_based:
        return self.stft.to_mag(input_signal).transpose(-1, -3)
    else:
        spec_complex = self.stft.to_spec_complex(input_signal)  # *, N, T, 2, ch
        spec_complex = torch.flatten(spec_complex, start_dim=-2)  # *, N, T, 2ch
        return spec_complex.transpose(-1, -3)  # *, 2ch, T, N


def forward(self, input_signal, input_condition) -> torch.Tensor:
    input_spec = self.to_spec(input_signal)
    output_spec = self.spec2spec(input_spec, input_condition)

    if self.masking_based:
        output_spec = input_spec * output_spec

    return output_spec

def separate(self, input_signal, input_condition) -> torch.Tensor:
    
    phase = None
    if self.magnitude_based:
        mag, phase = self.stft.to_mag_phase(input_signal)
            
    else:
        spec_complex = self.stft.to_spec_complex(input_signal)  # *, N, T, 2, ch
        spec_complex = torch.flatten(spec_complex, start_dim=-2)  # *, N, T, 2ch
        input_spec = spec_complex.transpose(-1, -3)  # *, 2ch, T, N

    output_spec = self.spec2spec(input_spec, input_condition)

    if self.masking_based:
        output_spec = input_spec * output_spec
    else:
        pass  # Use the original output_spec

    output_spec = output_spec.transpose(-1, -3)

    if self.magnitude_based:
        restored = self.stft.restore_mag_phase(output_spec, phase)
    else:
        # output_spec: *, N, T, 2ch
        output_spec = output_spec.view(list(output_spec.shape)[:-1] + [2, -1])  # *, N, T, 2, ch
        restored = self.stft.restore_complex(output_spec)

    return restored

'''