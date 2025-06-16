import torch
import torch.nn as nn
import torch.nn.functional as F


#  Frequency Domain Loss 
class FrequencyLoss(nn.Module):
    def __init__(self):
        super(FrequencyLoss, self).__init__()

    def forward(self, pred, target):
        pred_fft = torch.fft.fft2(pred, norm='ortho')
        target_fft = torch.fft.fft2(target, norm='ortho')
        
        return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))

frequency_loss = FrequencyLoss()

# Loss Selection Function
def compute_loss(loss_type):
    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'mae':
        return nn.L1Loss()  # Mean Absolute Error:
    elif loss_type == 'frequency':
        return frequency_loss # Frequency-domain loss
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")