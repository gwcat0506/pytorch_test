import torch.nn.functional as F
import torch.nn as nn

# mse_loss 추가 
def mse_loss(output, target):
    mse_loss = nn.MSELoss()
    return mse_loss(output.squeeze(1), target)
