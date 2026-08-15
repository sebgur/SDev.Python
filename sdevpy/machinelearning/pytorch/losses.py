import torch


# BPS RMSE loss
def bps_rmse_loss(pred, target):
    return 10000.0 * torch.sqrt(torch.mean((pred - target) ** 2))
