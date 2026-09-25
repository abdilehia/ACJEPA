import torch

class Transform():
    def __init__(self, mean, std, device: str | torch.device ="cpu"):
        self.mean = mean
        self.std = std
        self.device = device

    def transform(self, data, mean=None, std=None):
        if mean is None and std is None:
            return (data - self.mean) / self.std
        else:
            return (data - mean) / std

    def inv_transform(self, data, mean=None, std=None):
        if mean is None and std is None:
            return data * self.std + self.mean
        else:
            return data * std + mean