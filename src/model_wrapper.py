import torch
import torch.nn as nn

class NormalizedModelWrapper(nn.Module):
    """
    A wrapper that applies normalization as the first layer.
    This decouples the dataloader from the model's expected mean/std.
    """
    def __init__(self, model, mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)):
        super().__init__()
        self.model = model
        
        # Register as buffers so they move with the model but aren't trainable
        # Shape: (1, 3, 1, 1, 1) to match (B, C, T, H, W)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1, 1))

    def forward(self, x):
        # x is assumed to be in [0, 1]
        x = (x - self.mean) / self.std
        return self.model(x)

    # Proxy attributes to the underlying model if needed
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
