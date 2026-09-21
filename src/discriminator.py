import torch
import torch.nn as nn
from src.config import NUM_FEATURES, D_HIDDEN_DIM

class Discriminator(nn.Module):
    """
    IDSGAN Discriminator network (Section 3.3).
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(NUM_FEATURES, D_HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(D_HIDDEN_DIM, D_HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(D_HIDDEN_DIM, 1)
            # No sigmoid at output as WGAN uses raw scores
        )

    def forward(self, x):
        """
        Forward pass for the discriminator.
        Args:
            x (torch.Tensor): Traffic records [batch_size, 122]
        Returns:
            torch.Tensor: Raw scores
        """
        return self.model(x)
