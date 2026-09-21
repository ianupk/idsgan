import torch
import torch.nn as nn
from src.config import NUM_FEATURES, NOISE_DIM, G_HIDDEN_DIM
from src.feature_utils import get_binary_feature_indices_encoded, get_categorical_feature_indices_encoded

class Generator(nn.Module):
    """
    IDSGAN Generator network (Section 3.3).
    """
    def __init__(self):
        super(Generator, self).__init__()
        input_dim = NUM_FEATURES + NOISE_DIM
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, G_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(G_HIDDEN_DIM, G_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(G_HIDDEN_DIM, G_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(G_HIDDEN_DIM, G_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(G_HIDDEN_DIM, NUM_FEATURES)
            # No activation on the output layer
        )
        
        self.binary_indices = get_binary_feature_indices_encoded()
        self.categorical_indices = get_categorical_feature_indices_encoded()

    def forward(self, original_records, noise):
        """
        Forward pass to get raw output.
        Args:
            original_records (torch.Tensor): Original malicious records [batch_size, 122]
            noise (torch.Tensor): Noise vector [batch_size, 9]
        Returns:
            torch.Tensor: Raw generated features [batch_size, 122]
        """
        x = torch.cat([original_records, noise], dim=1)
        return self.model(x)

    def generate(self, original_records, noise, functional_mask):
        """
        Full generation pipeline with restricted modification mechanism.
        Args:
            original_records (torch.Tensor): Original malicious records [batch_size, 122]
            noise (torch.Tensor): Noise vector [batch_size, 9]
            functional_mask (np.ndarray or torch.Tensor): Boolean mask of functional features
        Returns:
            torch.Tensor: Adversarial records
        """
        # 1. Forward pass to get raw output
        raw_output = self.forward(original_records, noise)
        
        # 2. Clamp to [0, 1]
        clamped_output = torch.clamp(raw_output, 0.0, 1.0)
        
        # 3. Apply restricted modification mechanism
        device = original_records.device
        
        if not isinstance(functional_mask, torch.Tensor):
            func_mask = torch.tensor(functional_mask, dtype=torch.bool, device=device)
        else:
            func_mask = functional_mask.to(device)
            
        # Ensure gradients flow correctly using torch.where instead of in-place modifications
        
        # Handle functional features (keep unchanged)
        func_mask_tensor = func_mask.unsqueeze(0).expand_as(original_records)
        adversarial_records = torch.where(func_mask_tensor, original_records, clamped_output)
        
        # Handle categorical features (keep unchanged)
        cat_mask = torch.zeros_like(func_mask)
        cat_mask[self.categorical_indices] = True
        cat_mask_tensor = cat_mask.unsqueeze(0).expand_as(original_records)
        adversarial_records = torch.where(cat_mask_tensor, original_records, adversarial_records)
        
        # Handle binary features (round those that were modified)
        bin_mask = torch.zeros_like(func_mask)
        bin_mask[self.binary_indices] = True
        bin_mask_tensor = bin_mask.unsqueeze(0).expand_as(original_records)
        
        # Rounding with straight-through estimator to preserve gradients
        rounded = (adversarial_records >= 0.5).float()
        rounded_with_grad = adversarial_records + (rounded - adversarial_records).detach()
        
        adversarial_records = torch.where(bin_mask_tensor, rounded_with_grad, adversarial_records)
        
        return adversarial_records
