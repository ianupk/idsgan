import torch
import torch.optim as optim
import numpy as np
from src.generator import Generator
from src.discriminator import Discriminator
from src.config import LR_GENERATOR, LR_DISCRIMINATOR, CLIP_VALUE, NOISE_DIM, N_CRITIC, DEVICE
from src.feature_utils import get_functional_feature_mask

class IDSGAN:
    """
    IDSGAN training module implementing Algorithm 1.
    """
    def __init__(self, attack_category, ids_model):
        self.attack_category = attack_category
        self.ids_model = ids_model
        
        self.generator = Generator().to(DEVICE)
        self.discriminator = Discriminator().to(DEVICE)
        
        self.opt_G = optim.RMSprop(self.generator.parameters(), lr=LR_GENERATOR)
        self.opt_D = optim.RMSprop(self.discriminator.parameters(), lr=LR_DISCRIMINATOR)
        
        self.functional_mask = get_functional_feature_mask(attack_category)

    def train(self, normal_data, attack_data, epochs, batch_size):
        """
        Full training loop for IDSGAN.
        """
        normal_data = torch.tensor(normal_data, dtype=torch.float32).to(DEVICE)
        attack_data = torch.tensor(attack_data, dtype=torch.float32).to(DEVICE)
        
        loss_history = {'G_loss': [], 'D_loss': []}
        
        for epoch in range(epochs):
            # 1. Critic (Discriminator) steps
            for _ in range(N_CRITIC):
                self.opt_D.zero_grad()
                
                # Sample batch
                idx_normal = torch.randint(0, normal_data.size(0), (batch_size,))
                idx_attack = torch.randint(0, attack_data.size(0), (batch_size,))
                
                batch_normal = normal_data[idx_normal]
                batch_attack = attack_data[idx_attack]
                
                noise = torch.rand(batch_size, NOISE_DIM).to(DEVICE)
                
                # Generate adversarial examples
                adv_attack = self.generator.generate(batch_attack, noise, self.functional_mask)
                
                # Query black-box IDS with combined batch
                combined_batch = torch.cat([batch_normal, adv_attack.detach()], dim=0)
                combined_np = combined_batch.cpu().numpy()
                
                ids_preds = self.ids_model.predict(combined_np)
                
                # Identify which records the IDS predicted as normal vs attack
                # We assume the IDS outputs 0 for normal, 1 for attack, or string labels
                if np.issubdtype(ids_preds.dtype, np.number):
                    pred_normal_mask = (ids_preds == 0)
                    pred_attack_mask = (ids_preds == 1)
                else:
                    # Handle string outputs like 'normal' and 'attack'
                    pred_normal_mask = (ids_preds == 'Normal') | (ids_preds == 'normal')
                    pred_attack_mask = ~pred_normal_mask
                    
                b_normal = combined_batch[pred_normal_mask]
                b_attack = combined_batch[pred_attack_mask]
                
                loss_D = torch.tensor(0.0, device=DEVICE)
                
                # L_D = E[D(s)]_{B_normal} - E[D(s)]_{B_attack}
                # We minimize L_D
                if len(b_normal) > 0 and len(b_attack) > 0:
                    loss_D = torch.mean(self.discriminator(b_normal)) - torch.mean(self.discriminator(b_attack))
                    loss_D.backward()
                    self.opt_D.step()
                elif len(b_normal) > 0:
                    loss_D = torch.mean(self.discriminator(b_normal))
                    loss_D.backward()
                    self.opt_D.step()
                elif len(b_attack) > 0:
                    loss_D = -torch.mean(self.discriminator(b_attack))
                    loss_D.backward()
                    self.opt_D.step()
                
                # Clip discriminator weights
                for p in self.discriminator.parameters():
                    p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)
                    
            # 2. Generator step
            self.opt_G.zero_grad()
            
            idx_attack = torch.randint(0, attack_data.size(0), (batch_size,))
            batch_attack = attack_data[idx_attack]
            noise = torch.rand(batch_size, NOISE_DIM).to(DEVICE)
            
            adv_attack = self.generator.generate(batch_attack, noise, self.functional_mask)
            
            # L_G = E[D(G(M, N))]
            # We minimize L_G
            loss_G = torch.mean(self.discriminator(adv_attack))
            loss_G.backward()
            self.opt_G.step()
            
            loss_history['G_loss'].append(loss_G.item())
            loss_history['D_loss'].append(loss_D.item())
            
            # Log losses every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")
                
        return loss_history

    def generate_adversarial(self, attack_data):
        """
        Generate adversarial examples using the trained Generator.
        """
        self.generator.eval()
        with torch.no_grad():
            attack_data_tensor = torch.tensor(attack_data, dtype=torch.float32).to(DEVICE)
            noise = torch.rand(attack_data_tensor.size(0), NOISE_DIM).to(DEVICE)
            adv = self.generator.generate(attack_data_tensor, noise, self.functional_mask)
        self.generator.train()
        return adv.cpu().numpy()
        
    def save(self, path):
        """Save model checkpoints."""
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict()
        }, path)
        
    def load(self, path):
        """Load model checkpoints."""
        checkpoint = torch.load(path, map_location=DEVICE)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
