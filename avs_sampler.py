"""Code containing the base AVS sampler and a few variants that we experimented
with. The Variational Sampler is the AVS sampler in the final paper."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, MultivariateNormal, MixtureSameFamily, Categorical
import numpy as np

class Encoder(nn.Module):
    """p_θ(a|x): x -> a (encoder)"""
    
    def __init__(self, target_dim, aux_dim, hidden_dim, num_layers=3):
        super().__init__()
        
        layers = []
        in_dim = target_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        
        self.net = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(hidden_dim, aux_dim)
        self.logvar_layer = nn.Linear(hidden_dim, aux_dim)
    
    def forward(self, x):
        h = self.net(x)
        mean = self.mean_layer(h)
        logvar = self.logvar_layer(h)
        std = torch.exp(0.5 * logvar)
        return Normal(mean, std)


class Decoder(nn.Module):
    """q_φ(x|a): a -> x (decoder)"""
    
    def __init__(self, target_dim, aux_dim, hidden_dim, num_layers=3):
        super().__init__()
        
        layers = []
        in_dim = aux_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        
        self.net = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(hidden_dim, target_dim)
        self.logvar_layer = nn.Linear(hidden_dim, target_dim)
    
    def forward(self, a):
        h = self.net(a)
        mean = self.mean_layer(h)
        logvar = self.logvar_layer(h)
        std = torch.exp(0.5 * logvar)
        return Normal(mean, std)


class AuxiliaryVariationalSampler:
    def __init__(self, target_logprob, target_dim, aux_dim=1, hidden_dim=64, 
                 num_layers=3, perturb=0.5, device='cpu'):
        
        self.target_logprob = target_logprob
        self.target_dim = target_dim
        self.aux_dim = aux_dim
        self.perturb = perturb
        self.device = device
        
        self.encoder = Encoder(target_dim, aux_dim, hidden_dim, num_layers).to(device)
        self.decoder = Decoder(target_dim, aux_dim, hidden_dim, num_layers).to(device)
        
        # Prior на a: N(0, I)
        self.prior_a = Normal(
            torch.zeros(aux_dim, device=device),
            torch.ones(aux_dim, device=device)
        )
    
    def train(self, max_iters=5000, batch_size=100, lr=1e-3, print_freq=500):
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr
        )
        
        losses = []
        
        for t in range(max_iters):
            optimizer.zero_grad()
            
            # a ~ q(a) = N(0, I)
            a = self.prior_a.sample((batch_size,))
            
            # x ~ q(x|a)
            q_x_given_a = self.decoder(a)
            x = q_x_given_a.rsample()  # reparameterization trick
            
            # Calculate p(a|x)
            p_a_given_x = self.encoder(x)
            
            # Loss = KL(q(a,x) || p(x)p(a|x))
            # = E_q [log q(a) + log q(x|a) - log p(a|x) - log p(x)]
            loss = (
                self.prior_a.log_prob(a).sum(dim=-1) +
                q_x_given_a.log_prob(x).sum(dim=-1) -
                p_a_given_x.log_prob(a).sum(dim=-1) -
                self.target_logprob(x)
            ).mean()
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if t % print_freq == 0:
                print(f"Iter {t}: loss = {loss.item():.4f}")
        
        return losses
    
    def sample(self, num_samples, num_chains=10, burn_in=500, initial_x=None):
        self.encoder.eval()
        self.decoder.eval()
        
        samples = []
        
        with torch.no_grad():
            if initial_x is not None:
                if initial_x.dim() == 1:
                    x = initial_x.unsqueeze(0).repeat(num_chains, 1).to(self.device)
                else:
                    x = initial_x.to(self.device)
            else:
            # a ~ N(0, I), x ~ q(x|a)
                a_init = self.prior_a.sample((num_chains,))
                x = self.decoder(a_init).sample()
            
            accepts = 0
            total = 0
            
            for i in range(burn_in + num_samples):
                # 1. Encode: a ~ p(a|x)
                p_a_given_x = self.encoder(x)
                a = p_a_given_x.sample()
                
                # 2. Random walk in space a
                a_prime = a + torch.randn_like(a) * self.perturb
                
                # 3. Decode: x' ~ q(x|a')
                q_x_given_a_prime = self.decoder(a_prime)
                x_proposed = q_x_given_a_prime.sample()
                
                # 4. Calculate acceptance ratio
                q_x_given_a = self.decoder(a)
                p_a_prime_given_x_proposed = self.encoder(x_proposed)
                
                log_alpha = (
                    self.target_logprob(x_proposed) +                   # log p(x')
                    p_a_prime_given_x_proposed.log_prob(a_prime).sum(dim=-1) +  # log p(a'|x')
                    q_x_given_a.log_prob(x).sum(dim=-1) -              # log q(x|a)
                    self.target_logprob(x) -                            # log p(x)
                    p_a_given_x.log_prob(a).sum(dim=-1) -              # log p(a|x)
                    q_x_given_a_prime.log_prob(x_proposed).sum(dim=-1)  # log q(x'|a')
                )
                
                # 5. Accept/reject
                u = torch.rand(num_chains, device=self.device)
                accept = torch.log(u) < log_alpha
                
                x = torch.where(accept.unsqueeze(-1), x_proposed, x)
                
                if i >= burn_in:
                    samples.append(x.clone())
                    accepts += accept.sum().item()
                    total += num_chains
            
            print(f"Acceptance rate: {accepts / total:.2%}")
        
        # [num_samples, num_chains, target_dim]
        return torch.stack(samples).cpu().numpy()