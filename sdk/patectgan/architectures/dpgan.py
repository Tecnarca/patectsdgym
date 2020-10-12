import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from opacus import PrivacyEngine, utils, autograd_grad_sample

from patectgan.privacy_utils import weights_init, pate, moments_acc

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim, binary=True):
        super(Generator, self).__init__()
        def block(in_, out, Activation):
            return nn.Sequential(
                nn.Linear(in_, out, bias=False),
                nn.LayerNorm(out),
                Activation(),
            )
        
        self.layer_0 = block(latent_dim, latent_dim, nn.Tanh if binary else lambda: nn.LeakyReLU(0.2))
        self.layer_1 = block(latent_dim, latent_dim, nn.Tanh if binary else lambda: nn.LeakyReLU(0.2))
        self.layer_2 = block(latent_dim, output_dim, nn.Tanh if binary else lambda: nn.LeakyReLU(0.2))
        
    def forward(self, noise):
        noise = self.layer_0(noise) + noise
        noise = self.layer_1(noise) + noise
        noise = self.layer_2(noise)
        return noise


class Discriminator(nn.Module):
    def __init__(self, input_dim, wasserstein=False):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 2*input_dim // 3),
            nn.LeakyReLU(0.2),
            nn.Linear(2*input_dim // 3, input_dim // 3),
            nn.LeakyReLU(0.2),
            nn.Linear(input_dim // 3, 1)
        )

        if not wasserstein:
            self.model.add_module("activation", nn.Sigmoid())

    def forward(self, x):
        return self.model(x)

class DPGAN:
    def __init__(self, 
                 binary=False,
                 latent_dim=64, 
                 batch_size=64,
                 epochs=1000,
                 delta=1e-5,
                 epsilon=1.0):
        self.binary = binary
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.delta = delta
        self.epsilon = epsilon
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.pd_cols = None
        self.pd_index = None

    def train(self, data, categorical_columns=None, ordinal_columns=None, update_epsilon=None, verbose=False):
        if update_epsilon:
            self.epsilon = update_epsilon

        if isinstance(data, pd.DataFrame):
            for col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='ignore')
            self.pd_cols = data.columns
            self.pd_index = data.pd_index
            data = data.to_numpy()
        elif not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy array or pandas dataframe")

        dataset = TensorDataset(torch.from_numpy(data.astype('float32')).to(self.device))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        if not hasattr(self, "generator"):
            self.generator = Generator(self.latent_dim, data.shape[1], binary=self.binary).to(self.device)
        if not hasattr(self, "discriminator"):
            self.discriminator = Discriminator(data.shape[1]).to(self.device)
        
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=4e-4, betas=(0.5, 0.9))
        if hasattr(self, "state_dict"):
            self.optimizer_d.load_state_dict(self.state_dict)
        
        privacy_engine = PrivacyEngine(
            self.discriminator,
            batch_size=self.batch_size,
            sample_size=len(data),
            alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            noise_multiplier=3.5,
            max_grad_norm=1.0,
            clip_per_layer=True
            )
        
        privacy_engine.attach(self.optimizer_d)
        
        if not hasattr(self, "optimizer_g"):
            self.optimizer_g = optim.Adam(self.generator.parameters(), lr=1e-4)

        eps = 0

        criterion = nn.BCELoss()
        
        if hasattr(self,"trained_eps"):
            if self.epsilon < eps+self.trained_eps:
                return
        else:
            self.trained_eps = 0

        for epoch in range(self.epochs):
            for i, data in enumerate(dataloader):
                self.discriminator.zero_grad()
                
                real_data = data[0].to(self.device)

                # train with fake data
                noise = torch.randn(self.batch_size, self.latent_dim, 1, 1, device=self.device)
                noise = noise.view(-1, self.latent_dim)
                fake_data = self.generator(noise)
                label_fake = torch.full((self.batch_size,1), 0, dtype=torch.float, device=self.device)
                output = self.discriminator(fake_data.detach())
                loss_d_fake = criterion(output, label_fake)
                loss_d_fake.backward()
                self.optimizer_d.step()
                
                # train with real data
                label_true = torch.full((self.batch_size,1), 1, dtype=torch.float, device=self.device)
                output = self.discriminator(real_data.float())
                loss_d_real = criterion(output, label_true)
                loss_d_real.backward()
                self.optimizer_d.step()
                
                loss_d = loss_d_real + loss_d_fake

                max_grad_norm = []
                for p in self.discriminator.parameters():
                    param_norm = p.grad.data.norm(2).item()
                    max_grad_norm.append(param_norm)
                
                privacy_engine.max_grad_norm = max_grad_norm
            
                # train generator
                self.generator.zero_grad()
                label_g = torch.full((self.batch_size,1), 1, dtype=torch.float, device=self.device)
                output_g = self.discriminator(fake_data)
                loss_g = criterion(output_g, label_g)
                loss_g.backward()
                self.optimizer_g.step()
            
                # manually clear gradients
                for p in self.discriminator.parameters():
                    if hasattr(p, "grad_sample"):
                        del p.grad_sample
                # autograd_grad_sample.clear_backprops(discriminator)

                if self.delta is None:
                    self.delta = 1 / data.shape[0]
                
                eps, best_alpha = self.optimizer_d.privacy_engine.get_privacy_spent(self.delta)
                self.alpha = best_alpha

            if(verbose):
                print ('eps: {:f} \t alpha: {:f} \t G: {:f} \t D: {:f}'.format(eps+self.trained_eps, best_alpha, loss_g.detach().cpu(), loss_d.detach().cpu()))

            if self.epsilon < eps+self.trained_eps:
                break

        self.trained_eps = eps+self.trained_eps
        privacy_engine.detach()
        self.state_dict = self.optimizer_d.state_dict()

    def generate(self, n):
        steps = n // self.batch_size + 1
        data = []
        for i in range(steps):
            noise = torch.randn(self.batch_size, self.latent_dim, 1, 1, device=self.device)
            noise = noise.view(-1, self.latent_dim)
            
            fake_data = self.generator(noise)
            data.append(fake_data.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return data

    def save(self, path):
        assert hasattr(self, "generator")
        assert hasattr(self, "discriminator")

        # always save a cpu model.
        device_bak = self.device
        self.device = torch.device("cpu")
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        torch.save(self, path)

        self.device = device_bak
        self.generator.to(self.device)
        self.discriminator.to(self.device)

    @classmethod
    def load(cls, path):
        model = torch.load(path)
        model.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.generator.to(model.device)
        model.discriminator.to(model.device)
        return model