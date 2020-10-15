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

class PATEGAN:
    def __init__(self,
                 binary=False,
                 latent_dim=64,
                 batch_size=64,
                 teacher_iters=5,
                 student_iters=5,
                 epsilon=1.0,
                 delta=1e-5,
                 verbose=False,
                 sample_per_teacher=1000,
                 moments_order=100):
        self.binary = binary
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.teacher_iters = teacher_iters
        self.student_iters = student_iters
        self.epsilon = epsilon
        self.delta = delta
        self.sample_per_teacher = sample_per_teacher

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.moments_order = moments_order
        self.verbose = verbose
        self.pd_cols = None
        self.pd_index = None
    
    def train(self, data, categorical_columns=None, ordinal_columns=None, update_epsilon=None, verbose=False, mlflow=False):
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

        data_dim = data.shape[1]

        if not hasattr(self, "teacher_disc"):
            sample_per_teacher = self.sample_per_teacher if self.sample_per_teacher < len(data) else 1000
            self.num_teachers = int(len(data) / sample_per_teacher) + 1
            self.data_partitions = np.array_split(data, self.num_teachers)
            self.teacher_disc = [Discriminator(data_dim).double().to(self.device) for i in range(self.num_teachers)]
            for i in range(self.num_teachers):
                self.teacher_disc[i].apply(weights_init)
        
        tensor_partitions = [TensorDataset(torch.from_numpy(data.astype('double')).to(self.device)) for data in self.data_partitions]
        
        loader = []
        for teacher_id in range(self.num_teachers):
            loader.append(DataLoader(tensor_partitions[teacher_id], batch_size=self.batch_size, shuffle=True))

        if not hasattr(self, "generator"):
            self.generator = Generator(self.latent_dim, data_dim, binary=self.binary).double().to(self.device)
            self.generator.apply(weights_init)
        
        if not hasattr(self,"student_disc"):
            self.student_disc = Discriminator(data_dim).double().to(self.device)
            self.student_disc.apply(weights_init)
        
        if not hasattr(self, "optimizer_g"):
            self.optimizer_g = optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.9))
        if not hasattr(self, "optimizer_s"):    
            self.optimizer_s = optim.Adam(self.student_disc.parameters(), lr=2e-4, betas=(0.5, 0.9))
        if not hasattr(self, "optimizer_t"):    
            self.optimizer_t = [optim.Adam(self.teacher_disc[i].parameters(), lr=2e-4, betas=(0.5, 0.9)) for i in range(self.num_teachers)]
        
        if not hasattr(self, "train_eps"):
            self.alphas = torch.tensor([0.0 for i in range(self.moments_order)], device=self.device)
            self.l_list = 1 + torch.tensor(range(self.moments_order), device=self.device)
            self.train_eps = 0
            self.noise_multiplier = 1e-3

        criterion = nn.BCELoss()

        
        epoch = 0
        while self.train_eps < self.epsilon:
            
            # train teacher discriminators
            for t_2 in range(self.teacher_iters):
                for i in range(self.num_teachers):
                    real_data, category = None, None
                    for j, data in enumerate(loader[i], 0):
                        real_data = data[0].to(self.device)
                        break
                    
                    self.optimizer_t[i].zero_grad()

                    # train with real data
                    label_real = torch.full((real_data.shape[0],1), 1, dtype=torch.float, device=self.device)
                    output = self.teacher_disc[i](real_data)
                    loss_t_real = criterion(output, label_real.double())
                    loss_t_real.backward()

                    # train with fake data
                    noise = torch.rand(self.batch_size, self.latent_dim, device=self.device)
                    label_fake = torch.full((self.batch_size,1), 0, dtype=torch.float, device=self.device)
                    fake_data = self.generator(noise.double())
                    output = self.teacher_disc[i](fake_data)
                    loss_t_fake = criterion(output, label_fake.double())
                    loss_t_fake.backward()
                    self.optimizer_t[i].step()

            # train student discriminator
            for t_3 in range(self.student_iters):
                noise = torch.rand(self.batch_size, self.latent_dim, device=self.device)
                fake_data = self.generator(noise.double()).to(self.device)
                predictions, votes = pate(fake_data, self.teacher_disc, self.noise_multiplier, device=self.device)
                output = self.student_disc(fake_data.detach())

                # update moments accountant
                self.alphas = self.alphas + moments_acc(self.num_teachers, votes, self.noise_multiplier, self.l_list, self.device)

                loss_s = criterion(output, predictions.to(self.device))
                self.optimizer_s.zero_grad()
                loss_s.backward()
                self.optimizer_s.step()

            # train generator
            label_g = torch.full((self.batch_size,1), 1, dtype=torch.float, device=self.device)
            noise = torch.rand(self.batch_size, self.latent_dim, device=self.device)
            gen_data = self.generator(noise.double())
            output_g = self.student_disc(gen_data)
            loss_g = criterion(output_g, label_g.double())
            self.optimizer_g.zero_grad()
            loss_g.backward()
            self.optimizer_g.step()

            self.train_eps = min((self.alphas - math.log(self.delta)) / self.l_list)

            if(verbose):
                print ('eps: {:f} \t G: {:f} \t D: {:f}'.format(self.train_eps, loss_g.detach().cpu(), loss_s.detach().cpu()))

            if(mlflow):
                import mlflow
                mlflow.log_metric("loss_g", float(loss_g.detach().cpu()), step=epoch)
                mlflow.log_metric("loss_d", float(loss_s.detach().cpu()), step=epoch)
                mlflow.log_metric("epsilon", float(self.train_eps), step=epoch)

            epoch += 1
 
    def generate(self, n):
        steps = n // self.batch_size + 1
        data = []
        for i in range(steps):
            noise = torch.randn(self.batch_size, self.latent_dim, device=self.device)
            noise = noise.view(-1, self.latent_dim)
            
            fake_data = self.generator(noise.double())
            data.append(fake_data.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return data

    def save(self, path):
        assert hasattr(self, "generator")
        assert hasattr(self, "teacher_disc")
        assert hasattr(self, "student_disc")

        # always save a cpu model.
        device_bak = self.device
        self.device = torch.device("cpu")
        self.generator.to(self.device)
        for i in range(self.num_teachers):
            self.teacher_disc[i].to(self.device)
        self.student_disc.to(self.device)

        torch.save(self, path)

        self.device = device_bak
        self.generator.to(self.device)
        for i in range(self.num_teachers):
            self.teacher_disc[i].to(self.device)
        self.student_disc.to(self.device)

    @classmethod
    def load(cls, path):
        model = torch.load(path)
        model.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.generator.to(model.device)
        model.student_disc.to(model.device)
        for i in range(model.num_teachers):
            model.teacher_disc[i].to(model.device)
        return model
