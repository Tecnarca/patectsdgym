import torch
import torch.nn as nn
import math
import numpy as np

def weights_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def pate(data, teachers, lap_scale):
    num_teachers = len(teachers)
    labels = torch.Tensor(num_teachers, data.shape[0]).type(torch.int64)
    for i in range(num_teachers):
        output = teachers[i](data)
        pred = (output > 0.5).type(torch.Tensor).squeeze()
        labels[i] = pred

    votes = torch.sum(labels, dim=0).unsqueeze(1).type(torch.DoubleTensor)
    noise = torch.from_numpy(np.random.laplace(loc=0, scale=1/lap_scale, size=votes.size()))
    noisy_votes = votes + noise
    noisy_labels = (noisy_votes > num_teachers/2).type(torch.DoubleTensor)

    return noisy_labels, votes

def moments_acc(num_teachers, votes, lap_scale, l_list):
    q = (2 + lap_scale * torch.abs(2*votes - num_teachers)
        ) / (4 * torch.exp(lap_scale * torch.abs(2*votes - num_teachers)))

    alpha = []
    for l in l_list:
        a = 2 * lap_scale**2 * l * (l+1)
        t_one = (1-q) * torch.pow((1-q)/(1-math.exp(2*lap_scale) * q), l)
        t_two = q * torch.exp(2*lap_scale * l)
        t = t_one + t_two
        alpha.append(torch.clamp(t, max=a).sum())

    return torch.DoubleTensor(alpha)