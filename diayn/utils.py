
import os
import warnings

import garage.envs
import gym
import numpy as np
import torch


def setup_experiment(savedir, on_conflict='raise'):
    assert on_conflict in {'ignore', 'raise'}
    if savedir is None:
        return
    if os.path.exists(savedir) and on_conflict == 'raise':
        raise ValueError(f'Savedir "{savedir}" already exists')
    else:
        os.mkdir(savedir)
        # TODO: maybe save metadata, like git info, if available

def setup_gpu(gpu, gpu_id=None):
    if gpu:
        if not torch.cuda.is_available():
            warnings.warn('GPU not available, falling back on CPU')
            return torch.device('cpu')
        else:
            gpu_id = 0 if gpu_id is None else gpu_id
            return torch.device(f'cuda:{gpu_id}')
    else:
        return torch.device('cpu')

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def make_env(name):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="^.*Box bound precision lowered.*$"
        )
        return garage.envs.GarageEnv(
            garage.envs.normalize(gym.make('HalfCheetah-v2'))
        )


def get_optimizer(name, *args, **kwargs):
    return {
        'adam': torch.optim.Adam
    }[name](*args, **kwargs)

def iter_optim_params(optimizer):
    for param_group in optimizer.param_groups:
        yield from param_group['params']

def optimize(loss, optimizer, norm_clip=None, norm_order=2):
    optimizer.zero_grad()
    loss.backward()
    if norm_clip:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            iter_optim_params(optimizer), norm_clip, norm_order
        )
    else:
        grad_norm = get_grad_norm(
            iter_optim_params(optimizer), norm_order
        )
    optimizer.step()
    return grad_norm

def get_grad_norm(parameters, order=2):
    return sum(
        p.grad.data.norm(order) for p in parameters
        if not p.grad is None
    )
