
import os
import pathlib
import warnings

import garage.envs
import gym
import numpy as np
import torch


def setup_experiment(path, on_conflict='enumerate'):
    assert on_conflict in {'ignore', 'raise', 'enumerate'}
    if path is None:
        return None
    else:
        path = pathlib.Path(path).absolute()
    if on_conflict == 'enumerate':
        for i in range(1000):
            savedir = path / f'exp_{str(i).zfill(3)}'
            if not savedir.exists():
                os.makedirs(savedir)
                break
        else:
            raise ValueError('Exceeded 1k enumerated experiments')
    elif on_conflict == 'raise' and savedir.exists():
        raise ValueError(f'Experiment path "{path}" already exists')
    else:
        os.makedirs(path)
        savedir = path
    # TODO: maybe save metadata, like git info, if available

    return savedir

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
        env = garage.envs.GarageEnv(
            garage.envs.normalize(gym.make(name))
        )
    # Generate random seed for env.
    # Assuming np has been seeded, this makes everything reproducible
    env.seed(np.random.randint(1 << 32))
    env.action_space.seed(np.random.randint(1 << 32))
    env.observation_space.seed(np.random.randint(1 << 32))
    return env



def to_one_hot(indices, n):
    indices = torch.as_tensor(indices, dtype=torch.int64)
    shape = tuple(indices.shape) + (n,)
    one_hot = torch.zeros(*shape, device=indices.device)
    one_hot.scatter_(-1, indices.unsqueeze(-1), 1)
    return one_hot


def get_optimizer(name, params, *args, **kwargs):
    return {
        'adam': torch.optim.Adam
    }[name](params, *args, **kwargs)

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

def polyak(net, target, tau):
    """
    Perform one polyak update on parameters

    Args:
        net (nn.Module) torch module to average in params from
        target (nn.Module) torch module
        tau (flaot) update strength. 1.0 will completely copy to target
    """
    params = net.parameters()
    params_target = target.parameters()
    for param, param_target in zip(params, params_target):
        exp_avg = (1.0 - tau) * param_target.data + tau * param.data
        param_target.data.copy_(exp_avg)
