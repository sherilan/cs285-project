
import os

import torch
import torch.nn as nn
import numpy as np

import project.utils as utils
import project.loggers as loggers
import project.configs as configs


class Agent(nn.Module):
    """
    Base class with boilerplate for setting up experiments etc.
    """

    class Config(configs.Config):
        path = None
        seed = np.random.randint(1<<32)
        gpu = True
        gpu_id = None

    def __init__(self, cfg, device=None, logger=None, savedir=None):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.logger = logger or loggers.Logger()
        self.savedir = savedir
        self.init()

    def init(self):
        """Initializes the agent (create policy etc)"""
        pass

    def train(self):
        """Trains an agent fully"""
        pass

    def save(self, name='agent.pt'):
        # TODO: change to state_dict solution for better portability
        if self.savedir:
            path = os.path.join(self.savedir, name)
            torch.save(self, path)
            self.logger.info('Agent saved to %s', path)

    @classmethod
    def restore(cls, path):
        # TODO: this is pointless without state dict
        return torch.load(path)
        # cfg = cls.Config.from_yaml(os.path.join(savedir, 'config.yml'))
        # agent = cls(cfg, **kwargs)
        # agent.load()
        # return agent

    @classmethod
    def run(cls, cfg, **kwargs):
        # Setup experiment dir and save the config if configured
        savedir = utils.setup_experiment(cfg.path)
        if savedir:
            cfg.save_yaml(savedir / 'config.yml')
        # Setup logger (with tensorboard if savedir is not None)
        logger = loggers.Logger(logdir=savedir)
        logger.info('Using Config: \n%s', cfg)
        # Configure GPU and get the device that will be used
        device = utils.setup_gpu(cfg.gpu, cfg.gpu_id)
        logger.info('Using device: %s', device)
        # Set random seeds for numpy as torch
        utils.seed_everything(cfg.seed)
        logger.info('Using seed: %s', cfg.seed)
        # Create agent, train, save, and return it
        agent = cls(
            cfg, device=device, logger=logger, savedir=savedir, **kwargs
        )
        logger.info('Created agent: \n%s', agent)
        agent.train()
        if savedir:
            agent.save()
        return agent

    @classmethod
    def run_cli(cls):
        cfg = cls.Config.from_cli()
        return cls.run(cfg)
