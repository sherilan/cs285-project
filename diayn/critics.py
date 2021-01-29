

import torch
import torch.nn as nn
import numpy as np

import diayn.networks as networks


class Critic(nn.Module):
    pass

class QAMLPCritic(Critic):
    """
    A Q-function critic that takes an observation and action as
    input, feeds it through an MLP, and spits out a single scalar.
    """

    def __init__(self, ob_dim, ac_dim, **mlp_kwargs):
        super().__init__()
        # Unwrap ob|ac_dim if they were given as single-element tuples
        assert np.isscalar(ob_dim) or len(ob_dim) == 1
        assert np.isscalar(ac_dim) or len(ac_dim) == 1
        self.ob_dim = ob_dim if np.isscalar(ob_dim) else ob_dim[0]
        self.ac_dim = ac_dim if np.isscalar(ac_dim) else ac_dim[0]
        # Build MLP base
        self.base = networks.MLP(
            input_size=self.ob_dim + self.ac_dim,
            output_size=1,
            **mlp_kwargs
        )

    def forward(self, observation, action):
        x = torch.cat([observation, action], dim=-1)
        q = self.base(x).squeeze(dim=-1)
        return q
