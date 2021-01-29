

import torch
import torch.nn as nn
import numpy as np

import diayn.distributions as distributions
import diayn.networks as networks



class Policy(nn.Module):
    """
    Very agnostic base class for policies
    """

    def __init__(self):
        super().__init__()
        # Hack to figure out which device the policy is currently on
        self.device_detector = nn.Parameter(torch.zeros(0), requires_grad=False)

    @property
    def device(self):
        """
        Returns the torch device this policy is currently assigned to
        """
        return self.device_detector.device

    def reset(self, *args, **kwargs):
        """
        Hook for resetting potential policy state.
        Useful for, e.g. recurrent policies.
        """
        pass

    def get_action(self, *observations, greedy=False):
        """
        Sample an action from the policy

        Args:
            *observations (np.ndarray) input to condition the policy on
            greedy (bool) whether to sample a greedy (deterministic) action

        Returns:
            np.ndarry with the action sampled from the policy
        """
        with torch.no_grad():
            observations = [
                torch.as_tensor(o, dtype=torch.float32, device=self.device)
                for o in observations
            ]
            pi = self(*observations)
            action = pi.sample(greedy=greedy)
            return action.cpu().numpy()


class GaussianMLPPolicy(Policy):

    def __init__(
        self,
        ob_dim,
        ac_dim,
        static_std=False,
        init_std=1.0,
        min_std=np.exp(-20.0),
        max_std=np.exp(2.0),
        **mlp_kwargs
    ):
        super().__init__()
        # Unwrap ob|ac_dim if they were given as single-element tuples
        assert np.isscalar(ob_dim) or len(ob_dim) == 1
        assert np.isscalar(ac_dim) or len(ac_dim) == 1
        self.ob_dim = ob_dim if np.isscalar(ob_dim) else ob_dim[0]
        self.ac_dim = ac_dim if np.isscalar(ac_dim) else ac_dim[0]
        self.static_std = static_std
        # Store limits for std
        if not min_std is None:
            min_logstd = torch.tensor(min_std).log()
            self.min_logstd = nn.Parameter(min_logstd, requires_grad=False)
        else:
            self.min_logstd = None
        if not max_std is None:
            max_logstd = torch.tensor(max_std).log()
            self.max_logstd = nn.Parameter(max_logstd, requires_grad=False)
        else:
            self.max_logstd = False
        # Case: static std -> use a simple mlp to predict just the mean
        if static_std:
            self.base = networks.MLP(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                **mlp_kwargs
            )
            self.logstd = nn.Parameter(torch.ones(self.ac_dim))
        # Case dynamic std -> use multi-headed mlp to predic mean and log(std)
        else:
            self.base = networks.MultiHeadMLP(
                input_size=self.ob_dim,
                output_sizes=[self.ac_dim, self.ac_dim],
                output_names=['mean', 'logstd'],
                **mlp_kwargs
            )

    def forward(self, observation):
        mean, std = self.get_mean_and_std(observation)
        return distributions.Gaussian(mean, std)

    def get_mean_and_std(self, observation):
        # Compute mean/std depending on whether std is static
        if self.static_std:
            mean, logstd = self.base(observation), self.logstd
        else:
            mean, logstd = self.base(observation)
        # Possibly clamp std based on configured min/max values
        if self.min_logstd is not None or self.max_logstd is not None:
            logstd = logstd.clamp(
                min=None if self.min_logstd is None else self.min_logstd,
                max=None if self.max_logstd is None else self.max_logstd,
            )
        # Exponentiate to get actual std
        std = torch.exp(logstd)

        return mean, std


class TanhGaussianMLPPolicy(GaussianMLPPolicy):
    """
    A Gaussian Policy, except that actions are sqeezed through a tanh.
    """

    def forward(self, observation):
        mean, std = self.get_mean_and_std(observation)
        return distributions.TanhGaussian(mean, std)
