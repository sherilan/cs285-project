
import contextlib

import torch
import torch.nn as nn
import numpy as np

import project.distributions as distributions
import project.networks as networks



class Policy(nn.Module):
    """
    Very agnostic base class for policies
    """

    def __init__(self, greedy=False):
        super().__init__()
        self.greedy = greedy
        self.debug = False
        # Hack to figure out which device the policy is currently on
        self.device_detector = nn.Parameter(torch.zeros(0), requires_grad=False)

    @contextlib.contextmanager
    def configure(self, **temp_vals):
        """
        Temporarily set values for policy (e.g. greedy)
        """
        try:
            # Store backup of old vals
            old_vals = {k: getattr(self, k) for k in temp_vals}
            # Set temporary values
            for k, v in temp_vals.items():
                setattr(self, k, v)
            # Let context run
            yield old_vals
        finally:
            # Revert to old values
            for k, v in old_vals.items():
                setattr(self, k, v)

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

    def get_action(self, *observations, greedy=None):
        """
        Sample an action from the policy

        Args:
            *observations (np.ndarray) input to condition the policy on
            greedy (bool) whether to sample a greedy (deterministic) action

        Returns:
            an np.ndarray with the action sampled from the policy
            a dict with additional info
        """
        with torch.no_grad():
            observations = [
                torch.as_tensor(o, dtype=torch.float32, device=self.device)
                for o in observations
            ]
            pi = self(*observations)
            action = pi.sample(greedy=self.greedy if greedy is None else greedy)
            return action.cpu().numpy(), {}


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

    def forward(self, *observations):
        mean, std = self.get_mean_and_std(*observations)
        return distributions.Gaussian(mean, std)

    def get_mean_and_std(self, *observations):
        # Compute mean/std depending on whether std is static
        if self.static_std:
            mean, logstd = self.base(*observations), self.logstd
        else:
            mean, logstd = self.base(*observations)
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

    def forward(self, *observations):
        mean, std = self.get_mean_and_std(*observations)
        return distributions.TanhGaussian(mean, std)



class GMMMLPPolicy(Policy):
    """
    An MLP policy that generates a GMM distribution
    """

    def __init__(
        self,
        ob_dim,
        ac_dim,
        num_components=4,
        static_std=False,
        init_std=1.0,
        min_std=np.exp(-20.0),
        max_std=np.exp(2.0),
        min_component_prob=np.exp(-10),
        qf=None,
        **mlp_kwargs
    ):
        super().__init__()
        # Unwrap ob|ac_dim if they were given as single-element tuples
        assert np.isscalar(ob_dim) or len(ob_dim) == 1
        assert np.isscalar(ac_dim) or len(ac_dim) == 1
        self.ob_dim = ob_dim if np.isscalar(ob_dim) else ob_dim[0]
        self.ac_dim = ac_dim if np.isscalar(ac_dim) else ac_dim[0]
        self.num_components = num_components
        self.static_std = static_std
        self.qf = qf
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
            self.max_logstd = None
        # Store lower limit for mixture probability
        if not min_component_prob is None:
            min_component_prob = torch.tensor(min_component_prob).log()
            self.min_component_prob = nn.Parameter(min_component_prob, requires_grad=False)
        else:
            self.min_component_prob = None
        # Case: static std -> use a simple mlp to predict just the logits and means
        if static_std:
            self.base = networks.MultiHeadMLP(
                input_size=self.ob_dim,
                output_sizes=[
                    self.num_components,  # Logits for mixture
                    self.num_components * self.ac_dim,  # Mean for each component
                ],
                output_names=['logits', 'means'],
                # output_w_init=lambda x: x.data.normal(0, 0.1),
                **mlp_kwargs
            )
            self.logstd = nn.Parameter(torch.ones(self.ac_dim))
        # Case dynamic std -> use multi-headed mlp to predict logits, means, and log(stds)
        else:
            self.base = networks.MultiHeadMLP(
                input_size=self.ob_dim,
                output_sizes=[
                    self.num_components,  # Logits for mixture
                    self.num_components * self.ac_dim,  # Mean for each component
                    self.num_components * self.ac_dim,  # Log(std) for each component
                ],
                output_names=['logits', 'means', 'logstds'],
                # output_w_init=lambda x: x.data.normal(0, 0.1),
                **mlp_kwargs
            )

    def forward(self, *observations):
        logits, means, logstds = self.get_logits_means_and_logstds(*observations)
        q_values = self.get_q_values(observations, means)
        return distributions.GMM(logits, means, logstds, q_values=q_values)

    def get_logits_means_and_logstds(self, *observations):
        # Compute mean/std depending on whether std is static
        if self.static_std:
            logits, means = self.base(*observations)
            means = self.reshape_components(means)
            logstds = self.logstd
        else:
            logits, means, logstds = self.base(*observations)
            means = self.reshape_components(means)
            logstds = self.reshape_components(logstds)
        # Possibly clamp std based on configured min/max values
        if self.min_logstd is not None or self.max_logstd is not None:
            logstds = logstds.clamp(
                min=None if self.min_logstd is None else self.min_logstd,
                max=None if self.max_logstd is None else self.max_logstd,
            )
        # Possibly clamp logits
        if self.min_component_prob is not None:
            logits = logits.clamp(self.min_component_prob)
        return logits, means, logstds

    def reshape_components(self, components_flat):
        """
        Reshapes components to make them ready for the GMM

        The MLP base will return means (and logstds if applicable)
        on the form <batch..., num_components * ac_dim>. This method
        reshapes the component to <batch..., num_components, ac_dim>
        so that they have the right format for GMM distributions
        """
        b_dim = components_flat.shape[:-1]  # Get batch dim agnostically
        out_shape = b_dim + (self.num_components, self.ac_dim)
        return components_flat.reshape(out_shape)

    def get_q_values(self, observations, means):
        """
        Generates q-values for the means of each component.
        This is used by the GMM to generate greedy actions that maximize Q.
        """
        # Ignore if no q-function is provided
        if self.qf is None:
            return None
        # Otherwise, generate a q-value for every component mean
        else:
            nc = self.num_components
            observations_repeated = tuple(
                obs.repeat((1,) * (len(means.size()) -1) + (nc,))
                   .reshape(obs.shape[:-1] + (nc,) + obs.shape[-1:])
                for obs in observations
            )
            return self.qf(*(observations_repeated + (means,)))



class TanhGMMMLPPolicy(GMMMLPPolicy):
    """A GMMMLPolicy, except that actions are squeezed through a tanh"""

    def forward(self, *observations):
        logits, means, logstds = self.get_logits_means_and_logstds(*observations)
        q_values = self.get_q_values(observations, means)
        return distributions.TanhGMM(logits, means, logstds, q_values=q_values)




class SkillConditionedPolicy(Policy):
    """
    Wrapper around any other policy that allows skill conditioning

    TODO: deprecate. Use mixin instead.
    """

    def __init__(self,
        base,
        num_skills,
        one_hot=True,
        skill_dist=None,
        skill=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base = base
        self.num_skills = num_skills
        self.one_hot = one_hot
        self.skill_dist = skill_dist
        self.skill = skill

    def reset(self):
        super().reset()
        self.base.reset()
        # If no prob dist, sample uniform
        if self.skill_dist is None:
            self.skill = np.random.randint(self.num_skills)
        # If an scalar, assume har_coded skill integer
        elif np.isscalar(self.skill_dist):
            self.skill = int(self.skill_dist)
        # If callable, assume that it will return a new skill
        elif callable(self.skill_dist):
            self.skill = self.skill_dist()
        # Otherwise, assume just a normal categorical distribution
        else:
            self.skill = np.random.choice(self.num_skills, p=self.skill_dist)

    def get_action(self, *observations):
        # Augment observations with a one-hot vector of current skill
        skill_one_hot = torch.zeros(self.num_skills)
        skill_one_hot[self.skill] = 1.0
        observations_aug = observations + (skill_one_hot,)
        # Get action as normal and include current skill in the info dict
        ac, info = super().get_action(*observations_aug)
        info['skill'] = self.skill
        return ac, info

    def forward(self, *observations):
        return self.base(*observations)


class SkillConditionedMixin:
    """
    Mixin for MLP-like policies that are conditioned on the current skill

    To mix with another policy PI, combine it like:

    class SkillConditionedPI(SkillConditionedMixin, PI):
        pass

    That way, the methods below will appear first in the MRO
    call order of the mixed class.
    """

    def __init__(self, ob_dim, num_skills, **kwargs):
        super().__init__(ob_dim=ob_dim + num_skills, **kwargs)
        self.num_skills = num_skills
        self.skill_dist = None
        self.skill = None

    def reset(self):
        super().reset()
        # If no prob dist, sample uniform
        if self.skill_dist is None:
            self.skill = np.random.randint(self.num_skills)
        # If an scalar, assume har_coded skill integer
        elif np.isscalar(self.skill_dist):
            self.skill = int(self.skill_dist)
        # If callable, assume that it will return a new skill
        elif callable(self.skill_dist):
            self.skill = self.skill_dist()
        # Otherwise, assume just a normal categorical distribution
        else:
            self.skill = np.random.choice(self.num_skills, p=self.skill_dist)

    def get_action(self, *observations):
        # Augment observations with a one-hot vector of current skill
        skill_one_hot = torch.zeros(self.num_skills)
        skill_one_hot[self.skill] = 1.0
        observations_aug = observations + (skill_one_hot,)
        # Get action as normal and include current skill in the info dict
        ac, info = super().get_action(*observations_aug)
        info['skill'] = self.skill
        return ac, info


class SkillConditionedTanhGaussianMLPPolicy(SkillConditionedMixin, TanhGaussianMLPPolicy):
    pass

class SkillConditionedTanhGMMMLPPolicy(SkillConditionedMixin, TanhGMMMLPPolicy):
    pass
