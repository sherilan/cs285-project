

import torch
import torch.nn.functional as F
import numpy as np


class PolicyDistribution:

    def sample(self, greedy=False, grad=False):
        raise NotImplementedError()


    def log_prob(self, action):
        raise NotImplementedError()


    def sample_with_log_prob(self, greedy=False, grad=False):
        action = self.sample(greedy=greedy, grad=grad)
        logp = self.log_prob(action)
        return action, logp



class Gaussian(PolicyDistribution):

    def __init__(self, mean, std):
        self.normal = torch.distributions.Independent(
            torch.distributions.Normal(mean, std),
            reinterpreted_batch_ndims=-1,
        )


    def sample(self, greedy=False, grad=False):
        # Case: greedy -> return the mean of the gaussian
        if greedy:
            if grad:
                return self.normal.mean
            else:
                return self.normal.mean.detach()
        # Case: stochastic -> simply sample the normal
        else:
            if grad:
                return self.normal.rsample()
            else:
                return self.normal.sample()


    def log_prob(self, action):
        return self.normal.log_prob(action)



class TanhGaussian(Gaussian):

    # TODO: Clean up to look more like TanhGMM
    #       I just don't want to touch anything right now

    def __init__(self, *args, use_spinup_logp=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_spinup_logp = use_spinup_logp

    def sample(self, greedy=False, grad=False):
        return torch.tanh(super().sample(greedy=greedy, grad=grad))

    def log_prob(self, action):
        """
        WARNING: be careful with this one due to inverting tanh
        """
        # Calculate pre_tanh action with arctanh
        action = action.detach()
        action_pre_tanh = torch.log((1 + action) / (1 - action)) / 2
        return self.log_prob_from_pre_tanh(action, action_pre_tanh)

        return super().log_prob(torch.arctanh(action))

    def sample_with_log_prob(self, greedy=False, grad=False):
        """
        Samples actions and returns it with its log prob
        Calculating the logp before the tanh greatly helps
        with numerical stability (so we don't have to arctan).
        """
        action_pre_tanh = super().sample(greedy=greedy, grad=grad)
        action = torch.tanh(action_pre_tanh)
        logp = self.log_prob_from_pre_tanh(action, action_pre_tanh)
        return action, logp

    def log_prob_from_pre_tanh(self, action, action_pre_tanh):
        if self.use_spinup_logp:
            return self.log_prob_from_pre_tanh_spinup(action_pre_tanh)
        else:
            return self.log_prob_from_pre_tanh_sac(action, action_pre_tanh)

    def log_prob_from_pre_tanh_sac(self, action, action_pre_tanh):
        """
        Computes log prob based on action before and after tanh sqeeze

        (From SAC Paper appendix C)

        Given action a = tanh(u), where u~N(u|s). The density is given by:
        - pi(a|s) = N(u|s) * |det(da/du)|^-1
        The jacobian of the element-wise tanh is diag(1 - tanh^2(u)), so:
        - pi(a|s) = N(u|s) * PROD_i[1 - tanh^2(u_i)]
        Which, when taking the logarithm, becomes:
        - log(pi(a|s)) = log(N(u|s)) + SUM_i[log(1 - tanh^2(u_i))]
        """
        # Compute logp under the underlying gaussian
        logp_gaus = super().log_prob(action_pre_tanh)
        # Compute elementwise da_du (action = tanh(u))
        da_du = 1 - action ** 2
        # Use clip trick from garage impl to make sure da_du is in (0,1)
        # while preserving gradient.
        clip_hi = (da_du > 1.0).float()
        clip_lo = (da_du < 0.0).float()
        with torch.no_grad():
            clip_diff = (1.0 - da_du) * clip_hi + (0.0 - da_du) * clip_lo
        da_du_clipped = da_du + clip_diff
        # Combine: log of product -> sum of logs
        return logp_gaus - torch.log(da_du_clipped + 1e-6).sum(dim=-1)

    def log_prob_from_pre_tanh_spinup(self, action_pre_tanh):
        """
        Variation on log_prob_from_pre_tanh_sac sourced from the
        "Spinning Up" repository that is supposed to be more
        numerically stable.
        """
        logp_gaus = super().log_prob(action_pre_tanh)
        log_det_da_du = 2 * (
            + np.log(2)
            - action_pre_tanh
            - F.softplus(-2 * action_pre_tanh)
        )
        return logp_gaus - log_det_da_du.sum(dim=-1)



class GMM(PolicyDistribution):

    def __init__(self, logits, means, logstds, q_values=None):
        self.logits = logits
        self.means = means
        self.logstds = logstds
        self.q_values = q_values

    @property
    def b_dim(self):
        return self.means.shape[:-2]

    @property
    def k_dim(self):
        return self.means.shape[-2]

    @property
    def a_dim(self):
        return self.means.shape[-1]

    def reg_loss(self):
        loss = 0
        loss += 0.5 * (self.means ** 2).mean()
        loss += 0.5 * (self.logstds ** 2).mean()
        return loss

    def sample(self, greedy=False, grad=False):

        # Case: greedy -> select mean of strongest mixture
        if greedy:
            # Case: no q-values -> select component with highest mixture prob
            if self.q_values is None:
                z = self.logits.argmax(dim=-1)
            # Case: q-values -> argmax over them instead
            else:
                z = self.q_values.argmax(dim=-1)

            # Create a softmax-based path fort the gradient
            grad_path = self.logits.softmax(dim=-1)

            # Create a 1-hot mask of the argmax values
            z_mask = torch.zeros_like(self.logits).scatter_(-1, z, 1.0)

            # Add zero to the argmax mask, but keep the gradient of the
            # positive term. That way, the ones in the mask will get
            # gradient proportional to the softmax strength
            mask = z_mask - grad_path.detach() + grad_path

            # Select from the means
            action = (self.means * mask.unsqueeze(-1)).sum(dim=-2)

        # Case: stochastic -> properly sample mixture
        else:
            # Get components with a differentiable gumbel softmax distribution
            mask = F.gumbel_softmax(self.logits, hard=True, dim=-1).unsqueeze(-1)
            mean = (self.means * mask).sum(dim=-2)
            logstd = (self.logstds * mask).sum(dim=-2)
            # Then use standard reparametrization on the result
            eps = torch.randn(self.b_dim + (self.a_dim,), device=mean.device)
            action = mean + logstd.exp() * eps

        return action if grad else action.detach()


    def log_prob(self, action):
        # Compute log p(x|z) = log N(m(z), s(z))
        logp_given_z = self.log_normal(
            action[...,None,:], self.means, self.logstds
        )

        # Compute log p(x) = log sum(p(z_i)*p(x|z_i))
        #                  = log sum(exp(log(z_i) + log(x|z_i)))
        logp = (
            torch.logsumexp(self.logits + logp_given_z, dim=-1)
          - torch.logsumexp(self.logits, dim=-1)  # Normalizing factor
        )
        return logp


    def log_normal(self, x, mean, logstd):
        """
        Computes the logarithm of a (diagonal) multivariate normal PDF
        evaluated at x
        """
        # Squared value in the exponent of a gaussian
        dist = (x - mean) * torch.exp(-logstd)
        quadratic = -0.5 * (dist ** 2).sum(dim=-1)
        # Divisor of ormalizing factor (1/(sig * sqrt(2 * PI)))
        # The right term is multiplied by x_dim so that the value is repeated
        # for each dimension in the (diagonal) multivariate
        c = logstd.sum(dim=-1) + self.a_dim * 0.5 * np.log(2 * np.pi)
        # Combine, negative c since it is log of a divisor
        return quadratic - c


class TanhGMM(GMM):
    """GMM, except with a tanh squeeze"""

    def __init__(self, *args, use_spinup_logp=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_spinup_logp = use_spinup_logp

    def sample(self, greedy=False, grad=False):
        return torch.tanh(super().sample(greedy=greedy, grad=grad))

    def log_prob(self, action):
        """
        WARNING: be careful with this one due to inverting tanh
        """
        # Calculate pre_tanh action with arctanh
        action = action.detach()
        action_pre_tanh = torch.log((1 + action) / (1 - action)) / 2
        logp_pre_tanh = super().log_prob(action_pre_tanh)
        logp = logp_pre_tanh + self.logp_correction(action_pre_tanh)
        return logp

    def sample_with_log_prob(self, greedy=False, grad=False, pre_tanh=False):
        """
        Samples actions and returns it with its log prob
        Calculating the logp before the tanh greatly helps
        with numerical stability (so we don't have to arctan).
        """
        # Sample action and compute logp, both before and after tanh squeeze
        action_pre_tanh = super().sample(greedy=greedy, grad=grad)
        action = torch.tanh(action_pre_tanh)
        logp_pre_tanh = super().log_prob(action_pre_tanh)
        logp = logp_pre_tanh + self.logp_correction(action_pre_tanh)
        # Optionally return all 4 quantities (needed for vanilla sac)
        if pre_tanh:
            return action, logp, action_pre_tanh, logp_pre_tanh
        else:
            return action, logp

    def logp_correction(self, action_pre_tanh):
        """Get the correction term for logp after tanh sqeeze"""
        if self.use_spinup_logp:
            return tanh_logp_correction_spinup(action_pre_tanh)
        else:
            return tanh_logp_correction_sac(action_pre_tanh)



def tanh_logp_correction_sac(action_pre_tanh):
    """
    Computes the logp correction term for a = tanh(p(u|s))

    (From SAC Paper appendix C)

    Given action a = tanh(u), where u~N(u|s). The density is given by:
    - pi(a|s) = N(u|s) * |det(da/du)|^-1
    The jacobian of the element-wise tanh is diag(1 - tanh^2(u)), so:
    - pi(a|s) = N(u|s) * PROD_i[1 - tanh^2(u_i)]
    Which, when taking the logarithm, becomes:
    - log(pi(a|s)) = log(N(u|s)) + SUM_i[log(1 - tanh^2(u_i))]

    This function computes the SUM_i[log(1 - tanh^2(u_i))] part
    """
    # Compute elementwise da_du (action = tanh(u))
    da_du = 1 - torch.tanh(action_pre_tanh) ** 2
    # Use clip trick from garage impl to make sure da_du is in (0,1)
    # while preserving gradient.
    clip_hi = (da_du > 1.0).float()
    clip_lo = (da_du < 0.0).float()
    with torch.no_grad():
        clip_diff = (1.0 - da_du) * clip_hi + (0.0 - da_du) * clip_lo
    da_du_clipped = da_du + clip_diff
    # Combine: log of product -> sum of logs
    return - torch.log(da_du_clipped + 1e-6).sum(dim=-1)


def tanh_logp_correction_spinup(action_pre_tanh):
    """
    Variation on log_prob_from_pre_tanh_sac sourced from the
    "Spinning Up" repository that is supposed to be more
    numerically stable.
    """
    corr = -2 * (
        + np.log(2)
        - action_pre_tanh
        - F.softplus(-2 * action_pre_tanh)
    )
    return corr.sum(dim=-1).squeeze(-1)


def tanh_logp_correction_eysenbach(action_pre_tanh):
    """
    The simplest corection of tanh logp. Used in Eysenbach's repo
    """
    return - torch.log(1 - torch.tanh(action_pre_tanh) ** 2 + 1e-6).sum(dim=-1)
