

import torch


class PolicyDistribution:

    def sample(self, greedy=False, grad=False):
        raise NotImplementedError()


    def log_prob(self, action):
        raise NotImplementedError()


    def sample_with_log_prob(self, greedy=False, grad=False):
        action = self.rsample(greedy=greedy, grad=grad)
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


    def log_prob_from_pre_tanh(self, action, action_pre_tanh, eps=1e-6):
        """
        Computes log prob based on action before and after tanh sqeeze

        (From SAC Paper appendix C)

        Given action a = tanh(u), where u~N(u|s). The density is given by:
        - pi(a|s) = N(u|s) * |det(da/du)|^-1
        The jacobian of the element-wise tanh is diag(1 - tanh^2(u)), so:
        - pi(a|s) = N(u|s) * PROD_i[log(1 - tanh^2(u_i))]
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
        return logp_gaus - torch.log(da_du_clipped + eps).sum(dim=-1)
