
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F



class SAC(nn.Module):

    def __init__(
        self,
        policy,
        qf1,
        qf2,
        log_alpha=0.0,
    ):
        super().__init__()
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.log_alpha = nn.Parameter(
            torch.as_tensor(log_alpha, dtype=torch.float32)
        )
        # Make copies of Q-functions for bootstrap targets
        self.qf1_target = copy.deepcopy(qf1)
        self.qf2_target = copy.deepcopy(qf2)


    @property
    def alpha(self):
        """Returns the current temperature by exponentiating its log value"""
        return self.log_alpha.exp().detach()


    def actor_objective(self, obs, target_entropy=None):

        # Sample actions, along with log(p(a|s)), from current policy
        pi = self.policy(obs)
        acs, log_prob = pi.sample_with_log_prob(grad=True)  # Enable r-sampling

        # Estimate value for each action and take the lower between q1 and q2
        q_values = torch.min(
            self.qf1(obs, acs), self.qf2(obs, acs)
        )

        # Climb the gradient of the (lower) q-function and add entropy bonus
        # -> loss is negative q + negative entropy
        policy_loss = (self.alpha * log_prob - q_values).mean()

        # Diagnostics
        info = {}
        info['Entropy'] = -log_prob.mean().detach()
        info['PolicyLoss'] = policy_loss.detach()

        # If not target entropy is requested, stop here
        if target_entropy is None:
            return policy_loss, info
        # Otherwise, generate a loss for alpha
        else:
            target_entropy_plus_logp = log_prob.detach() + target_entropy
            alpha_loss = (-self.log_alpha * target_entropy_plus_logp).mean()
            info['AlphaLoss'] = alpha_loss.detach()
            info['AlphaValue'] = self.alpha
            return policy_loss, alpha_loss, info


    def critic_objective(self, obs, acs, rews, dones, next_obs, gamma):

        # Make action-value predictions with both q-functions
        q1_pred = self.qf1(obs, acs)
        q2_pred = self.qf2(obs, acs)

        # Bootstrap target from next observation
        with torch.no_grad():

            # Sample actions and their log probabilities at next step
            pi = self.policy(next_obs)
            next_acs, next_acs_logp = pi.sample_with_log_prob()

            # Select the smallest estimate of action-value in the next step
            target_q_values_raw = torch.min(
                self.qf1_target(next_obs, next_acs),
                self.qf2_target(next_obs, next_acs),
            )

            # And add the weighted entropy bonus (negative log)
            target_q_values = target_q_values_raw - self.alpha * next_acs_logp

            # Combine with rewards using the Bellman recursion
            q_target = rews + (1. - dones) * gamma * target_q_values

        # Minimize squared distance
        qf1_mse = F.mse_loss(q1_pred, q_target)
        qf2_mse = F.mse_loss(q2_pred, q_target)

        # Diagonstics
        info = {} # For later
        info['QTarget'] = q_target.mean().detach()
        info['Qf1Loss'] = qf1_mse.detach()
        info['Qf2Loss'] = qf2_mse.detach()
        # info['qf1_pred'] = q1_pred.mean().detach()
        # info['qf2_pred'] = q2_pred.mean().detach()

        return qf1_mse, qf2_mse, info


    def update_critic_targets(self, tau):
        """
        Polyak averages the target critics

        Args:
            tau: strength of polyak averaging (1.0 -> full update)
        """
        assert 0.0 <= tau <= 1.0
        for qf, qf_target in [
            (self.qf1, self.qf1_target), (self.qf2, self.qf2_target)
        ]:
            for param, param_target in zip(
                qf.parameters(), qf_target.parameters()
            ):
                param_target.data.copy_(
                    (1.0 - tau) * param_target.data + tau * param.data
                )
