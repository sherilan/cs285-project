"""
A variation of SAC trying to match the original implementation of Haarnoja

This is a pytorch adapation of the implementation found in
- https://github.com/haarnoja/sac/tree/master/sac

But using the default hyperparameters of the fork:
- https://github.com/ben-eysenbach/sac/blob/master/examples/mujoco_all_sac.py

It differs from more conventional implementations of SAC in that:

 * A gaussian mixture (GMM) distribution is used as the policy.

 * The policy is updated with conventional policy gradient against
   (gradient-free) Q-targets. In contrast most other SAC implementations
   use a DDPG-style update where the policy climbs the gradient
   of the Q-function.

 * Since a standard policy gradient is used, there is also a baseline
   (value function) to reduce the variance of the gradient estimator.

 * The baseline is used as targets for the Q-functions.


"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import project.agents as agents
import project.policies as policies
import project.networks as networks
import project.utils as utils
import project.critics as critics
import project.buffers as buffers
import project.samplers as samplers
import project.normalizers as normalizers


class SACGMM(agents.Agent):

    class Config(agents.Agent.Config):

        # -- General
        env = 'HalfCheetah-v2'
        gamma = 0.99  # Discount factor
        num_epochs = 10_000
        num_steps_per_epoch = 1_000
        num_train_steps = 1
        batch_size = 128
        buffer_capacity = 10_000_000
        min_buffer_size = 1_000  # Min samples in replay buffer before training starts
        max_path_length_train = 1_000

        # -- Evaluation and logging
        eval_freq = 50
        eval_size = 10_000
        eval_video_freq = -1
        eval_video_length = 200  # Max length for eval video
        eval_video_size = [200, 200]
        eval_video_fps = 10
        max_path_length_eval = 1_000


        LR = 3e-4
        WIDTH = 128

        # -- Policy
        policy_hidden_num = 2
        policy_num_components = 4
        policy_hidden_size = WIDTH
        policy_hidden_act = 'relu'
        policy_optimizer = 'adam'
        policy_lr = LR
        policy_grad_norm_clip = None #10.0
        policy_gmm_reg = 0.001

        # -- Value Function (baseline)
        vf_hidden_num = 2
        vf_hidden_size = WIDTH
        vf_hidden_act = 'relu'
        vf_optimizer = 'adam'
        vf_lr = LR
        vf_grad_norm_clip = None

        # -- Q-function (critic)
        critic_hidden_num = 2
        critic_hidden_size = WIDTH
        critic_hidden_act = 'relu'
        target_update_tau = 0.01  # Strength of target network polyak averaging
        target_update_freq = 1  # How often to update the target networks
        critic_optimizer = 'adam'
        critic_lr = LR
        critic_grad_norm_clip = None

        # -- Temperature
        alpha_initial = 1.0  # Haarnoja's  SAC doesn't do entropy scaling
        target_entropy = None
        alpha_optimizer = 'adam'
        alpha_lr = LR
        train_alpha = False  # ... so naturally, it doesn't train alpha either



    def init(self):
        """
        Initializes the SAC agent by setting up
            - Networks
                - An MLP-based tanh gaussian policy: o -> a
                - An MLP-based v-function (baseline): o -> v
                - An MLP-based q-function (critic): o x a -> v
                - A tunably entropy weight (temperature/alpha)
            - Optimizers, one for each of the aforementioned
            - A simple ring-buffer for experience replay
            - Samplers for gathering training data and evaluating
            - Target entropy for the soft policy
        """
        # Initialize environment to get input/output dimensions
        self.train_env = utils.make_env(self.cfg.env)
        self.eval_env = utils.make_env(self.cfg.env)
        ob_dim, = self.train_env.observation_space.shape
        ac_dim, = self.train_env.action_space.shape
        # Setup policy, baseline, and critic
        self.policy = policies.TanhGMMMLPPolicy(
            ob_dim, ac_dim,
            num_components=self.cfg.policy_num_components,
            hidden_num=self.cfg.policy_hidden_num,
            hidden_size=self.cfg.policy_hidden_size,
            hidden_act=self.cfg.policy_hidden_act,
        )
        self.vf = critics.MLPBaseline(
            ob_dim,
            hidden_num=self.cfg.vf_hidden_num,
            hidden_size=self.cfg.vf_hidden_size,
            hidden_act=self.cfg.vf_hidden_act,
        )
        self.qf = critics.QAMLPCritic(
            ob_dim, ac_dim,
            hidden_num=self.cfg.critic_hidden_num,
            hidden_size=self.cfg.critic_hidden_size,
            hidden_act=self.cfg.critic_hidden_act,
        )
        # Temperature parameter used to weight the entropy bonus
        self.log_alpha = nn.Parameter(
            torch.as_tensor(self.cfg.alpha_initial, dtype=torch.float32).log()
        )

        # Make copy of baseline for Q-targets.
        self.vf_target = copy.deepcopy(self.vf)

        # And send everything to the right device
        self.to(self.device)

        # Setup optimizers for all networks (and log_alpha)
        self.policy_optimizer = utils.get_optimizer(
            name=self.cfg.policy_optimizer,
            params=self.policy.parameters(),
            lr=self.cfg.policy_lr,
        )
        self.vf_optimizer = utils.get_optimizer(
            name=self.cfg.vf_optimizer,
            params=self.vf.parameters(),
            lr=self.cfg.vf_lr,
        )
        self.qf_optimizer = utils.get_optimizer(
            name=self.cfg.critic_optimizer,
            params=self.qf.parameters(),
            lr=self.cfg.critic_lr,
        )
        self.alpha_optimizer = utils.get_optimizer(
            name=self.cfg.alpha_optimizer,
            params=[self.log_alpha],
            lr=self.cfg.alpha_lr,
        )

        # Setup replay buffer
        self.buffer = buffers.RingBuffer(
            capacity=int(self.cfg.buffer_capacity),
            keys=['ob',  'ac',   'rew', 'next_ob', 'done'],
            dims=[ob_dim, ac_dim, None,  ob_dim,    None]
        )

        # Setup samplers (used for data generating / evaluating rollouts)
        self.train_sampler = samplers.Sampler(
            env=self.train_env,
            policy=self.policy,
            max_steps=self.cfg.max_path_length_train
        )
        self.eval_sampler = samplers.Sampler(
            env=self.eval_env,
            policy=self.policy,
            max_steps=self.cfg.max_path_length_eval
        )

        # Set target entropy, derive from size of action space if non-obvious
        if self.cfg.target_entropy is None:
            self.target_entropy = -ac_dim
            self.logger.info(
                'Using dynamic target entropy: %s', self.target_entropy
            )
        else:
            self.target_entropy = self.cfg.target_entropy
            self.logger.info(
                'Using static target entropy: %s', self.target_entropy
            )

    @property
    def alpha(self):
        """Un-logs the temperature parameter to get the entropy weight"""
        return self.log_alpha.exp().detach()

    def train(self):
        """
        Trains the SAC agent with the following procedure:
        - Sample initial data for the replay buffer
        - Loop for the configured number of epochs:
            - Loop for the configured number of steps per epoch:
                - Sample one environment transition
                - Take a configured number of gradient steps
            - Evaluate the current policy (sometimes)
        """

        # Generate initial data for the replay buffer
        missing_data = self.cfg.min_buffer_size - len(self.buffer)
        if missing_data > 0:
            self.logger.info(f'Seeding buffer with {missing_data} samples')
            self.buffer << self.train_sampler.sample_steps(missing_data)

        self.logger.info('Begin training')
        for epoch in range(1, self.cfg.num_epochs + 1):

            # Create a logger for dumping diagnostics
            epoch_logs = self.logger.epoch_logs(epoch)

            # Loop for the configured number of env steps in each epoch
            for step in range(self.cfg.num_steps_per_epoch):

                # Advance environment one more step
                self.buffer << self.train_sampler.sample_steps(1)

                # And then train
                for train_step in range(self.cfg.num_train_steps):

                    # Sample batch (and convert all to torch tensors)
                    obs, acs, rews, next_obs, dones = self.buffer.sample(
                        self.cfg.batch_size,
                        tensor=True,
                        device=self.device,
                        as_dict=False,
                    )

                    # Train q function
                    critic_info = self.update_critic(obs, acs, rews, next_obs, dones)
                    epoch_logs.add_scalar_dict(critic_info, prefix='Critic')

                    # Train policy and baseline
                    actor_info = self.update_actor(obs)
                    epoch_logs.add_scalar_dict(actor_info, prefix='Actor')

                # Apply polyak averaging to target networks
                self.update_target()

            # Eval, on occasions
            if epoch % self.cfg.eval_freq == 0:
                eval_info, eval_frames = self.evaluate(epoch, greedy=True)
                epoch_logs.add_scalar_dict(
                    eval_info, prefix='Eval', agg=['min', 'max', 'mean']
                )
                if not eval_frames is None:
                    epoch_logs.add_video(
                        'EvalRollout', eval_frames, fps=self.cfg.eval_video_fps,
                    )

            # Add information about replay buffer to logs and dump them
            epoch_logs.add_scalar_dict(self.get_data_info(), prefix='Data')
            epoch_logs.dump(step=self.train_sampler.total_steps)


    def get_data_info(self, debug=False):
        """
        Generates diagnostics about the gathered data
        """
        info = {}
        info['BufferSize'] = len(self.buffer)
        info['TotalEnvSteps'] = self.train_sampler.total_steps
        info['Last25TrainRets'] = self.train_sampler.returns[-25:]
        # Optionally add summary statistics about everything in buffer
        if debug:
            info.update(self.buffer.get_info())

        return info

    def update_critic(self, obs, acs, rews, next_obs, dones):
        """
        Updates parameters of the critic (Q-function)

        The key difference between this function and the one in
        sac.py is that the Bellman bootstraps simply use the value function
        (vf) to estimate the value of the next state (as opposed
        to a combination of Q-functions and the policy).

        Args:
            obs (Tensor<N,O>): A batch of obervations, each O floats
            acs (Tensor<N,A>): A batch of actions, each a float
            rews (Tensor<N>): A batch of rewards, each a single float
            next_obs (Tensor<N,O>): A batch of next observations (see obs)
            dones (Tensor<N>): A batch of done flags, each a single float

        Returns:
            A dictionary with diagnostics
        """
        # Make action-value predictions with both q-functions
        qf_pred = self.qf(obs, acs)

        # Bootstrap target from next observation
        with torch.no_grad():

            # Train against vf estimate of value in next state
            # It should already have the entropy-bonus baked in, so no
            # need to include it here.
            v_values = self.vf_target(next_obs)

            # Combine with rewards using the Bellman recursion
            q_target = rews + (1. - dones) * self.cfg.gamma * v_values

        # Use mean squared error as loss
        qf_loss = F.mse_loss(qf_pred, q_target)

        # And minimize it
        qf_grad_norm = utils.optimize(
            loss=qf_loss,
            optimizer=self.qf_optimizer,
            norm_clip=self.cfg.critic_grad_norm_clip,
        )

        # Diagonstics
        info = {}
        info['QTarget'] = q_target.mean().detach()
        info['QfLoss'] = qf_loss.detach()
        info['QfGradNorm'] = qf_grad_norm

        return info

    def update_actor(self, obs):
        """
        Updates parameters for the policy, vf (and possibly alpha)

        This function is fundamentally different from the corresponding
        update in the conventional SAC implementation found in this
        repository. Instead of climbing the gradient of the Q-function
        (like DDPG), a standard policy gradient is used.

        Args:
            obs (Tensor<N,O>): A batch of observations, each O floats

        Returns:
            A differentiable policy_loss, a differentiable (log) alpha loss,
            and a dictionary with detached diagnostic info
        """
        # Sample actions, along with log(p(a|s)), from current policy
        pi = self.policy(obs)
        acs, log_prob, _, logp_pre_tanh = pi.sample_with_log_prob(pre_tanh=True)

        # Generate vf estimate to reduce the variance
        v_values = self.vf(obs)

        # Generate Actor-Critic estimate for return
        with torch.no_grad():
            # Estimate value for each action
            q_values = self.qf(obs, acs)
            # Compute entropy bonus
            h_bonuses = self.alpha * -log_prob
            # And combine
            policy_objective = q_values - v_values + h_bonuses

        # Policy gradient loss (with gmm regularization)
        policy_loss = - (logp_pre_tanh * policy_objective).mean()
        policy_loss += self.cfg.policy_gmm_reg * pi.reg_loss()
        policy_grad_norm = utils.optimize(
            loss=policy_loss,
            optimizer=self.policy_optimizer,
            norm_clip=self.cfg.policy_grad_norm_clip,
        )

        # Train vf
        vf_loss = F.mse_loss(v_values, q_values + h_bonuses)
        vf_grad_norm = utils.optimize(
            loss=vf_loss,
            optimizer=self.vf_optimizer,
            norm_clip=self.cfg.vf_grad_norm_clip
        )

        # Generate a loss for the alpha value
        target_entropy_plus_logp = log_prob.detach() + self.target_entropy
        alpha_loss = (-self.log_alpha * target_entropy_plus_logp).mean()

        # But only optimize it if configured
        if self.cfg.train_alpha:
            utils.optimize(alpha_loss, self.alpha_optimizer)

        # Diagnostics
        info = {}
        info['Entropy'] = -log_prob.mean().detach()
        info['PolicyLoss'] = policy_loss.detach()
        info['PolicyGradNorm'] = policy_grad_norm
        info['VfLoss'] = vf_loss.detach()
        info['VfGradNorm'] = vf_grad_norm
        info['AlphaLoss'] = alpha_loss.detach()
        info['AlphaValue'] = self.alpha

        return info

    def update_target(self):
        """
        Applies polyak averaging to the target network of the vf (baseline)
        """
        utils.polyak(
            net=self.vf, target=self.vf_target,
            tau=self.cfg.target_update_tau
        )

    def evaluate(self, epoch=None, render=False, greedy=True):
        """Evaluates the current policy"""
        # Determine whether we should render for this epoch
        if render:
            render = True
        elif epoch is None:
            render = False
        elif self.cfg.eval_video_freq <= 0:
            render = False
        else:
            eval_num = epoch // self.cfg.eval_freq
            render = eval_num % self.cfg.eval_video_freq == 0

        # Rollout. Select greedy with action based on q-value
        with self.policy.configure(greedy=greedy, qf=self.qf):
            info, frames = self.eval_sampler.evaluate(
                n=self.cfg.eval_size,
                render=render,
                render_max=self.cfg.eval_video_length,
                render_size=tuple(self.cfg.eval_video_size),
            )

        return info, frames




if __name__ == '__main__':
    SACGMM.run_cli()
