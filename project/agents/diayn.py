
import copy
import functools
import operator

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


class DIAYN(agents.Agent):

    class Config(agents.Agent.Config):

        # -- General
        env = 'HalfCheetah-v2'
        num_skills = 25
        gamma = 0.99  # Discount factor
        num_epochs = 500 #1_000
        num_samples_per_epoch = 1_000
        num_train_steps_per_epoch = 1_000
        batch_size = 256
        buffer_capacity = 1_000_000
        min_buffer_size = 10_000  # Min samples in replay buffer before training starts
        max_path_length_train = 1_000
        max_path_length_eval = 1_000
        eval_freq = 10 # Not so frequent since it is quite expensive
        eval_size = 4_000
        eval_video_freq = -1
        eval_video_size = (100, 100)
        eval_video_length = 200  # Max length for eval video
        eval_video_fps = 10

        # -- Policy
        policy_hidden_num = 2
        policy_hidden_size = 256
        policy_hidden_act = 'relu'
        policy_optimizer = 'adam'
        policy_lr = 3e-4
        policy_grad_norm_clip = None #10.0

        # -- Discriminator
        clf_hidden_num = 2
        clf_hidden_size = 256
        clf_hidden_act = 'relu'
        clf_optimizer = 'adam'
        clf_lr = 3e-4
        clf_grad_norm_clip = None #20.0

        # -- Critic
        critic_hidden_num = 2
        critic_hidden_size = 256
        critic_hidden_act = 'relu'
        target_update_tau = 5e-3  # Strength of target network polyak averaging
        target_update_freq = 1  # How often to update the target networks
        critic_optimizer = 'adam'
        critic_lr = 3e-4
        critic_grad_norm_clip = None #20.0

        # -- Temperature
        alpha_initial = 0.1  # (same value ass appendix )
        target_entropy = None  # Will be inferred from action space if None
        alpha_optimizer = 'adam'
        alpha_lr = 3e-4
        train_alpha = False  # Don't train alpha by default


    def init(self):
        """
        Initializes the DIAYN agent by setting up
            - Networks
                - A skill conditioned policy wrapping
                    an MLP-based tanh gaussian policy
                - A classifier for predicting current skill from observation
                - 2 MLP-based Q-functions: s x a x z -> q
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
        # Setup actor and critics
        self.policy = policies.SkillConditionedPolicy(
            base=policies.TanhGaussianMLPPolicy(
                ob_dim + self.cfg.num_skills, ac_dim,
                hidden_num=self.cfg.policy_hidden_num,
                hidden_size=self.cfg.policy_hidden_size,
                hidden_act=self.cfg.policy_hidden_act,
            ),
            num_skills=self.cfg.num_skills,
        )
        self.clf = networks.MLP(
            input_size=ob_dim,
            output_size=self.cfg.num_skills,
            hidden_num=self.cfg.clf_hidden_num,
            hidden_size=self.cfg.clf_hidden_size,
            hidden_act=self.cfg.clf_hidden_act,
        )
        self.qf1 = critics.QAMLPCritic(
            ob_dim + self.cfg.num_skills, ac_dim,
            hidden_num=self.cfg.critic_hidden_num,
            hidden_size=self.cfg.critic_hidden_size,
            hidden_act=self.cfg.critic_hidden_act,
        )
        self.qf2 = critics.QAMLPCritic(
            ob_dim + self.cfg.num_skills, ac_dim,
            hidden_num=self.cfg.critic_hidden_num,
            hidden_size=self.cfg.critic_hidden_size,
            hidden_act=self.cfg.critic_hidden_act,
        )

        # Temperature parameter used to weight the entropy bonus
        self.log_alpha = nn.Parameter(
            torch.as_tensor(self.cfg.alpha_initial, dtype=torch.float32).log()
        )

        # Make copies of Q-functions for bootstrap targets
        self.qf1_target = copy.deepcopy(self.qf1)
        self.qf2_target = copy.deepcopy(self.qf2)

        # And send everything to the right device
        self.to(self.device)

        # Setup optimizers for all networks (and log_alpha)
        self.policy_optimizer = utils.get_optimizer(
            name=self.cfg.policy_optimizer,
            params=self.policy.parameters(),
            lr=self.cfg.policy_lr,
        )
        self.clf_optimizer = utils.get_optimizer(
            name=self.cfg.clf_optimizer,
            params=self.clf.parameters(),
            lr=self.cfg.clf_lr,
        )
        self.qf1_optimizer = utils.get_optimizer(
            name=self.cfg.critic_optimizer,
            params=self.qf1.parameters(),
            lr=self.cfg.critic_lr,
        )
        self.qf2_optimizer = utils.get_optimizer(
            name=self.cfg.critic_optimizer,
            params=self.qf2.parameters(),
            lr=self.cfg.critic_lr
        )
        self.alpha_optimizer = utils.get_optimizer(
            name=self.cfg.alpha_optimizer,
            params=[self.log_alpha],
            lr=self.cfg.alpha_lr,
        )

        # Setup replay buffer
        self.buffer = buffers.RingBuffer(
            capacity=int(self.cfg.buffer_capacity),
            keys=[ 'ob',  'skill', 'ac',   'rew', 'next_ob', 'done'],
            dims=[ ob_dim, None,    ac_dim, None,  ob_dim,    None],
            dtypes=[float, int,     float,  float, float,     float]
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
            - Sample some more data
            - Loop for the configured number of train steps:
                - Train critics against Bellman bootstraps
                - Train actor against critic with entropy bonus
                - Train alpha based on current and target entropy
                - Polyak average the target networks of the critics
            - Evaluate the current policy
        """

        # Generate initial data for the replay buffer
        missing_data = self.cfg.min_buffer_size - len(self.buffer)
        if missing_data > 0:
            self.logger.info(f'Seeding buffer with {missing_data} samples')
            self.buffer << self.train_sampler.sample_steps(n=missing_data)

        self.logger.info('Begin training')
        for epoch in range(1, self.cfg.num_epochs + 1):

            # Create a logger for dumping diagnostics
            epoch_logs = self.logger.epoch_logs(epoch)

            # Sample more steps for this epoch and add to replay buffer
            self.buffer << self.train_sampler.sample_steps(
                n=self.cfg.num_samples_per_epoch
            )

            # Write statistics from training data sampling
            # data_info = self.buffer.get_info()
            data_info = {}
            data_info['BufferSize'] = len(self.buffer)
            data_info['TotalEnvSteps'] = self.train_sampler.total_steps
            data_info['Last25TrainRets'] = self.train_sampler.returns[-25:]
            epoch_logs.add_scalar_dict(data_info, prefix='Data')

            # Train with data from replay buffer
            for step in range(self.cfg.num_train_steps_per_epoch):

                # Sample batch (and convert all to torch tensors)
                obs, skills, acs, _, next_obs, dones = self.buffer.sample(
                    self.cfg.batch_size,
                    tensor=True,
                    device=self.device,
                    as_dict=False,
                )

                # Train q functions
                clf_loss, qf1_loss, qf2_loss, critic_info = self.critic_objective(
                    obs=obs, skills=skills, acs=acs, next_obs=next_obs, dones=dones
                )
                critic_info['ClfGradNorm'] = utils.optimize(
                    loss=clf_loss,
                    optimizer=self.clf_optimizer,
                    norm_clip=self.cfg.clf_grad_norm_clip,
                )
                critic_info['Qf1GradNorm'] = utils.optimize(
                    loss=qf1_loss,
                    optimizer=self.qf1_optimizer,
                    norm_clip=self.cfg.critic_grad_norm_clip,
                )
                critic_info['Qf2GradNorm'] = utils.optimize(
                    loss=qf2_loss,
                    optimizer=self.qf2_optimizer,
                    norm_clip=self.cfg.critic_grad_norm_clip,
                )
                epoch_logs.add_scalar_dict(critic_info, prefix='Critic')

                # Train actor and tune temperature
                policy_loss, alpha_loss, policy_info = self.actor_objective(
                    obs=obs, skills=skills
                )
                policy_info['PolicyGradNorm'] = utils.optimize(
                    loss=policy_loss,
                    optimizer=self.policy_optimizer,
                    norm_clip=self.cfg.policy_grad_norm_clip,
                )
                if self.cfg.train_alpha:
                    utils.optimize(alpha_loss, self.alpha_optimizer)
                epoch_logs.add_scalar_dict(policy_info, prefix='Actor')

                # Apply polyak averaging to target networks
                utils.polyak(
                    net=self.qf1, target=self.qf1_target,
                    tau=self.cfg.target_update_tau
                )
                utils.polyak(
                    net=self.qf2, target=self.qf2_target,
                    tau=self.cfg.target_update_tau
                )

            # Eval, on occasions, but for every skill
            if epoch % self.cfg.eval_freq == 0:
                self.logger.info('Evaluating %s skills', self.cfg.num_skills)
                # Aggregate list of evaluations for each skil
                eval_infos = []
                render = (
                    self.cfg.eval_video_freq > 0 and
                    epoch % self.cfg.eval_video_freq == 0
                )
                for skill in range(self.cfg.num_skills):
                    with self.policy.configure(greedy=True, skill_dist=skill):
                        eval_info, eval_frames = self.eval_sampler.evaluate(
                            n=self.cfg.eval_size,
                            render=render,
                            render_max=self.cfg.eval_video_length,
                            render_size=self.cfg.eval_video_size,
                        )
                    eval_infos.append(eval_info)
                    if not eval_frames is None:
                        epoch_logs.add_video(
                            name='EvalRollout',
                            video=eval_frames,
                            fps=self.cfg.eval_video_fps,
                        )
                # Combine: list of dicts of lists -> dict of lists
                eval_infos = {
                    k: functools.reduce(operator.add, (i[k] for i in eval_infos))
                    for k in eval_infos[0]
                }
                epoch_logs.add_scalar_dict(
                    eval_infos, prefix='Eval', agg=['min', 'max', 'mean']
                )
            # Write logs
            epoch_logs.dump(step=self.train_sampler.total_steps)

    def critic_objective(self, obs, skills, acs, next_obs, dones):
        """
        Generates losses for the discriminator and the critics

        Args:
            obs (Tensor<N,O>) a batch of obervations, each O floats
            skills (Tensor<N>) a batch of skills, each a single integer
            acs (Tensor<N,A>) a batch of actions, each a float
            next_obs (Tensor<N,O>) a batch of next observations (see obs)
            dones (Tensor<N>) a batch of done flags, each a single float

        Returns:
            A differentiable discriminator loss (cross entropy),
            differentiable losses (entropy bellman mse) for both critics,
            and a dictionary with detached diagnostics.
        """

        # Generate standard cross-entropy loss for the discriminator
        logits = self.clf(obs)
        clf_xe = F.cross_entropy(logits, skills, reduction='none')
        clf_loss = clf_xe.mean()

        # Then, use the cross-entropy to generate a synthetic reward
        rews = -1 * clf_xe.detach()
        # Subtract (uniform) log likelihood
        p_skill = torch.tensor(1. / self.cfg.num_skills, device=rews.device)
        rews -= torch.log(p_skill)

        # Convert skills to one-hot so the fit into the policy and critics
        skills_one_hot = utils.to_one_hot(skills, self.cfg.num_skills)

        # Make action-value predictions with both q-functions
        q1_pred = self.qf1(obs, skills_one_hot, acs)
        q2_pred = self.qf2(obs, skills_one_hot, acs)

        # Bootstrap target from next observation
        with torch.no_grad():

            # Sample actions and their log probabilities at next step
            pi = self.policy(next_obs, skills_one_hot)
            next_acs, next_acs_logp = pi.sample_with_log_prob()

            # Select the smallest estimate of action-value in the next step
            target_q_values_raw = torch.min(
                self.qf1_target(next_obs, skills_one_hot, next_acs),
                self.qf2_target(next_obs, skills_one_hot, next_acs),
            )

            # And add the weighted entropy bonus (negative log)
            target_q_values = target_q_values_raw - self.alpha * next_acs_logp

            # Combine with rewards using the Bellman recursion
            q_target = rews + (1. - dones) * self.cfg.gamma * target_q_values

        # Minimize squared distance
        qf1_loss = F.mse_loss(q1_pred, q_target)
        qf2_loss = F.mse_loss(q2_pred, q_target)

        # Diagonstics
        info = {} # For later
        info['ClfLoss'] = clf_loss.detach()
        info['ClfAccuracy'] = (logits.argmax(dim=-1) == skills).float().mean()
        info['ClfReward'] = rews.mean()
        info['QTarget'] = q_target.mean().detach()
        info['QAbsDiff'] = (q1_pred - q2_pred).abs().mean().detach()
        info['Qf1Loss'] = qf1_loss.detach()
        info['Qf2Loss'] = qf2_loss.detach()

        return clf_loss, qf1_loss, qf2_loss, info

    def actor_objective(self, obs, skills):
        """
        Generates losses for the policy and alpha parameter

        Args:
            obs (Tensor<N,O>) a batch of observations, each O floats
            skills (Tensor<N>) a batch of skills, each a single integer

        Returns:
            A differentiable policy_loss, a differentiable (log) alpha loss,
            and a dictionary with detached diagnostic info
        """
        # Convert skills to one-hot
        skills_one_hot = utils.to_one_hot(skills, self.cfg.num_skills)

        # Sample actions, along with log(p(a|s)), from current policy
        pi = self.policy(obs, skills_one_hot)
        acs, log_prob = pi.sample_with_log_prob(grad=True)  # Enable r-sampling

        # Estimate value for each action and take the lower between q1 and q2
        q_values = torch.min(
            self.qf1(obs, skills_one_hot, acs),
            self.qf2(obs, skills_one_hot, acs)
        )

        # Climb the gradient of the (lower) q-function and add entropy bonus
        # -> loss is negative q + negative entropy
        policy_loss = (self.alpha * log_prob - q_values).mean()

        # Generate loss for the alpha value
        target_entropy_plus_logp = log_prob.detach() + self.target_entropy
        alpha_loss = (-self.log_alpha * target_entropy_plus_logp).mean()

        # Diagnostics
        info = {}
        info['Entropy'] = -log_prob.mean().detach()
        info['PolicyLoss'] = policy_loss.detach()
        info['AlphaLoss'] = alpha_loss.detach()
        info['AlphaValue'] = self.alpha

        return policy_loss, alpha_loss, info



if __name__ == '__main__':
    DIAYN.run_cli()
