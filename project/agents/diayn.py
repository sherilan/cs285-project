

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
import project.normalizers as normalizers
import project.envs as envs


class DIAYN(agents.Agent):

    class ConfigX(agents.Agent.Config):

        # -- General
        env = 'HalfCheetah-v2'
        num_skills = 50
        gamma = 0.99  # Discount factor
        num_epochs = 2_000
        num_samples_per_epoch = 250
        num_train_steps_per_epoch = 500
        batch_size = 128
        buffer_capacity = 1_000_000
        min_buffer_size = 1_000  # Min samples in replay buffer before training starts
        max_path_length_train = 1_000

        # -- Evaluation
        eval_freq = 100
        eval_size = 1_000
        eval_video_freq = -1
        eval_video_length = 200  # Max length for eval video
        eval_video_size = [100, 100]
        eval_video_fps = 10
        max_path_length_eval = 1_000

        LR = 1e-3
        WIDTH = 384

        # -- Policy
        policy_hidden_num = 2
        policy_hidden_size = WIDTH
        policy_hidden_act = 'relu'
        policy_optimizer = 'adam'
        policy_lr = LR
        policy_grad_norm_clip = None #10.0

        # -- Discriminator
        clf_hidden_num = 2
        clf_hidden_size = 128
        clf_hidden_act = 'relu'
        clf_optimizer = 'adam'
        clf_lr = LR
        clf_grad_norm_clip = None #20.0

        # -- Critic
        critic_hidden_num = 2
        critic_hidden_size = WIDTH
        critic_hidden_act = 'relu'
        target_update_tau = 5e-3  # Strength of target network polyak averaging
        target_update_freq = 1  # How often to update the target networks
        critic_optimizer = 'adam'
        critic_lr = LR
        critic_grad_norm_clip = None #20.0

        # -- Temperature
        alpha_initial = 0.1
        target_entropy = None  # Will be inferred from action space if None
        alpha_optimizer = 'adam'
        alpha_lr = LR
        train_alpha = False # Disable training of alpha if this is set

    class Config(agents.Agent.Config):

        # -- General
        env = 'HalfCheetah-v2'
        num_skills = 50
        gamma = 0.99  # Discount factor
        num_epochs = 10_000
        num_samples_per_epoch = 1000
        num_train_steps_per_epoch = 1000
        batch_size = 128
        buffer_capacity = 10_000_000
        min_buffer_size = 10_000  # Min samples in replay buffer before training starts
        max_path_length_train = 1_000
        log_freq = 1

        # -- Evaluation
        eval_freq = 100
        eval_size = 1_000
        eval_video_freq = -1
        eval_video_length = 100  # Max length for eval video
        eval_video_size = [100, 100]
        eval_video_fps = 10
        max_path_length_eval = 1_000

        LR = 3e-4
        WIDTH = 300

        # -- Policy
        policy_hidden_num = 2
        policy_hidden_size = WIDTH
        policy_hidden_act = 'relu'
        policy_optimizer = 'adam'
        policy_lr = LR
        policy_grad_norm_clip = None #10.0

        # -- Discriminator
        clf_hidden_num = 2
        clf_hidden_size = WIDTH
        clf_hidden_act = 'relu'
        clf_optimizer = 'adam'
        clf_lr = LR
        clf_grad_norm_clip = None #20.0

        # -- Critic
        critic_hidden_num = 2
        critic_hidden_size = WIDTH
        critic_hidden_act = 'relu'
        target_update_tau = 0.01  # Strength of target network polyak averaging
        target_update_freq = 1  # How often to update the target networks
        critic_optimizer = 'adam'
        critic_lr = LR
        critic_grad_norm_clip = None #20.0

        # -- Temperature
        alpha_initial = 0.1
        target_entropy = None  # Will be inferred from action space if None
        alpha_optimizer = 'adam'
        alpha_lr = LR
        train_alpha = False # Disable training of alpha if this is set



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
            self.buffer << self.train_sampler.sample_steps(
                n=missing_data, random=False
            )

        self.logger.info('Begin training')
        for epoch in range(1, self.cfg.num_epochs + 1):

            # Create a logger for dumping diagnostics
            epoch_logs = self.logger.epoch_logs(epoch)

            # Sample more steps for this epoch and add to replay buffer
            self.buffer << self.train_sampler.sample_steps(
                n=self.cfg.num_samples_per_epoch,
            )
            epoch_logs.add_scalar_dict(self.get_data_info(), prefix='Data')

            # Train with data from replay buffer
            for step in range(self.cfg.num_train_steps_per_epoch):

                # Sample batch (and convert all to torch tensors)
                obs, skills, acs, _, next_obs, dones = self.buffer.sample(
                    self.cfg.batch_size,
                    tensor=True,
                    device=self.device,
                    as_dict=False,
                )

                # Train discriminator
                disc_info = self.update_disc(obs=obs, skills=skills)
                epoch_logs.add_scalar_dict(disc_info, prefix='Disc')

                # Train q functions
                critic_info = self.update_critics(
                    obs=obs, skills=skills, acs=acs, next_obs=next_obs, dones=dones
                )
                epoch_logs.add_scalar_dict(critic_info, prefix='Critic')

                # Train actor and tune temperature
                actor_info = self.update_actor(obs=obs, skills=skills)
                epoch_logs.add_scalar_dict(actor_info, prefix='Actor')

                # Apply polyak averaging to target networks
                self.update_targets()

            # Eval, on occasions
            if epoch % self.cfg.eval_freq == 0:
                self.logger.info('Evaluating %s skills', self.cfg.num_skills)
                eval_info, eval_frames = self.evaluate(epoch, greedy=False)
                epoch_logs.add_scalar_dict(
                    eval_info, prefix='Eval', agg=['min', 'max', 'mean']
                )
                if eval_frames:
                    epoch_logs.add_videos(
                        'EvalRollout', eval_frames, fps=self.cfg.eval_video_fps
                    )

            # Write logs
            if (
                epoch % self.cfg.log_freq == 0 or
                epoch % self.cfg.eval_freq == 0
            ):
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

    def update_disc(self, obs, skills):
        """
        Updates parameters of the discriminator (classifier)

        Args:
            obs (Tensor<N,O>): A batch of obervations, each O floats
            skills (Tensor<N>): A batch of skills, each a single integer

        Returns:
            A dictionary with diagnostics
        """
        # Make a prediction and minimize cross-entropy-loss
        logits = self.clf(obs)
        clf_loss = F.cross_entropy(logits, skills)

        # Minimize it
        clf_grad_norm = utils.optimize(
            loss=clf_loss,
            optimizer=self.clf_optimizer,
            norm_clip=self.cfg.clf_grad_norm_clip,
        )

        # Diagnostics
        info = {}
        info['ClfAccuracy'] = (logits.argmax(dim=-1) == skills).float().mean()
        info['ClfLoss'] = clf_loss.detach()
        info['ClfGradNorm'] = clf_grad_norm

        return info

    def update_critics(self, obs, skills, acs, next_obs, dones):
        """
        Updates parameters of the critics (Q-functions)

        Args:
            obs (Tensor<N,O>): A batch of obervations, each O floats
            skills (Tensor<N>): A batch of skills, each a single integer
            acs (Tensor<N,A>): A batch of actions, each a float
            next_obs (Tensor<N,O>): A batch of next observations (see obs)
            dones (Tensor<N>): A batch of done flags, each a single float

        Returns:
            A dictinoary with diagnostics
        """
        # Generate standard cross-entropy loss for the discriminator
        clf_xe = F.cross_entropy(self.clf(next_obs), skills, reduction='none')
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

        # Use mean squared error as loss
        qf1_loss = F.mse_loss(q1_pred, q_target)
        qf2_loss = F.mse_loss(q2_pred, q_target)

        # And minimize it
        qf1_grad_norm = utils.optimize(
            loss=qf1_loss,
            optimizer=self.qf1_optimizer,
            norm_clip=self.cfg.critic_grad_norm_clip,
        )
        qf2_grad_norm = utils.optimize(
            loss=qf2_loss,
            optimizer=self.qf2_optimizer,
            norm_clip=self.cfg.critic_grad_norm_clip,
        )

        # Diagonstics
        info = {} # For later
        info['DIAYNReward'] = rews.mean()
        info['QTarget'] = q_target.mean().detach()
        info['QAbsDiff'] = (q1_pred - q2_pred).abs().mean().detach()
        info['Qf1Loss'] = qf1_loss.detach()
        info['Qf2Loss'] = qf2_loss.detach()
        info['Qf1GradNorm'] = qf1_grad_norm
        info['Qf2GradNorm'] = qf2_grad_norm

        return info

    def update_actor(self, obs, skills):
        """
        Updates parameters for the policy (and possibly alpha)

        Args:
            obs (Tensor<N,O>): A batch of observations, each O floats
            skills (Tensor<N>): A batch of skills, each a single integer

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

        # Minimize it
        policy_grad_norm = utils.optimize(
            loss=policy_loss,
            optimizer=self.policy_optimizer,
            norm_clip=self.cfg.policy_grad_norm_clip,
        )

        # Generate loss for the alpha value
        target_entropy_plus_logp = log_prob.detach() + self.target_entropy
        alpha_loss = (-self.log_alpha * target_entropy_plus_logp).mean()
        # But only minimize it if configured
        if self.cfg.train_alpha:
            utils.optimize(alpha_loss, self.alpha_optimizer)

        # Diagnostics
        info = {}
        info['Entropy'] = -log_prob.mean().detach()
        info['PolicyLoss'] = policy_loss.detach()
        info['PolicyGradNorm'] = policy_grad_norm
        info['AlphaLoss'] = alpha_loss.detach()
        info['AlphaValue'] = self.alpha

        return info

    def update_targets(self):
        """
        Applies polyak averaging to the target network of both q-functions
        """
        utils.polyak(
            net=self.qf1, target=self.qf1_target,
            tau=self.cfg.target_update_tau
        )
        utils.polyak(
            net=self.qf2, target=self.qf2_target,
            tau=self.cfg.target_update_tau
        )

    def evaluate(self, epoch=None, render=False, greedy=False):
        """
        Evaluate the current policy across all skills

        Args:
            epoch (int): Used to determine whether to do rendering if set
            render (bool): Manually specifies whether to render
            greedy (bool): Whether to do mean action (false=sample action)

        Returns:
            A tuple containing
            - A dictionary of eval results {Return: [...], TrajLen: [...]}
            - A list of video frames for each skill (None if not rendering)
        """
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

        # Rollout all skills
        info = []
        frames = []
        for skill in range(self.cfg.num_skills):
            with self.policy.configure(greedy=greedy, skill_dist=skill):
                info_, frames_ = self.eval_sampler.evaluate(
                    n=self.cfg.eval_size,
                    render=render,
                    render_max=self.cfg.eval_video_length,
                    render_size=tuple(self.cfg.eval_video_size),
                )
            info.append(info_)
            frames.append(frames_)

        # Combine: list of dicts of lists -> dict of lists
        info = {
            k: functools.reduce(operator.add, (i[k] for i in info))
            for k in info[0]
        }
        frames = frames if render else None

        return info, frames



if __name__ == '__main__':
    DIAYN.run_cli()
