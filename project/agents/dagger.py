import pickle

import torch

import project.agents as agents
import project.agents.sac as sac
import project.buffers as buffers
import project.policies as policies
import project.samplers as samplers
import project.utils as utils


class Dagger(agents.Agent):
    class Config(agents.Agent.Config):
        env = 'HalfCheetah-v2'
        expert_data = None  # << must be set
        expert_data_size = None  # Amount of expert data to use
        expert_policy = None  # << must be set
        num_epochs = 100
        num_train_steps_per_epoch = 100
        num_samples_to_collect_per_epoch = 1000  # num trajectories to collect per epoch

        # -- Evaluation
        eval_size = 10_000
        eval_video_freq = -1
        eval_video_length = 200  # Max length for eval video
        eval_video_size = [200, 200]
        eval_video_fps = 10
        max_path_length_eval = 1_000

        # -- Policy
        policy_hidden_num = 2
        policy_hidden_size = 256
        policy_hidden_act = 'relu'
        policy_optimizer = 'adam'
        policy_lr = 1e-3
        policy_grad_norm_clip = None  # 10.0

    def init(self):
        # Initialize environment to get input/output dimensions
        self.train_env = utils.make_env(self.cfg.env)
        self.env = utils.make_env(self.cfg.env)
        ob_dim, = self.ob_dim, = self.env.observation_space.shape
        ac_dim, = self.ac_dim, = self.env.action_space.shape

        # Setup policies
        self.expert_policy = sac.SAC.restore(self.cfg.expert_policy)  # Expert
        self.policy = policies.GaussianMLPPolicy(  # Cloned
            ob_dim, ac_dim,
            hidden_num=self.cfg.policy_hidden_num,
            hidden_size=self.cfg.policy_hidden_size,
            hidden_act=self.cfg.policy_hidden_act,
            static_std=True,
        )

        # Send policies to device
        self.to(self.device)

        # Setup optimizer
        self.policy_optimizer = utils.get_optimizer(
            name=self.cfg.policy_optimizer,
            params=self.policy.parameters(),
            lr=self.cfg.policy_lr,
        )

        # Setup samplers
        self.train_sampler = samplers.Sampler(
            env=self.train_env,
            policy=self.policy,
            max_steps=self.cfg.max_path_length_train
        )

        # Create sampler for evaluations
        self.eval_sampler = samplers.Sampler(
            env=self.env,
            policy=self.policy,
            max_steps=self.cfg.max_path_length_eval
        )

        # Setup replay buffer
        self.buffer = buffers.RingBuffer(
            capacity=int(self.cfg.buffer_capacity),
            keys=['ob', 'ac'],
            dims=[ob_dim, ac_dim, None, ob_dim, None]
        )
        self.logger.info('Initialized buffer (%s)', self.buffer)

    def train(self):
        # Initialize buffer with expert data
        self.logger.info('Procuring expert data')
        self.load_expert_data()

        self.logger.info('Begin training')

        for epoch in range(1, self.cfg.num_epochs):

            # Create a logger for dumping diagnostics
            epoch_logs = self.logger.epoch_logs(epoch)

            for steps in range(self.cfg.num_train_steps_per_epoch):
                # Sample expert data from buffer
                obs, acs = self.buffer.sample(
                    self.cfg.batch_size,
                    tensor=True,
                    device=self.device,
                    as_dict=False)

                # Train policy
                actor_info = self.update_actor(obs, acs)
                epoch_logs.add_scalar_dict(actor_info, prefix="Actor")

            # Sample new trajectories
            data = self.train_sampler.sample_steps(self.cfg.num_samples_to_collect_per_epoch)

            # Relabel with expert
            data['ac'] = self.relabel_with_expert(data["ob"])

            # Aggregate new data
            self.buffer << data

            # Evaluate at the end of every epoch
            eval_info, eval_frames = self.evaluate(greedy=True)
            epoch_logs.add_scalar_dict(
                eval_info, prefix='Eval', agg=['min', 'max', 'mean']
            )
            if not eval_frames is None:
                epoch_logs.add_video(
                    'EvalRollout', eval_frames, fps=self.cfg.eval_video_fps,
                )

            # Write logs
            epoch_logs.dump()

    def load_expert_data(self):
        """
        Initializes a buffer with expert data from config
        """
        # Get expert data
        if self.cfg.expert_data is None:
            raise ValueError('No expert data provided in config')
        elif isinstance(self.cfg.expert_data, str):
            self.logger.info('Loading expert data from: %s', self.cfg.expert_data)
            with open(self.cfg.expert_data, 'rb') as f:
                expert_data = pickle.load(f)
        else:
            self.logger.info('Using configured expert data as-is')
            expert_data = self.cfg.expert_data
        # Assert that it is a list of dicts
        if (
                not isinstance(expert_data, (list, tuple)) or
                not len(expert_data) or
                not all(isinstance(path, dict) for path in expert_data) or
                any({'ob', 'ac'} - set(path) for path in expert_data)
        ):
            raise ValueError(
                'Expert data must be a non-empty list of dicts with'
                '"ob" and "ac" keys'
            )
        # Limit amount of expert data if configured
        if self.cfg.expert_data_size:
            self.logger.info(
                'Filtering expert data to %s entries', self.cfg.expert_data_size
            )
            expert_data_filtered = []
            n = 0
            for path in expert_data:
                j = min(len(path['ob']), self.cfg.expert_data_size - n)
                path = {k: v[:j] for k, v in path.items()}
                expert_data_filtered.append(path)
                n += j
                if n >= self.cfg.expert_data_size:
                    break
            else:
                raise ValueError('Not enough expert data!')
            expert_data = expert_data_filtered

        # Fill buffer
        for path in expert_data:
            self.buffer << path
        self.logger.info('Filled buffer (%s)', self.buffer)

    def update_actor(self, obs, acs):
        """
        Updates parameters for the policy with behavior cloning

        Args:
            obs (Tensor<N,O>): A batch of observations, each O floats
            acs (Tensor<N,A>): A batch of actions, each A float

        Returns:
            A dictionary with diagnostics
        """
        # Generate loss equal to log probability of expert actions
        pi = self.policy(obs)
        logp = pi.log_prob(acs)
        loss = -logp.mean()
        # Minimize it
        grad_norm = utils.optimize(
            loss=loss,
            optimizer=self.policy_optimizer,
            norm_clip=self.cfg.policy_grad_norm_clip,
        )
        # Generate diagnostics
        info = {}
        info['PolicyLoss'] = loss.detach()
        info['PolicyGradNorm'] = grad_norm

        return info

    def evaluate(self, epoch=None, render=None, greedy=True):
        """
        Evaluates the current policy
        """
        # Determine whether to render
        if not render is None:
            render = bool(render)
        elif epoch is None:
            render = False
        elif self.cfg.eval_video_freq <= 0:
            render = False
        else:
            render = epoch % self.cfg.eval_video_freq == 0

        # Rollout
        with self.policy.configure(greedy=greedy):
            info, frames = self.eval_sampler.evaluate(
                n=self.cfg.eval_size,
                render=render,
                render_max=self.cfg.eval_video_length,
                render_size=tuple(self.cfg.eval_video_size),
            )

        return info, frames

    def relabel_with_expert(self, obs):
        obs = torch.as_tensor(obs, device=self.device)
        with torch.no_grad():
            pi = self.expert_policy(obs)
            acs = pi.sample(greedy=True)
        return acs.cpu().numpy()
