
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import diayn.sac as sac
import diayn.policies as policies
import diayn.networks as networks
import diayn.distributions as distributions
import diayn.utils as utils
import diayn.critics as critics
import diayn.buffers as buffers
import diayn.samplers as samplers
import diayn.loggers as loggers
import diayn.configs as configs



class SACConfig(configs.Config):

    # -- Logistics
    savedir = None
    seed = np.random.randint(1<<32)
    gpu = True
    gpu_id = None

    # -- General
    env = 'HalfCheetah-v2'
    gamma = 0.99  # Discount factor
    num_epochs = 1_000
    num_samples_per_epoch = 1_000
    num_train_steps_per_epoch = 1_000
    eval_freq = 1
    eval_size = 4_000
    batch_size = 1024
    buffer_capacity = 1_000_000
    min_buffer_size = 10_000  # Min samples in replay buffer before training starts
    max_path_length_train = 1_000
    max_path_length_eval = 1_000

    # -- Policy
    policy_hidden_num = 2
    policy_hidden_size = 256
    policy_hidden_act = 'relu'
    policy_optimizer = 'adam'
    policy_lr = 3e-4

    # -- Critic
    critic_hidden_num = 2
    critic_hidden_size = 256
    critic_hidden_act = 'relu'
    target_update_tau = 5e-3  # Strength of target network polyak averaging
    target_update_freq = 1  # How often to update the target networks
    critic_optimizer = 'adam'
    critic_lr = 3e-4

    # -- Temperature
    alpha_initial = 1.0
    target_entropy = None  # Will be inferred from action space if None
    alpha_optimizer = 'adam'
    alpha_lr = 3e-4


    def build(self):
        """
        Constructs a gym environment and  from this config.
        Useful for restoring trained models after `config.from_yaml`
        """
        env = utils.make_env(self.env)
        ob_dim = env.observation_space.shape
        ac_dim = env.action_space.shape
        agent = sac.SAC(
            policy = policies.TanhGaussianMLPPolicy(
                ob_dim, ac_dim,
                hidden_num=self.policy_hidden_num,
                hidden_size=self.policy_hidden_size
            ),
            qf1=critics.QAMLPCritic(
                ob_dim, ac_dim,
                hidden_num=self.critic_hidden_num,
                hidden_size=self.critic_hidden_size,
            ),
            qf2=critics.QAMLPCritic(
                ob_dim, ac_dim,
                hidden_num=self.critic_hidden_num,
                hidden_size=self.critic_hidden_size,
            ),
            log_alpha=np.log(self.alpha_initial),
        )
        return env, agent


def main(cfg):

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #  LOGISTICS
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    # Setup experiment dir and save the config if configured
    if cfg.savedir:
        cfg.savedir = utils.setup_experiment(cfg.savedir, on_conflict='enumerate')
        cfg.save_yaml(os.path.join(cfg.savedir, 'config.yml'))
    # Setup logger (with tensorboard if savedir is not None)
    logger = loggers.Logger(logdir=cfg.savedir)
    logger.info('Using Config: \n%s', cfg)
    # Configure GPU and get the device that will be used
    device = utils.setup_gpu(cfg.gpu, cfg.gpu_id)
    logger.info('Using device: %s', device)
    # Set random seeds for numpy as torch
    utils.seed_everything(cfg.seed)
    logger.info('Using seed: %s', cfg.seed)


    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #  SETUP
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    # Create envs and agent
    train_env, agent = cfg.build()
    eval_env = utils.make_env(cfg.env)
    ob_dim = train_env.observation_space.shape
    ac_dim = train_env.action_space.shape
    agent.to(device)
    logger.info('Created agent: \n%s', agent)

    # Setup optimizers
    policy_optimizer = utils.get_optimizer(
        cfg.policy_optimizer, agent.policy.parameters(), lr=cfg.policy_lr
    )
    qf1_optimizer = utils.get_optimizer(
        cfg.critic_optimizer, agent.qf1.parameters(), lr=cfg.critic_lr
    )
    qf2_optimizer = utils.get_optimizer(
        cfg.critic_optimizer, agent.qf2.parameters(), lr=cfg.critic_lr
    )
    alpha_optimizer = utils.get_optimizer(
        cfg.alpha_optimizer, [agent.log_alpha], lr=cfg.alpha_lr
    )

    # Setup target entropy (which alpha will be tuned against)
    if cfg.target_entropy is None:
        target_entropy = -np.prod(ac_dim)
        logger.info('Using dynamic target entropy: %s', target_entropy)
    else:
        target_entropy = cfg.target_entropy
        logger.info('Using static target entropy: %s', target_entropy)

    # Setup replay buffer
    buffer = buffers.RingBuffer(
        capacity=int(cfg.buffer_capacity),
        keys=['obs', 'acs',  'rews', 'next_obs', 'dones'],
        dims=[ob_dim, ac_dim, None,   ob_dim,     None]
    )

    # Setup samplers
    train_sampler = samplers.Sampler(
        train_env, agent.policy, max_steps=cfg.max_path_length_train
    )
    eval_sampler = samplers.Sampler(
        eval_env, agent.policy, max_steps=cfg.max_path_length_eval
    )


    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #  TRAINING LOOP
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    logger.info(f'Seeding buffer with {cfg.min_buffer_size} samples')
    buffer << train_sampler.sample_steps(cfg.min_buffer_size)

    logger.info('Begin training')
    for epoch in range(cfg.num_epochs):

        epoch_logs = logger.epoch_logs(epoch)

        # Sample more steps for this epoch and add to replay buffer
        buffer << train_sampler.sample_steps(cfg.num_samples_per_epoch)

        # Write statistics from training data sampling
        with epoch_logs.prefix('Data'):
            epoch_logs.add_scalar('TotalEnvSteps', train_sampler.total_steps)
            epoch_logs.add_scalar('BufferSize', len(buffer))
            avg_return_25 = train_sampler.returns[-25:]
            epoch_logs.add_scalar('Last25TrainRets', avg_return_25, agg='mean')

        # Train with data from replay buffer
        for step in range(cfg.num_train_steps_per_epoch):

            # Sample batch (and convert all to torch tensors at correct device)
            batch = buffer.sample(cfg.batch_size, tensor=True, device=device)

            # Train critics
            qf1_loss, qf2_loss, critic_info = agent.critic_objective(
                obs=batch['obs'], acs=batch['acs'], rews=batch['rews'],
                next_obs=batch['next_obs'], dones=batch['dones'],
                gamma=cfg.gamma
            )
            qf1_grad_norm = utils.optimize(qf1_loss, qf1_optimizer)
            qf2_grad_norm = utils.optimize(qf2_loss, qf2_optimizer)
            with epoch_logs.prefix('Critic'):
                epoch_logs.add_scalar_dict(critic_info, agg='mean')
                epoch_logs.add_scalar('Qf1GradNorm', qf1_grad_norm, agg='mean')
                epoch_logs.add_scalar('Qf2GradNorm', qf2_grad_norm, agg='mean')

            # Train actor and tune temerature
            policy_loss, alpha_loss, policy_info = agent.actor_objective(
                obs=batch['obs'], target_entropy=target_entropy
            )
            policy_grad_norm = utils.optimize(policy_loss, policy_optimizer)
            utils.optimize(alpha_loss, alpha_optimizer)
            with epoch_logs.prefix('Actor'):
                epoch_logs.add_scalar_dict(policy_info, agg='mean')
                epoch_logs.add_scalar('PolicyGradNorm', policy_grad_norm, agg='mean')

            # Apply polyak averaging to target networks
            agent.update_critic_targets(cfg.target_update_tau)

        # Eval, on occasions
        if epoch % cfg.eval_freq == 0:
            eval_rets, eval_lens = eval_sampler.evaluate(n=cfg.eval_size, greedy=True)
            with epoch_logs.prefix('Eval'):
                epoch_logs.add_scalar('Return', eval_rets, agg=['min', 'max', 'mean'])
                epoch_logs.add_scalar('TrajLen', eval_lens, agg=['min', 'max', 'mean'])

        # Write logs
        epoch_logs.dump(step=train_sampler.total_steps)


    # Save final agents if configured
    if cfg.savedir:
        pass # TODO: Implement simple parameter saving


    return agent, buffer





if __name__ == '__main__':
    main(SACConfig.from_cli())
