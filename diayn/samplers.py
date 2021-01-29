

import numpy as np
import tqdm
import mujoco_py



class Sampler:

    def __init__(self, env, policy, max_steps=None):
        self.env = env
        self.policy = policy
        self.max_steps = max_steps
        self.returns = []
        self.total_steps = 0
        self.new_episode()

    def new_episode(self):
        self.last_ob = None
        self.current_return = 0.0
        self.current_step = 0

    def step(self, greedy=False):

        # Clear if done
        if self.last_ob is None:
            ob = self.env.reset()
        else:
            ob = self.last_ob

        # Get action and advance env
        ac = self.policy.get_action(ob, greedy=greedy)
        try:
            next_ob, rew, done, info = self.env.step(ac)
        except mujoco_py.MujocoException as e:
            print('Received mujoco exception', e)
            import pdb; pdb.set_trace()
            raise e


        # Bookkeeping
        self.last_ob = next_ob
        self.current_return += rew
        self.current_step += 1
        self.total_steps += 1

        # Artificial done
        done = done or self.max_steps and self.current_step >= self.max_steps

        # Stop if done or max steps exceeded
        if done:
            self.returns.append(self.current_return)
            self.new_episode()

        return ob, ac, rew, next_ob, done

    def sample_steps(self, n=None, greedy=False, pbar=False):
        obs, acs, rews, next_obs, dones = data = [], [], [], [], []

        for _ in tqdm.tqdm(range(int(n or 1)), disable=not pbar):
            ob, ac, rew, next_ob, done = self.step(greedy=greedy)
            obs.append(ob)
            acs.append(ac)
            rews.append(rew)
            next_obs.append(next_ob)
            dones.append(done)

        # Wrap in np array
        data = map(np.array, data)
        # Unpack if no number of steps were requested
        if not n:
            data = [d[0] for d in data]

        return dict(zip(['obs', 'acs', 'rews', 'next_obs', 'dones'], data))

    def sample_paths(self, n=None, max_steps=None, greedy=False):
        paths = []
        for _ in range(int(n or 1)):
            obs, acs, rews, next_obs, dones = data = [], [], [], [], []
            self.new_episode()
            while True:
                ob, ac, rew, next_ob, done = self.step(greedy=greedy)
                obs.append(ob)
                acs.append(ac)
                rews.append(rew)
                next_obs.append(next_ob)
                dones.append(done)
                if done:
                    break

            paths.append(dict(zip(
                ['obs', 'acs', 'rews', 'next_obs', 'dones'], map(np.array, data)
            )))

        return paths[0] if not n else paths

    def evaluate(self, n, greedy=False):
        rews = []
        rets = []
        lens = []
        self.new_episode()
        for i in range(n):
            _, _, rew, _, done = self.step(greedy=greedy)
            rews.append(rew)
            if done:
                rets.append(sum(rews))
                lens.append(len(rews))
                rews = []
        return rets, lens
