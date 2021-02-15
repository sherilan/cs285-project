

import numpy as np
import tqdm
import mujoco_py
import cv2



class Sampler:

    def __init__(self, env, policy, max_steps=None):
        self.env = env
        self.policy = policy
        self.max_steps = max_steps
        self.returns = []
        self.total_steps = 0
        self.new_episode()

    def new_episode(self):
        self.policy.reset()
        self.last_ob = None
        self.current_return = 0.0
        self.current_step = 0

    @property
    def is_terminal(self):
        return self.last_ob is None

    def step(self, random=False):

        # Clear if done
        if self.last_ob is None:
            ob = self.env.reset()
        else:
            ob = self.last_ob

        # Get action and advance env
        if random:
            ac, policy_info = self.env.action_space.sample(), {}
        else:
            ac, policy_info = self.policy.get_action(ob)
        next_ob, rew, done, _ = self.env.step(ac)

        # Bookkeeping
        self.last_ob = next_ob
        self.current_return += rew
        self.current_step += 1
        self.total_steps += 1

        # Stop if done or max steps exceeded
        if done or self.max_steps and self.current_step >= self.max_steps:
            self.returns.append(self.current_return)
            self.new_episode()

        return dict(
            ob=ob, ac=ac, rew=rew, next_ob=next_ob, done=done, **policy_info
        )

    def sample_steps(self, n=None, random=False):
        data = []
        for _ in range(int(n or 1)):
            data.append(self.step(random=random))
        # If single step, convert list of dicts -> dict
        if not n:
            data = data[0]
        # Otherwise, convert list of dicts -> dicts of np.array
        else:
            data = {k: np.array([step[k] for step in data]) for k in data[0]}

        return data

    def sample_paths(self, n=None, random=False):
        paths = []
        for _ in range(int(n or 1)):
            # Reset and sample until terminal
            self.new_episode()
            data = []
            while True:
                data.append(self.step(random=random))
                if self.is_terminal:
                    break
            # Convert list of dicts -> dicts of np.array
            path = {k: np.array([step[k] for step in data]) for k in data[0]}
            paths.append(path)
        return paths[0] if not n else path

    def evaluate(
        self, n, random=False, render=False, render_max=None, render_size=None
    ):
        rews = []
        rets = []
        lens = []
        frames = []
        self.new_episode()
        for i in range(n):
            step = self.step(random=random)
            rews.append(step['rew'])
            if render and (render_max is None or i < render_max):
                frame = self.env.render(mode='rgb_array')
                if not render_size is None:
                    frame = cv2.resize(frame, render_size)
                frames.append(frame)
            if self.is_terminal:
                rets.append(sum(rews))
                lens.append(len(rews))
                rews = []
        info = {}
        info['Return'] = rets
        info['TrajLen'] = lens
        frames = np.array(frames) if render else None
        return info, frames


class SkillSampler(Sampler):

    def __init__(
        self, env, policy, num_skills, skill_dist=None, max_steps=None
    ):
        self.num_skills = num_skills
        self.skill_dist = skill_dist
        super().__init__(env, policy, max_steps=max_steps)

    def new_episode(self):
        super().new_episode()
        # If distribution is give, sample it
        if self.skill_dist is not None:
            self.skill = self.skill_dist()
        # Otherwise, sample uniformly
        else:
            self.skill = np.random.randint(self.num_skills)

    def step(self, greedy=False, skill=None):
        # Clear if done
        if self.last_ob is None:
            ob = self.env.reset()
        else:
            ob = self.last_ob

        # Get action conditioned on current skill and advance env
        skill = skill or self.skill
        skill_1h = np.zeros(self.num_skills, dtype=np.float32)
        skill_1h[self.skill if skill is None else skill] = 1.0
        ac = self.policy.get_action(ob, skill_1h, greedy=greedy)
        next_ob, rew, done, info = self.env.step(ac)

        # Bookkeeping
        self.last_ob = next_ob
        self.current_return += rew
        self.current_step += 1
        self.total_steps += 1

        # Stop if done or max steps exceeded
        if done or self.max_steps and self.current_step >= self.max_steps:
            self.returns.append(self.current_return)
            self.new_episode()

        return dict(
            ob=ob, ac=ac, rew=rew, next_ob=next_ob, done=done, skill=skill
        )
