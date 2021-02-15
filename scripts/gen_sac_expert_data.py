
import argparse
import os
import pickle

import numpy as np
import tqdm

import project.agents.sac as sac
import project.utils as utils


def main(agent_dir, path_length, num_paths, greedy=False, save_path=None):

    print(f'Restoring agent from: {agent_dir}')
    agent = sac.SAC.restore(agent_dir)

    print(f'Creating env: {agent.cfg.env}')
    env = utils.make_env(agent.cfg.env)

    print(f'Generating {num_paths} of length {path_length}')
    paths = []
    for _ in tqdm.tqdm(range(num_paths)):
        path = []
        ob = env.reset()
        for _ in range(path_length):
            with agent.policy.configure(greedy=greedy):
                ac, _ = agent.policy.get_action(ob)
            next_ob, rew, done, _ = env.step(ac)
            path.append(dict(ob=ob, ac=ac, rew=rew, next_ob=next_ob, done=done))
            ob = next_ob
            if done:
                break
        paths.append(
            {
                k: np.array([p[k] for p in path], dtype=np.float32)
                for k in path[0]
            }
        )

    print('Returns:', [path['rew'].sum() for path in paths])

    if save_path is not None:
        print(f'Print saving to specified path: {save_path}')
    else:
        save_path = os.path.join(agent_dir, 'expert_data.pkl')
        print(f'Saving to agent dir: {save_path}')

    with open(save_path, 'wb') as f:
        pickle.dump(paths, f)

    print('Done')




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('agent_dir')
    parser.add_argument('--path-length', '-l', type=int, default=1000)
    parser.add_argument('--num-paths', '-n', type=int, default=10)
    parser.add_argument('--greedy', action='store_true')
    parser.add_argument('--save-path')

    args = parser.parse_args()
    main(**vars(args))
