
import os
import argparse

import cv2
import tqdm
import numpy as np
import moviepy.video.io.ImageSequenceClip

import project.agents.diayn_enc as diayn_enc
import project.utils as utils

parser = argparse.ArgumentParser()
parser.add_argument('agent_dirs', nargs='+')
parser.add_argument('--name')
parser.add_argument('--image-size', type=int, default=200)
parser.add_argument('--traj-length', type=int, default=200)
parser.add_argument('--greedy', action='store_true')
parser.add_argument('--width', type=int, default=10)
parser.add_argument('--fps', type=int, default=16)
parser.add_argument('--out', default='movie.mp4')

args = parser.parse_args()

for agent_dir in args.agent_dirs:

    agent = diayn_enc.DIAYN.restore(agent_dir, name=args.name)
    env = utils.make_env(agent.cfg.env)


    print('Rolling out skills')
    all_frames = []
    for skill in tqdm.tqdm(range(agent.cfg.num_skills)):
        with agent.policy.configure(skill=skill, skill_dist=skill, greedy=args.greedy):
            ob = env.reset()
            frames = []
            all_frames.append(frames)
            for t in range(args.traj_length):
                ac = agent.policy.get_action(ob)[0]
                ob, *_ = env.step(ac)
                frame = env.render(mode='rgb_array')
                frame = cv2.resize(frame, (args.image_size, args.image_size))
                frames.append(frame)

    print('Generating mosaic')
    mosaics = []
    for t in tqdm.tqdm(range(args.traj_length)):
        tiles = [frames[t] for frames in all_frames]

        n_pad = -len(tiles) % args.width

        tiles += [np.zeros_like(tiles[0])] * n_pad

        rows = []
        for i in range(0, len(tiles), args.width):
            rows.append(np.concatenate(tiles[i:i+args.width], axis=1))

        mosaic = np.concatenate(rows, axis=0)
        mosaics.append(mosaic)

    print('Making movie')
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(mosaics, fps=args.fps)
    clip.write_videofile(os.path.join(agent_dir, args.out))
