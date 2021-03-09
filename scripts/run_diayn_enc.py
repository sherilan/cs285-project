

import argparse
import os

import project.agents.diayn_enc as diayn_enc



variants = dict(
    point=dict(
        env='PointMaze-v3',
        policy_hidden_size=32,
        clf_hidden_size=32,
        critic_hidden_size=32,
        min_buffer_size=100,
        num_samples_per_epoch=100,
        num_train_steps_per_epoch=100,
        max_path_length_train=100,
        max_path_length_eval=100,
        eval_size=100,
        num_skills=10,


    ),
    cheetah=dict(
        env='HalfCheetah-v2',
    ),
    hopper=dict(
        env='Hopper-v2',
    ),
    ant=dict(
        env='Ant-v2'
    )
)


parser = argparse.ArgumentParser()
parser.add_argument('variant', choices=list(variants))
parser.add_argument('--path')
parser.add_argument('--encode', type=int)
parser.add_argument('--gpu-id', default=0, type=int)
parser.add_argument('--data')

args = parser.parse_args()

if args.path is None:
    path = None
else:
    path = os.path.join(args.path, args.variant)
    if args.encode:
        path += f'_enc{args.encode}'

if args.data is None:
    data = None
else:
    if os.path.isdir(args.data):
        data = os.path.join(args.data, 'expert_data.pkl')
    else:
        data = args.data

if args.encode:
    enc_enable = True
    enc_dim = args.encode
else:
    enc_enable = False
    enc_dim = 0

variant_cfg = variants[args.variant]
cfg = diayn_enc.DIAYN.Config(
    path=path,
    clf_enc_enable=enc_enable,
    clf_enc_dim=enc_dim,
    gpu_id=args.gpu_id,
    expert_data=data,
    **variant_cfg
)

diayn_enc.DIAYN.run(cfg)
