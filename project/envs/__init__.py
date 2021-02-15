
import gym


# -- Point Maze

gym.envs.register(
    id='PointMaze-v0',
    entry_point='project.envs.point_maze:PointMaze',
    max_episode_steps=100,
    kwargs=dict(
        maze='MAZE_A',
        origin=(3.5, 3.5),
        speed=1.0,
    )
)

gym.envs.register(
    id='PointMaze-v1',
    entry_point='project.envs.point_maze:PointMaze',
    max_episode_steps=100,
    kwargs=dict(
        maze='MAZE_B',
        origin=(0.5, 3.5),
        speed=1.0,
    )
)

gym.envs.register(
    id='PointMaze-v2',
    entry_point='project.envs.point_maze:PointMaze',
    max_episode_steps=100,
    kwargs=dict(
        maze='MAZE_C',
        origin=(3.5, 3.5),
        speed=1.0,
    )
)

gym.envs.register(
    id='PointMaze-v3',
    entry_point='project.envs.point_maze:PointMaze',
    max_episode_steps=100,
    kwargs=dict(
        maze='MAZE_D',
        origin=(3.5, 3.5),
        speed=1.0,
    )
)
