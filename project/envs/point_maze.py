

import gym
import numpy as np


class PointMaze(gym.Env):

    def parse_maze(maze_string, levels):
        rows = [row.strip() for row in maze_string.split('\n') if row.strip()]
        assert all(cell in levels for row in rows for cell in row)
        return np.array([
            [levels[cell] for cell in row] for row in rows
        ]).astype(int)

    MAZE_A = parse_maze(
        '''
        ⬛⬛⬛⬜⬛⬛⬛
        ⬛⬛⬛⬜⬛⬛⬛
        ⬛⬛⬛⬜⬛⬛⬛
        ⬜⬜⬜⬜⬜⬜⬜
        ⬛⬛⬛⬜⬛⬛⬛
        ⬛⬛⬛⬜⬛⬛⬛
        ⬛⬛⬛⬜⬛⬛⬛
        ''',
        levels = {'⬜': 0, '⬛': 1}
    )

    MAZE_B = parse_maze(
        '''
        ⬛⬛⬛⬛⬜⬜⬜⬜
        ⬛⬛⬛⬛⬜⬜⬜⬜
        ⬛⬛⬛⬛⬜⬜⬜⬜
        ⬜⬜⬜⬜⬜⬜⬜⬜
        ⬛⬛⬛⬛⬜⬜⬜⬜
        ⬛⬛⬛⬛⬜⬜⬜⬜
        ⬛⬛⬛⬛⬜⬜⬜⬜
        ''',
        levels = {'⬜': 0, '⬛': 1}
    )

    MAZE_C = parse_maze(
        '''
        ⬛⬛⬛🟧⬛⬛⬛
        ⬛⬛⬛⬜⬛⬛⬛
        ⬛⬛⬛⬜⬛⬛⬛
        🟧⬜⬜⬜⬜⬜🟧
        ⬛⬛⬛⬜⬛⬛⬛
        ⬛⬛⬛⬜⬛⬛⬛
        ⬛⬛⬛🟧⬛⬛⬛
        ''',
        levels = {'🟧': 0, '⬜': 1, '⬛': 2}
    )

    MAZE_D = parse_maze(
        '''
        ⬛⬛⬛⬛⬛⬛⬛
        ⬛⬜⬜⬜⬜⬜⬛
        ⬛⬜⬜⬜⬜⬜⬛
        ⬛⬜⬜⬜⬜⬜⬛
        ⬛⬜⬜⬜⬜⬜⬛
        ⬛⬜⬜⬜⬜⬜⬛
        ⬛⬛⬛⬛⬛⬛⬛
        ''',
        levels = {'⬜': 0, '⬛': 1}
    )

    MAZE_E = parse_maze(
        '''
        ⬛⬛⬛🟧🟧🟧🟧
        ⬛⬛⬛⬜⬛⬛🟧
        ⬛⬛⬛⬜⬛⬛🟧
        🟧⬜⬜⬜⬜⬜🟧
        🟧⬛⬛⬜⬛⬛⬛
        🟧⬛⬛⬜⬛⬛⬛
        🟧🟧🟧🟧⬛⬛⬛
        ''',
        levels = {'🟧': 0, '⬜': 1, '⬛': 2}
    )


    def __init__(self, maze, origin, speed=1):
        if isinstance(maze, str):
            maze = getattr(self, maze)
        self.maze = maze
        self.origin = origin
        self.speed = speed
        self.norm = max(maze.shape)
        self.observation_space = gym.spaces.Box(-1, 1, shape=[2])
        self.action_space = gym.spaces.Box(-1, 1, shape=[2])


    def reset(self):
        self.xy = self.origin
        self.path = [self.xy]
        return self.get_obs()

    @property
    def width(self):
        return self.maze.shape[1] - 1e-6

    @property
    def height(self):
        return self.maze.shape[0] - 1e-6


    def step(self, action):
        dx, dy = np.array(action).clip(-1, 1) * self.speed
        x, y = self.xy
        m = self.get_maze_val(x, y)
        ddx = 1e-6 * np.sign(dx)
        ddy = 1e-6 * np.sign(dy)

        for t, x, y in self.get_crossings(x, y, dx, dy):
            # Stop if value in maze (elevation) is bigger at new positions
            if (
                x <= 0 or y <= 0 or
                x >= self.width or y >= self.height or
                self.get_maze_val(x + ddx, y + ddy) > m
            ):
                x = np.clip(x - ddx, 0, self.width)
                y = np.clip(y - ddy, 0, self.height)
                break
        else:
            x, y = self.xy
            x = np.clip(x + dx, 0, self.width)
            y = np.clip(y + dy, 0, self.height)

        self.xy = x, y
        self.path.append(self.xy)
        return self.get_obs(), 0, False, {}


    def render(self, mode='rgb_array'):
        assert mode in {'human', 'rgb_array'}

        import matplotlib.pyplot as plt
        import matplotlib.backends.backend_agg as mplagg

        fig = plt.Figure(figsize=self.maze.shape[::-1], frameon=False)
        fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        canvas = mplagg.FigureCanvas(fig)
        ax = fig.gca()
        cmin = self.maze.min()
        cmax = self.maze.max()
        ax.matshow(
            self.maze,
            cmap='PuBu_r',
            vmin=cmin - 0.5 * (cmax - cmin),
            vmax=cmax + 0.0 * (cmax - cmin),
        )
        ax.set_axis_off()

        for i, ((x0, y0), (x1, y1)) in enumerate(zip(self.path, self.path[1:])):
            ax.add_line(
                plt.Line2D(
                    (x0 - 0.5, x1 - 0.5),
                    (y0 - 0.5, y1 - 0.5),
                    color='r',
                    alpha=0.4 + 0.6 * i / len(self.path)
                )
            )
        if self.path:
            ax.add_patch(
                plt.Circle(
                    (self.path[-1][0]-0.5, self.path[-1][1]-0.5),
                    0.1,
                    color='r'
                )
            )

        fig.canvas.draw()

        if mode == 'human':
            plt.show()
        else:
            width, height = (fig.get_size_inches() * fig.get_dpi()).astype(int)
            img = np.frombuffer(
                fig.canvas.tostring_rgb(),
                dtype='uint8'
            )
            return img.reshape(height, width, 3)


    def get_obs(self):
        return np.array(self.xy) / self.norm


    def get_maze_val(self, x, y):
        x = np.clip(x, 0, self.maze.shape[0] - 1e-6)
        y = np.clip(y, 0, self.maze.shape[1] - 1e-6)
        return self.maze[int(y)][int(x)]

    @staticmethod
    def get_crossings(x, y, dx, dy):

        # Compute crossings of grid in x direction
        if dx > 0:
            x_crossings_x = np.arange(np.ceil(x), x + dx, 1)
        else:
            x_crossings_x = np.arange(np.floor(x), x + dx, -1)
        if dx != 0:
            x_crossing_times = (x_crossings_x - x) / dx
        else:
            x_crossing_times = np.array([])
        x_crossings_y = y + x_crossing_times * dy

        # Compute crossings of grid in y direction
        if dy > 0:
            y_crossings_y = np.arange(np.ceil(y), y + dy, 1)
        else:
            y_crossings_y = np.arange(np.floor(y), y + dy, -1)
        if dy != 0:
            y_crossing_times = (y_crossings_y - y) / dy
        else:
            y_crossing_times = np.array([])
        y_crossings_x = x + y_crossing_times * dx

        # Combine
        crossings = np.concatenate([
            np.stack([x_crossing_times, x_crossings_x, x_crossings_y], axis=-1),
            np.stack([y_crossing_times, y_crossings_x, y_crossings_y], axis=-1),
        ])
        return sorted(crossings, key=lambda txy: txy[0])
