
import collections
import contextlib
import logging
import os

import numpy as np
import tensorboardX
import torch




class Logger(logging.Logger):

    def __init__(self, name=None, rank=None, logdir=None, level=None, fmt=None):
        name = name or 'AgentLogger'
        level = level or logging.INFO

        fmt = fmt or '%(asctime)s :: %(levelname)s :: %(message)s'
        if rank is not None:
            fmt = f'[{rank}] {fmt}'

        super().__init__(name, level)

        sh = logging.StreamHandler()
        sh.setLevel(logging.WARNING if rank else logging.DEBUG)
        sh.setFormatter(logging.Formatter(fmt))
        self.addHandler(sh)

        if logdir:
            if rank is None:
                logpath = os.path.join(logdir, 'log.txt')
            else:
                logpath = os.path.join(logdir, f'log_{rank}.txt')
            fh = logging.FileHandler(logpath)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(fmt))
            self.addHandler(fh)
            self.tb_writer = tensorboardX.SummaryWriter(logdir)
        else:
            self.tb_writer = None

    def epoch_logs(self, epoch):
        return EpochLogs(epoch, printf=self.info, tb_writer=self.tb_writer)



class EpochLogs:

    def __init__(self, epoch, printf=None, tb_writer=None):
        self.epoch = epoch
        self.printf = printf or print
        self.tb_writer = tb_writer
        self.scalars = collections.OrderedDict()
        self.videos = collections.OrderedDict()
        self.prefixes = []

    def push_prefix(self, *prefixes):
        for prefix in prefixes:
            self.prefixes.append(prefix)

    def pop_prefix(self, n=1):
        for _ in range(n):
            self.prefixes.pop()

    @contextlib.contextmanager
    def prefix(self, *prefixes):
        try:
            self.push_prefix(*prefixes)
            yield
        finally:
            self.pop_prefix(len(prefixes))

    def add_video(self, name, video, **video_kwargs):
        if self.prefixes:
            name = f'{"/".join(self.prefixes)}/{name}'
        video = Video(name, video, **video_kwargs)
        if name in self.videos:
            self.videos[name].add(video)
        else:
            self.videos[name] = video

    def add_scalar(self, name, value, **scalar_kwargs):
        if self.prefixes:
            name = f'{"/".join(self.prefixes)}/{name}'
        scalar = EpochScalar(name, value, **scalar_kwargs)
        if not name in self.scalars:
            self.scalars[name] = scalar
        else:
            self.scalars[name].add(scalar)

    def add_scalar_dict(self, dict, prefix=None, **scalar_kwargs):
        try:
            if prefix:
                self.push_prefix(prefix)
            for k, v in dict.items():
                self.add_scalar(k, v, **scalar_kwargs)
        finally:
            if prefix:
                self.pop_prefix(1)


    def dump(self, step=None, debug=False, ffmt='%.4f'):
        # Extract scalars
        names = []
        values = []
        for name, scalar in self.scalars.items():
            if not debug and scalar.debug:
                continue
            for agg, value in scalar:
                names.append(str(name if agg is None else f'{name}:{agg}'))
                values.append(value)
        # Cast to stirng
        value_strings = []
        for value in values:
            if isinstance(value, float):
                value_strings.append(ffmt % value)
            else:
                value_strings.append(str(value))
        # Compute dimensions
        max_length_name = max(len(n) for n in names)
        max_length_value = max(len(vs) for vs in value_strings)
        width = max_length_name + max_length_value + 3
        # Generate string
        string = [
            f'Epoch {self.epoch}',
            '-' * max_length_name + '   ' + '-' * max_length_value
        ]
        for name, value_string in zip(names, value_strings):
            space = width - len(name) - len(value_string)
            string.append(f'{name}{" " * space}{value_string}')
        string.append('=' * width)
        # Print to wherever the printf sends stuff to
        self.printf('\n'.join(string))
        # If available, write scalars to tensorboard too
        if self.tb_writer:
            # Write scalars
            tb_step = self.epoch if step is None else step
            for name, value in zip(names, values):
                self.tb_writer.add_scalar(name, value, tb_step)
            # Write videos
            for name, video in self.videos.items():
                self.tb_writer.add_video(
                    name, video.tb_frames, tb_step, fps=video.fps
                )
            self.tb_writer.flush()


    def __repr__(self):
        return f'<EpochLogs {self.epoch} {list(self.scalars)}>'


class Video:

    def __init__(self, name, frames, fps=10):
        frames = list(frames) if isinstance(frames, (list, int)) else [frames]
        frames = [np.asarray(fs) for fs in frames]
        if not all(len(fs.shape) == 4 for fs in frames):
            raise ValueError('Video shape must be <L,H,W,C>')
        if not all(fs.shape == frames[0].shape for fs in frames):
            raise ValueError('All video frames must have the same dimensions')
        self.name = name
        self.frames = frames
        self.fps = fps

    @property
    def dims(self):
        return self.frames[0].shape

    def add(self, other):
        if not self == other:
            raise ValueError(f'Incompatible video types: {self} != {other}')
        self.frames += other.frames

    @property
    def tb_frames(self):
        # Combine into single array and transpose to <N, L, C, H, W>
        return np.stack(self.frames).transpose(0, 1, 4, 2, 3)

    def __eq__(self, other):
        return (
            isinstance(other, Video) and
            self.name == other.name and
            self.dims == other.dims and
            self.fps == other.fps
        )

    def __len__(self):
        return len(self.frames)

    def __repr__(self):
        return f'<Video {len(self)}x{self.dims} @{self.fps}fps>'

class EpochScalar:

    AGGS = {'mean', 'max', 'min', 'std', 'last'}

    def __init__(self, name, values, agg='mean', debug=False):
        self.name = name
        self.is_single_agg = not isinstance(agg, (tuple, list))
        self.aggs = list(agg) if not self.is_single_agg else [agg]
        self.debug = debug
        assert all(a in self.AGGS for a in self.aggs)
        self.values = list(values) if isinstance(values, (tuple, list)) else [values]

    def add(self, other):
        if isinstance(other, EpochScalar):
            if self != other:
                raise ValueError(
                    f'Inconsistent epoch scalar format: {self} != {other}'
                )
            self.values += other.values
        else:
            self.values.append(other)

    def __eq__(self, other):
        return (
            self.name == other.name and
            set(self.aggs) == set(other.aggs) and
            self.debug == other.debug
        )

    def __iter__(self):
        if not self.values:
            return
        processed = []
        for value in self.values:
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            processed.append(value)

        used = set()
        for agg in self.aggs:
            if agg in used:
                continue
            used.add(agg)
            name = None if self.is_single_agg else agg
            if agg == 'last':
                yield name, processed[-1]
            else:
                yield name, {
                    'mean': np.mean,
                    'std': np.std,
                    'min': np.min,
                    'max': np.max,
                }[agg](processed)


    def __repr__(self):
        debug = 'debug' if self.debug else ''
        return f'<EpochScalar: {self.name} {self.aggs} {debug}>'
