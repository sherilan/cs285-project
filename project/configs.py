

import argparse
import sys
import yaml


class Config:

    # Put default config here, e.g:
    # > env = 'HalfCheetahPyBulletEnv-v0'

    def __init__(self, **kwargs):
        for k, v in self.get_members().items():
            setattr(self, k, kwargs.pop(k, v))
        if kwargs:
            raise ValueError(f'Received unexpected arguments: {kwargs}')

    def build(self, *args, **kwargs):
        raise NotImplementedError()

    @classmethod
    def get_members(cls):
        """Returns a dict of all class-level members (across inheritance)"""
        return {
            k: v for cls in cls.mro()[:-1][::-1]
            for k, v in vars(cls).items()
            if not callable(v) and
            not isinstance(v, (property, classmethod, staticmethod)) and
            not k.startswith('__')
        }

    @classmethod
    def from_yaml(cls, filepath, **extra_cfg):
        with open(filepath, 'r') as f:
            return cls(**{**yaml.safe_load(f.read()), **extra_cfg})

    @classmethod
    def from_cli(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument('yaml-config', nargs='?')
        for k, v in cls.get_members().items():
            parser.add_argument(
                '--' + k.replace('_', '-'),
                type=yaml.safe_load,
                default=v,
            )
        args = vars(parser.parse_args())
        yaml_file = args.pop('yaml-config', None)
        if yaml_file:
            return cls.from_yaml(yaml_file, **args)
        else:
            return cls(**args)

    @property
    def dict(self):
        return {k: getattr(self, k) for k in self.get_members()}

    def save_yaml(self, filepath):
        with open(filepath, 'w') as f:
            f.write(yaml.dump(self.dict))


    def __str__(self):
        return yaml.dump(self.dict)
