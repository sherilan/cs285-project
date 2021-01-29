

import numpy as np
import torch



class RingBuffer:

    def __init__(self, capacity, keys, dims, dtypes=np.float32, seed=None):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0
        self.buffer, self.keys, self.dims, self.dtypes = self.init_buffer(
            keys, dims, dtypes
        )
        self.random = np.random
        if seed:
            self.seed(seed)

    def seed(self, seed):
        self.random = np.random.RandomState(seed)

    def add(self, values):
        # Check that values is a dict matching the keys the buffer is set up with
        if not isinstance(values, dict):
            raise ValueError(f'Received non-dict values: {type(values)}')
        if set(values) != set(self.keys):
            raise ValueError(
                f'Received inconsitent value keys: '
                f'{set(values)} != {set(self.keys)}'
            )
        # Check that all values have the same length
        lengths = [len(v) for v in values.values()]
        length = lengths[0]
        if length == 0:
            return
        if not all(l == length for l in lengths):
            raise ValueError(f'Received inconsistent value lengths: {lengths}')
        # If the buffer can fit the data without looping the pointer, fill it in
        if self.ptr + length <= self.capacity:
            for key, value in values.items():
                self.buffer[key][self.ptr:self.ptr + length] = value
            self.ptr = (self.ptr + length) % self.capacity
            self.size = min(self.size + length, self.capacity)
        # Otherwise, chop up values and fill in
        else:
            i = 0
            while i < length:
                j = i + min(length - i, self.capacity - self.ptr)
                values_chunk = {
                    key: value[i:j] for key, value in values.items()
                }
                self.add(values_chunk)
                i = j


    def sample(self, n, replace=False, as_dict=True, tensor=False, device=None):
        if len(self) <= 0:
            raise ValueError(f'Cannot sample from empty buffer')
        if n > len(self) and not replace:
            raise ValueError(f'Cannot sample {n} values without replacement')
        # Generate indices
        if replace:
            idx = self.random.choice(self.size, size=n)
        else:
            idx = self.random.permutation(self.size)[:n]
        # Sample on indices
        data = {k: v[idx] for k, v in self.buffer.items()}
        # Optionally cast to torch tensors
        if tensor:
            data = {
                k: torch.as_tensor(v, device=device) for k, v in data.items()
            }
        if as_dict:
            return data
        else:
            return tuple(data[k] for k in self.keys)


    def init_buffer(self, keys, dims, dtypes):
        if not isinstance(keys, (tuple, list, np.ndarray)):
            raise ValueError(
                f'Buffer "keys" argument must be a list-like. Recieved "{keys}"'
            )
        if not isinstance(dims, (tuple, list, np.ndarray)):
            raise ValueError(
                f'Buffer "dims" argument must be a list-like. Received "{dims}"'
            )
        if not len(keys) == len(dims):
            raise ValueError(
                f'Buffer "dims" and "keys" must have same length. Received '
                f'len({keys}) != len({dims})'
            )
        keys, dims = list(keys), list(dims)
        if not isinstance(dtypes, (tuple, list, np.ndarray)):
            dtypes = [dtypes] * len(keys)
        elif len(dtypes) != len(dims):
            raise ValueError(
                f'Buffer "dtypes" must have same size as "keys" and "dims" '
                f'if specified as a list-like. Received "{dtypes}"'
            )
        buffer = {}
        for key, dim, dtype in zip(keys, dims, dtypes):
            if dim is None:
                dim = (self.capacity,)
            elif np.isscalar(dim):
                dim = (self.capacity, dim)
            else:
                dim = (self.capacity,) + tuple(dim)
            buffer[key] = np.zeros(dim, dtype=dtype)
        return buffer, keys, dims, dtypes

    def __getitem__(self, key):
        return self.buffer[key]

    def __lshift__(self, values):
        self.add(values)

    def __len__(self):
        return self.size

    def __repr__(self):
        return f'<{self.__class__.__name__} {len(self)}/{self.capacity}>'
