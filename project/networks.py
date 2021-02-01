
import torch
import torch.nn as nn
import torch.nn.functional as F



def get_activation(act, **kwargs):
    if act is None:
        return torch.nn.Identity()
    elif isinstance(act, str):
        return {
            'tanh': torch.nn.Tanh,
            'relu': torch.nn.ReLU,
            'leaky_relu': nn.LeakyReLU,
            'sigmoid': nn.Sigmoid,
            'selu': nn.SELU,
            'softplus': nn.Softplus,
            'identity': nn.Identity,
        }[act](**kwargs)
    elif isinstance(act, torch.nn.Module):
        return act(**kwargs)
    else:
        raise ValueError(f'Bad argument: act "{act}" not understood')



def get_initializer(init):
    if init is None:
        return lambda x: x
    if callable(init):
        return init
    if isinstance(init, str):
        return {
            'xavier_normal': nn.init.xavier_normal_,
            'zeros': nn.init.zeros_,
        }[init]
    else:
        raise ValueError(f'Bad argument: init "{init}" not understood')



class LinearAct(nn.Linear):

    def __init__(
        self,
        n_ipt,
        n_out,
        bias=True,
        act=None,
        w_init=None,
        b_init=None
    ):
        super().__init__(n_ipt, n_out, bias)
        self.n_ipt = n_ipt
        self.n_out = n_out
        self.use_bias = bias
        self.act = act
        self.w_init = get_initializer(w_init) if w_init else None
        self.b_init = get_initializer(b_init) if b_init else None
        self.act_f = get_activation(act) if act else None
        self.initialize()

    def initialize(self):
        if not self.w_init is None:
            self.w_init(self.weight)
        if not self.b_init is None:
            self.b_init(self.bias)

    def forward(self, x):
        x = super().forward(x)
        return self.act_f(x) if not self.act_f is None else x

    def __repr__(self):
        ni, no, b, a = self.n_ipt, self.n_out, self.use_bias, self.act
        return f'LinearAct(n_ipt={ni}, n_out={no}, bias={b}, act={a})'



class MLP(nn.ModuleList):

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        hidden_num,
        hidden_act='tanh',
        output_act=None,
        w_init='xavier_normal',
        b_init='zeros',
        hidden_w_init=None,
        hidden_b_init=None,
        output_w_init=None,
        output_b_init=None,
    ):
        super().__init__()
        # Initialize lists of layer sizes and activations
        sizes = [input_size] + [hidden_size] * hidden_num + [output_size]
        acts = [hidden_act] * hidden_num + [output_act]
        # Initialize lists of parameter intializers
        w_inits = [hidden_w_init] * hidden_num + [output_w_init]
        b_inits = [hidden_b_init] * hidden_num + [output_b_init]
        # Build
        for i, (n_ipt, n_out, act, _w_init, _b_init) in enumerate(zip(
            sizes[:-1], sizes[1:], acts, w_inits, b_inits
        )):
            self.append(
                LinearAct(
                    n_ipt=n_ipt,
                    n_out=n_out,
                    act=act,
                    w_init=(_w_init or w_init),
                    b_init=(_b_init or b_init),
                )
            )

    def forward(self, *xs):
        x = torch.cat(list(xs), dim=-1)
        for layer in self:
            x = layer(x)
        return x



class MultiHeadMLP(nn.Module):

    def __init__(
        self,
        input_size,
        output_sizes,
        hidden_size,
        hidden_num,
        hidden_act='tanh',
        output_act=None,
        w_init='xavier_normal',
        b_init='zeros',
        hidden_w_init=None,
        hidden_b_init=None,
        output_w_init=None,
        output_b_init=None,
        output_names=None,
    ):
        super().__init__()
        # Initialize lists of layer sizes and activations
        sizes = [input_size] + [hidden_size] * hidden_num
        # Build main body
        self.layers = nn.ModuleList()
        for i, (n_ipt, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            self.layers.append(
                LinearAct(
                    n_ipt=n_ipt,
                    n_out=n_out,
                    act=hidden_act,
                    w_init=(hidden_w_init or w_init),
                    b_init=(hidden_b_init or b_init),
                )
            )
        # Build heads
        self.heads = nn.ModuleList()
        output_names = output_names or list(range(len(output_sizes)))
        assert len(output_names) == len(output_sizes)
        for name, output_size in zip(output_names, output_sizes):
            self.heads.add_module(
                str(name),
                LinearAct(
                    n_ipt=sizes[-1],
                    n_out=output_size,
                    act=output_act,
                    w_init=(output_w_init or w_init),
                    b_init=(output_b_init or b_init),
                )
            )


    def forward(self, *xs):
        x = torch.cat(list(xs), dim=-1)
        for layer in self.layers:
            x = layer(x)
        return tuple(head(x) for head in self.heads)
