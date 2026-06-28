# import torch
import torch.nn as nn


def _get_activation(name: str) -> nn.Module:
    """ Get PyTorch activation function from string name """
    activations = {
        # Classic
        'relu':         nn.ReLU(),
        'tanh':         nn.Tanh(),
        'sigmoid':      nn.Sigmoid(),
        # ReLU variants
        'leaky_relu':   nn.LeakyReLU(negative_slope=0.01),
        'relu6':        nn.ReLU6(),
        'elu':          nn.ELU(),
        'selu':         nn.SELU(),
        'prelu':        nn.PReLU(),          # has a learnable parameter
        # Modern / transformer-era
        'gelu':         nn.GELU(),           # used in BERT, GPT
        'silu':         nn.SiLU(),           # aka Swish, used in EfficientNet
        'mish':         nn.Mish(),           # smooth alternative to Swish
        # Smooth / bounded
        'softplus':     nn.Softplus(),       # smooth approximation of ReLU
        'softsign':     nn.Softsign(),       # smooth approximation of sign()
        'hardtanh':     nn.Hardtanh(),       # clipped linear, fast
        'hardsigmoid':  nn.Hardsigmoid(),    # piecewise linear sigmoid approx
        'hardswish':    nn.Hardswish(),      # efficient Swish approximation
        'logsigmoid':   nn.LogSigmoid(),
        # Shrinkage
        'tanhshrink':   nn.Tanhshrink(),     # x - tanh(x)
        'softshrink':   nn.Softshrink(),
        'hardshrink':   nn.Hardshrink(),
    }

    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}")

    return activations[name]


def compose_model(n_inputs: int, n_outputs: int, hidden_layer_activations: int, neurons: int,
                  dropout: float=0.2) -> nn.Sequential:
    """ Create a PyTorch Multi-Layer Perceptron.
        Args:
            - n_inputs: number of inputs (dimension)
            - n_outputs: number of outputs (dimension)
            - hidden_layer_activations: list of activation functions per hidden layer
            - neurons: number of neurons per hidden layer
            - dropout: drop-out rate
    """
    layers = []
    in_features = n_inputs

    for activation_name in hidden_layer_activations:
        linear = nn.Linear(in_features, neurons)
        nn.init.xavier_normal_(linear.weight) # Equivalent to Glorot in Keras
        nn.init.constant_(linear.bias, 0.1) # bias = 0.1
        layers.append(linear)
        layers.append(_get_activation(activation_name))
        layers.append(nn.Dropout(p=dropout))
        in_features = neurons

    # Output layer — linear, default init is fine
    layers.append(nn.Linear(in_features, n_outputs))

    return nn.Sequential(*layers)
