import torch.nn as nn


class MLP_tiny(nn.Module):
    """Original small ECG classifier: 2560 -> 128 -> 2 (~328K params).

    Kept as a reference/baseline -- small enough that an H100 is almost
    entirely idle waiting on Python/DataLoader overhead rather than doing
    meaningful compute (see profiling discussion: forward+backward+optimizer
    combined are a few percent of total step time at this size).
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)


class MLP(nn.Module):
    """Larger ECG classifier: 2560 -> 2048 x4 -> 2 (~17.85M params, ~54x
    MLP_tiny). Sized to give the GPU enough matmul FLOPs per kernel launch
    to be worth profiling/optimizing with TF32 and mixed precision -- this
    is the threshold where precision flags start showing measurable speedup
    *at large batch sizes* (e.g. bs=8192). At small batch sizes (bs=256)
    the GPU is still underutilized per kernel launch even at this width;
    see MLP_huge for the size where bs=256 should become compute-bound.

    BatchNorm1d after each hidden layer for stable training at this depth;
    Dropout to offset the higher overfitting risk from the larger capacity.
    """

    def __init__(self, input_dim=1280 * 2, hidden_dim=2048, num_classes=2,
                num_hidden_layers=4, dropout=0.2):
        super().__init__()

        layers = [nn.Flatten(), nn.Linear(input_dim, hidden_dim),
                 nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout)]

        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim),
                      nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout)]

        layers.append(nn.Linear(hidden_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLP_huge(nn.Module):
    """Much larger ECG classifier: 2560 -> 8192 x4 -> 2 (~285M params, ~16x MLP).

    Sized specifically to test the hypothesis that 'work per kernel launch'
    -- not batch size per se -- is what determines whether TF32/mixed
    precision can help. Each layer's matmul scales as ~hidden_dim^2, so
    going from hidden_dim=2048 (MLP) to hidden_dim=8192 (MLP_huge) gives
    roughly 16x more FLOPs per matmul kernel.

    Prediction: at bs=256 with this model, the per-launch compute should
    finally be large enough that TF32/mixed_precision show measurable
    speedup -- something they could not do at MLP's 2048-width even at
    bs=256, because each individual matmul finished too quickly relative
    to fixed kernel-launch overhead. If this prediction holds, it confirms
    the workshop lesson that batch size and model width are interchangeable
    levers on the same underlying variable (work amortized per launch).

    Memory caveat: 285M params at fp32 + Adam (which keeps two extra
    moment tensors per parameter) is ~3.4GB just for optimizer state on
    top of the 1.1GB for parameters. Plus activations at bs=8192. Easily
    fits in 80GB H100 VRAM, but worth being aware of if combined with
    --data_on_gpu (the dataset itself takes 7.23GB on top).
    """

    def __init__(self, input_dim=1280 * 2, hidden_dim=8192, num_classes=2,
                num_hidden_layers=4, dropout=0.2):
        super().__init__()

        layers = [nn.Flatten(), nn.Linear(input_dim, hidden_dim),
                 nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout)]

        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim),
                      nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout)]

        layers.append(nn.Linear(hidden_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_model(name):
    """Construct a model by name -- used by --model in train_MLP.py."""
    if name == "tiny":
        return MLP_tiny()
    elif name == "big":
        return MLP()
    elif name == "huge":
        return MLP_huge()
    raise ValueError(f"unknown model: {name}")
