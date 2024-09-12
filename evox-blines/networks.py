from flax import linen as nn


class MLPPolicy(nn.Module):
    action_dim: int
    hidden_size: tuple[int] = (16, 16)

    @nn.compact
    def __call__(self, x):
        for size in self.hidden_size:
            x = nn.Dense(size)(x)
            x = nn.relu(x)

        x = nn.Dense(self.action_dim)(x)
        x = nn.tanh(x)

        return x
