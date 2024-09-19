import jax.numpy as jnp
from flax import linen as nn


class MLPPolicy(nn.Module):
    action_dim: int
    hidden_layer_sizes: tuple[int] = (16, 16)
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        for size in self.hidden_layer_sizes:
            x = nn.Dense(size)(x)
            x = nn.relu(x)

        x = nn.Dense(self.action_dim)(x)
        x = nn.tanh(x)


        x = jnp.clip(x, -1+self.eps, 1-self.eps)

        return x
