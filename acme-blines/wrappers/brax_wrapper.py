from acme import specs
from acme import types

import brax.envs
import dm_env
import brax
import numpy as np
import tree

class BraxWrapper(brax.envs):
    def __init__(self, environment: brax.envs):
        
        self._environment = environment
        self._reset_next_step = True
        self._last_info = None

        obs_space = self._environment