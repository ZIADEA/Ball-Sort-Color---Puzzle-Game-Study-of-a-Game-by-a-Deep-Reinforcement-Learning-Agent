import numpy as np
import gymnasium as gym
from gymnasium import spaces


class OneHotObservationWrapper(gym.ObservationWrapper):
    """
    Convert integer observations (n_max, height) into one-hot (n_max, height, n_channels).
    Channels: 0-12 colors, 13 for padding (-1).
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.n_max = getattr(env, "n_max")
        self.height = getattr(env, "height")
        self.n_channels = 14  # 0..12 colors + padding channel
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(env.n_max, env.height, self.n_channels),
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        one_hot = np.zeros((self.n_max, self.height, self.n_channels), dtype=np.float32)
        for t in range(self.n_max):
            for h in range(self.height):
                val = observation[t, h]
                if val == -1:
                    idx = 13
                else:
                    idx = int(val)
                if 0 <= idx < self.n_channels:
                    one_hot[t, h, idx] = 1.0
        return one_hot

    def action_masks(self):
        """Delegate action masks to the underlying env (needed for MaskablePPO)."""
        if hasattr(self.env, "action_masks"):
            return self.env.action_masks()
        raise AttributeError("Underlying env does not implement action_masks")
