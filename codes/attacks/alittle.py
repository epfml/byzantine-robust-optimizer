import torch
import torch.distributed as dist
import numpy as np
from scipy.stats import norm

from ..simulators.worker import ByzantineWorker


class ALittleIsEnoughAttack(ByzantineWorker):
    """
    Args:
        n (int): Total number of workers
        m (int): Number of Byzantine workers
    """

    def __init__(self, n, m, z=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Number of supporters
        if z is not None:
            self.z_max = z
        else:
            s = np.floor(n / 2 + 1) - m
            cdf_value = (n - m - s) / (n - m)
            self.z_max = norm.ppf(cdf_value)
        self.n_good = n - m

    def get_gradient(self):
        return self._gradient

    def omniscient_callback(self):
        # Loop over good workers and accumulate their gradients
        gradients = []
        for w in self.simulator.workers:
            if not isinstance(w, ByzantineWorker):
                gradients.append(w.get_gradient())

        stacked_gradients = torch.stack(gradients, 1)
        mu = torch.mean(stacked_gradients, 1)
        std = torch.std(stacked_gradients, 1)

        self._gradient = mu - std * self.z_max

    def set_gradient(self, gradient) -> None:
        raise NotImplementedError

    def apply_gradient(self) -> None:
        raise NotImplementedError
