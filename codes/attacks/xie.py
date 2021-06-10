"""
A better name will be Inner Product Manipulation Attack.
"""

from ..simulators.worker import ByzantineWorker


class IPMAttack(ByzantineWorker):
    def __init__(self, epsilon, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self._gradient = None

    def get_gradient(self):
        return self._gradient

    def omniscient_callback(self):
        # Loop over good workers and accumulate their gradients
        gradients = []
        for w in self.simulator.workers:
            if not isinstance(w, ByzantineWorker):
                gradients.append(w.get_gradient())

        self._gradient = -self.epsilon * (sum(gradients)) / len(gradients)

    def set_gradient(self, gradient) -> None:
        raise NotImplementedError

    def apply_gradient(self) -> None:
        raise NotImplementedError
