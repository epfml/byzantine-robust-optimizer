import torch

from .base import _BaseAggregator


def _compute_euclidean_distance(v1, v2):
    return (v1 - v2).norm()


def smoothed_weiszfeld(weights, alphas, z, nu, T):
    m = len(weights)
    if len(alphas) != m:
        raise ValueError

    if nu < 0:
        raise ValueError

    for t in range(T):
        betas = []
        for k in range(m):
            distance = _compute_euclidean_distance(z, weights[k])
            betas.append(alphas[k] / max(distance, nu))

        z = 0
        for w, beta in zip(weights, betas):
            z += w * beta
        z /= sum(betas)
    return z


class RFA(_BaseAggregator):
    r""""""

    def __init__(self, T, nu=0.1):
        self.T = T
        self.nu = nu
        super(RFA, self).__init__()

    def rfa(self, weights, z):
        # print("=> ", [w.norm() for i, w in enumerate(weights)])
        return smoothed_weiszfeld(weights, alphas, z, nu=self.nu, T=self.T)

    def __call__(self, inputs):
        alphas = [1 / len(inputs) for _ in inputs]
        z = torch.zeros_like(inputs[0])
        return smoothed_weiszfeld(inputs, alphas, z=z, nu=self.nu, T=self.T)

    def __str__(self):
        return "RFA(T={},nu={})".format(self.T, self.nu)
