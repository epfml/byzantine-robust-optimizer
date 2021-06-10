import torch


class TorchServer(object):
    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def apply_gradient(self) -> None:
        self.optimizer.step()

    def set_gradient(self, gradient: torch.Tensor) -> None:
        beg = 0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # for p in self.model.parameters():
                end = beg + len(p.grad.view(-1))
                x = gradient[beg:end].reshape_as(p.grad.data)
                p.grad.data = x.clone().detach()
                beg = end
