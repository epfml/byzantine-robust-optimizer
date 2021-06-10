"""
"""
import argparse
import numpy as np
import os
import torch
from torchvision import datasets

from torch.nn.modules.loss import CrossEntropyLoss
from codes.sampler import DistributedSampler
from codes.simulators.simulator import (
    ParallelTrainer,
    DistributedEvaluator,
)
from codes.simulators.worker import WorkerWithMomentum, ByzantineWorker
from codes.simulators.server import TorchServer


# from codes.tasks.mnist import mnist, Net
from codes.tasks.cifar10 import cifar10, get_resnet20
from codes.utils import top1_accuracy, initialize_logger
from codes.attacks.labelflipping import LableFlippingWorker
from codes.attacks.bitflipping import BitFlippingWorker

from codes.attacks.xie import IPMAttack
from codes.attacks.alittle import ALittleIsEnoughAttack

from codes.aggregator.coordinatewise_median import CM
from codes.aggregator.clipping import Clipping
from codes.aggregator.rfa import RFA
from codes.aggregator.trimmed_mean import TM
from codes.aggregator.krum import Krum
from codes.aggregator.base import Mean


EXP_ID = __file__[:-3]

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../"
DATA_DIR = ROOT_DIR + "datasets/"
EXP_DIR = ROOT_DIR + f"outputs/{EXP_ID}/"

parser = argparse.ArgumentParser(description="")
parser.add_argument("--use-cuda", action="store_true", default=False)
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--log_interval", type=int, default=10)

parser.add_argument("--attack", type=str, help="Select from BF and LF.")
parser.add_argument("--agg", type=str, help="Aggregator.")
parser.add_argument("--momentum", type=float, default=0, help="momentum")
parser.add_argument(
    "--inner-iterations", type=int, default=1, help="[HP]: number of inner iterations."
)

args = parser.parse_args()


N_WORKERS = 25
N_BYZ = 11 if args.attack == "IPM" else 5
BATCH_SIZE = 32
TEST_BATCH_SIZE = 128
MAX_BATCHES_PER_EPOCH = 9999999
EPOCHS = 100
LR = 0.1
MOMENTUM = args.momentum

# LOG_DIR = EXP_DIR + "log"
LOG_DIR = (
    EXP_DIR
    + ("debug/" if args.debug else "")
    + f"f{N_BYZ}_{args.attack}_{args.agg}_m{args.momentum}_seed{args.seed}"
)


def get_sampler_callback(rank):
    def sampler_callback(x):
        return DistributedSampler(
            num_replicas=N_WORKERS - N_BYZ,
            rank=rank if rank < N_BYZ else rank - N_BYZ,
            shuffle=True,
            dataset=x,
        )

    return sampler_callback


def _get_aggregator():
    if args.agg == "avg":
        return Mean()

    if args.agg == "cm":
        return CM()

    if args.agg == "cp":
        return Clipping(tau=100, n_iter=1)

    if args.agg == "rfa":
        return RFA(T=3)

    if args.agg == "tm":
        return TM(b=N_BYZ)

    if args.agg == "krum":
        return Krum(n=N_WORKERS, f=N_BYZ, m=1)

    raise NotImplementedError(args.agg)


def initialize_worker(
    trainer, worker_rank, model, optimizer, loss_func, device, kwargs
):
    train_loader = cifar10(
        data_dir=DATA_DIR,
        train=True,
        download=True,
        batch_size=BATCH_SIZE,
        sampler_callback=get_sampler_callback(worker_rank),
        dataset_cls=datasets.CIFAR10,
        drop_last=True,  # Exclude the influence of non-full batch.
        **kwargs,
    )
    # NOTE: The first N_BYZ nodes are Byzantine
    if worker_rank < N_BYZ:
        if args.attack == "BF":
            return BitFlippingWorker(
                data_loader=train_loader,
                model=model,
                loss_func=loss_func,
                device=device,
                optimizer=optimizer,
                **kwargs,
            )

        if args.attack == "LF":
            return LableFlippingWorker(
                revertible_label_transformer=lambda target: 9 - target,
                data_loader=train_loader,
                model=model,
                loss_func=loss_func,
                device=device,
                optimizer=optimizer,
                **kwargs,
            )

        if args.attack == "IPM":
            attacker = IPMAttack(
                epsilon=0.1,
                data_loader=train_loader,
                model=model,
                loss_func=loss_func,
                device=device,
                optimizer=optimizer,
                **kwargs,
            )
            attacker.configure(trainer)
            return attacker

        if args.attack == "ALIE":
            attacker = ALittleIsEnoughAttack(
                n=N_WORKERS,
                m=N_BYZ,
                data_loader=train_loader,
                model=model,
                loss_func=loss_func,
                device=device,
                optimizer=optimizer,
                **kwargs,
            )
            attacker.configure(trainer)
            return attacker
        raise NotImplementedError(f"No such attack {args.attack}")
    return WorkerWithMomentum(
        momentum=MOMENTUM,
        data_loader=train_loader,
        model=model,
        loss_func=loss_func,
        device=device,
        optimizer=optimizer,
        **kwargs,
    )


def main(args):
    initialize_logger(LOG_DIR)

    if args.use_cuda and not torch.cuda.is_available():
        print("=> There is no cuda device!!!!")
        device = "cpu"
    else:
        device = torch.device("cuda" if args.use_cuda else "cpu")

    kwargs = {"pin_memory": True} if args.use_cuda else {}

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = get_resnet20(use_cuda=args.use_cuda, gn=False).to(device)
    # NOTE: no momentum
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    loss_func = CrossEntropyLoss().to(device)

    metrics = {"top1": top1_accuracy}

    server_opt = torch.optim.SGD(model.parameters(), lr=LR)
    server = TorchServer(server_opt)

    trainer = ParallelTrainer(
        server=server,
        aggregator=_get_aggregator(),
        pre_batch_hooks=[],
        post_batch_hooks=[],
        max_batches_per_epoch=MAX_BATCHES_PER_EPOCH,
        log_interval=args.log_interval,
        metrics=metrics,
        use_cuda=args.use_cuda,
        debug=False,
    )

    test_loader = cifar10(
        data_dir=DATA_DIR,
        train=False,
        download=True,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        **kwargs,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        server_opt, milestones=[75], gamma=LR
    )

    evaluator = DistributedEvaluator(
        model=model,
        data_loader=test_loader,
        loss_func=loss_func,
        device=device,
        metrics=metrics,
        use_cuda=args.use_cuda,
        debug=False,
    )

    for worker_rank in range(N_WORKERS):
        worker = initialize_worker(
            trainer=trainer,
            worker_rank=worker_rank,
            model=model,
            optimizer=optimizer,
            loss_func=loss_func,
            device=device,
            kwargs={},
        )
        trainer.add_worker(worker)

    for epoch in range(1, EPOCHS + 1):
        trainer.train(epoch)
        evaluator.evaluate(epoch)
        trainer.parallel_call(lambda w: w.data_loader.sampler.set_epoch(epoch))
        scheduler.step()
        print(f"E={epoch}; Learning rate = {scheduler.get_lr()[0]:}")


if __name__ == "__main__":
    main(args)
