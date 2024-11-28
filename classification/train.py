import argparse
import random
import os
import time
import pprint
from pathlib import Path
import json

import numpy as np
import torch
from torch import nn
import torchvision.transforms as T
from torchvision.datasets import CIFAR100, CIFAR10
from torchvision.models import resnet18
import geoopt
from sklearn.model_selection import train_test_split
import networkx as nx
import ot

from ahc import AverageHierarchicalCost
import resnet
from horospherical import (
    LinearLayer,
    HyperbolicLayer,
    HorosphericalLayer,
    HorosphericalDMM,
    BusemannPrototypes,
)
from gromov_protos import (
    label_map,
    build_cub_hierarchy,
    build_cifar10_hierarchy,
    build_cifar100_hierarchy,
    build_inat_hierarchy,
    compute_dist_matrix,
)
from datasets import (
    CUBDataset,
    CUB_MEAN,
    CUB_STD,
    SubsetDataset,
)
import resnet_cifar
from loss import PeBusePenalty
from learnt_prototypes import LearntPrototypes, DistortionLoss
from utils import RunLogger, Reporter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        default="euclidean",
        choices=[
            "euclidean", "hyperbolic",
            "horospherical", "horospherical_dmm",
            "busemann", "metric_guided",
        ],
    )
    parser.add_argument(
        "--dataset",
        choices=["cub", "cifar100", "cifar10"],
        default="cub",
    )
    parser.add_argument("--lambda_", type=float, default=0.)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--optimizer", choices=["adam", "sgd", "rmsprop"], default="adam")
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument(
        "--schedule",
        choices=["steplr", "reducelronplateau", "cosine",
                 "inat_schedule", "cosinewarmrestarts", "none"],
        default="steplr",
    )
    parser.add_argument(
        "--run_id",
        type=int,
        help="differentiate between runs with the same params",
    )
    parser.add_argument("--save_freq", type=int)
    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument("--proto_file")
    parser.add_argument("--gw_penalty", action="store_true")
    parser.add_argument("--momentum", default=None, type=float)
    parser.add_argument("--online_loss", type=float, default=0.0)
    return parser.parse_args()


class OnlineLoss(torch.nn.Module):
    def __init__(self, G):
        super().__init__()

        def is_leaf(G: nx.DiGraph, node) -> bool:
            return G.out_degree[node] == 0

        all_nodes = [n for n in G.nodes if is_leaf(G, n)]
        num_nodes = len(all_nodes)
        M_tree = torch.zeros(num_nodes, num_nodes)
        uG = nx.Graph(G)

        for i, n in enumerate(all_nodes):
            for j, m in enumerate(all_nodes):
                l = nx.shortest_path_length(uG, source=n, target=m)
                M_tree[i,j] = l

        self.register_buffer("M_tree", M_tree)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        M_sphere = 1 - (p @ p.T)
        num_nodes = p.size(0)
        p = q = torch.full((num_nodes,), 1/num_nodes, device=p.device)
        return ot.gromov_wasserstein2(self.M_tree, M_sphere, p, q)



def main(args):
    run_id = -1 if args.run_id is None else args.run_id

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def seed_node(_=None):
        seed = 42 + run_id
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.use_deterministic_algorithms(os.environ.get("CUBLAS_WORKSPACE_CONFIG", None) is not None)
    seed_node()

    if args.gw_penalty:
        assert args.method == "horospherical"

    if args.run_id is None:
        args.__delattr__("run_id")

    if args.save_freq is None:
        args.save_freq = args.epochs

    print(">> Args:")
    pprint.pprint(args.__dict__)

    identifier = os.getenv("SLURM_JOBID", str(time.time_ns()))
    global logger
    logger = RunLogger(name=args.method + "_" + identifier, config=args.__dict__)

    batch_size = 128
    if args.dataset == "cub":
        G = build_cub_hierarchy("assets/Birds_name.csv")
        num_classes = 200
        img_size = 256
        crop_size = 224

        train_transform = T.Compose([
            T.RandomResizedCrop(crop_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=CUB_MEAN, std=CUB_STD),
        ])

        test_transform = T.Compose([
            T.Resize(img_size),
            T.CenterCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=CUB_MEAN, std=CUB_STD),
        ])

        cub_root = "data/CUB_200_2011"

        crop_to_bb = False
        use_smart_crop = False

        train_dataset = CUBDataset(
            cub_root,
            train=True,
            transform=train_transform,
            crop_to_bb=crop_to_bb,
            use_smart_crop=use_smart_crop,
        )

        test_dataset = CUBDataset(
            cub_root,
            train=False,
            transform=test_transform,
            crop_to_bb=crop_to_bb,
            use_smart_crop=use_smart_crop,
        )

        assert len(train_dataset) == 5994
        assert len(test_dataset) == 5794

        model = resnet.ResNet32(args.dim)
    elif args.dataset == "cifar100":
        G = build_cifar100_hierarchy("./assets/graph_cifar100.csv")

        num_classes = 100
        train_transform = T.Compose(
            [
                T.RandomCrop(32, 4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.5071, 0.4867, 0.4408),
                    std=(0.2675, 0.2565, 0.2761),
                ),
            ]
        )
        test_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    mean=(0.5071, 0.4867, 0.4408),
                    std=(0.2675, 0.2565, 0.2761),
                ),
            ]
        )

        CIFAR_DATA_DIR = "../data/cifar100"
        train_dataset = CIFAR100(CIFAR_DATA_DIR, train=True, download=True, transform=train_transform)
        test_dataset = CIFAR100(CIFAR_DATA_DIR, train=False, transform=test_transform)

        model = resnet_cifar.ResNet(32, args.dim, 1)
    elif args.dataset == "cifar10":
        G = build_cifar10_hierarchy()
        num_classes = 10

        train_transform = T.Compose(
            [
                T.RandomCrop(32,4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.507, 0.487, 0.441),
                    std=(0.267, 0.256, 0.276),
                ),
            ]
        )
        test_transform = T.Compose(
            [
                T.Resize(32),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.507, 0.487, 0.441),
                    std=(0.267, 0.256, 0.276),
                ),
            ]
        )

        CIFAR_DATA_DIR = "../data/cifar10"
        train_dataset = CIFAR10(CIFAR_DATA_DIR, train=True, download=True, transform=train_transform)
        test_dataset = CIFAR10(CIFAR_DATA_DIR, train=False, transform=test_transform)

        model = resnet_cifar.ResNet(32, args.dim, 1)
    else:
        raise NotImplementedError(args.dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        worker_init_fn=seed_worker,
        batch_size=batch_size,
        num_workers=int(os.getenv("SLURM_CPUS_PER_TASK", "8")),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        worker_init_fn=seed_worker,
        num_workers=int(os.getenv("SLURM_CPUS_PER_TASK", "8")),
    )

    print(
        f">> Training with {len(train_dataset)} samples and testing"
        f" with {len(test_dataset)} samples."
    )

    ball = geoopt.PoincareBall()

    head_args = dict(
        n_in_feature=args.dim,
        n_decisions=num_classes,
    )

    if args.method == "busemann":
        head = BusemannPrototypes(proto_file=args.proto_file, **head_args)
    elif args.method == "horospherical":
        head = HorosphericalLayer(
            phi=args.lambda_ * args.dim,
            proto_file=args.proto_file,
            **head_args,
        )
    elif args.method == "horospherical_dmm":
        head = HorosphericalDMM(
            phi=args.lambda_ * args.dim,
            proto_file=args.proto_file,
            **head_args,
        )
    elif args.method == "hyperbolic":
        head = HyperbolicLayer(**head_args)
    elif args.method == "metric_guided":
        head = LearntPrototypes(n_prototypes=num_classes,
                                embedding_dim=args.dim)
    elif args.method == "euclidean":
        head = LinearLayer(**head_args)
    else:
        raise NotImplementedError(args.method)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "cuda is not available"

    optim, prefix = (
        (geoopt.optim, "Riemannian")
        if args.method in ("hyperbolic", "horospherical", "horospherical_dmm")
        else (torch.optim, "")
    )
    Optimizer = getattr(optim, prefix + {"sgd": "SGD", "adam": "Adam",
                                         "rmsprop": "RMSprop"}[args.optimizer])

    learning_rate = args.lr
    lr_steps = (
        args.epochs - args.epochs % 1000,
        args.epochs - args.epochs % 100,
    )

    print(f">> Optimizer = {Optimizer} with learning rate = {learning_rate}")

    opt_kwargs = {
        "lr": learning_rate,
        "weight_decay": args.weight_decay,
    }
    if args.momentum is not None:
        assert args.optimizer in ("sgd", "rmsprop")
        opt_kwargs["momentum"] = args.momentum

    opt = Optimizer(
        [{"params": model.parameters(), "initial_lr": learning_rate},
         {"params": head.parameters(), "initial_lr": learning_rate}],
        **opt_kwargs,
    )

    schedule = None
    if args.schedule == "reducelronplateau":
        schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, verbose=True)
    elif args.schedule == "cosine":
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, args.epochs,
            eta_min=learning_rate * 0.10,
        )
    elif args.schedule == "cosinewarmrestarts":
        schedule = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, args.epochs // 5, 1,
            eta_min=learning_rate * 0.10 ** 2,
            last_epoch=args.epochs,
        )
    elif args.schedule == "steplr":
        schedule = None
    else:
        schedule = None

    on_loss = OnlineLoss(G).to(device)
    bu_loss = PeBusePenalty(args.dim, mult=args.lambda_)
    ce_loss = nn.CrossEntropyLoss()

    if args.method == "metric_guided":
        dis_loss = DistortionLoss(compute_dist_matrix(G).to(device))

    model.to(device)
    head.to(device)

    reporter = Reporter(args)

    best_val_acc = 0.0
    best_val_checkpoint = None

    printed_epoch = False
    last_epoch = [-1]

    def compute_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        embeddings = model(x)
        if args.method == "busemann":
            embeddings = ball.expmap0(embeddings)
            loss = bu_loss(embeddings, head.prototypes[y, :])
            return loss

        if args.method in ["hyperbolic", "horospherical", "horospherical_dmm"]:
            embeddings = ball.expmap0(embeddings)

        logits = head.logits(embeddings)
        loss = ce_loss(logits, y)

        if args.method == "metric_guided":
            # NOTE: authors report using lambda_ = 1.0
            loss += dis_loss(head.prototypes)

        if args.online_loss != 0.0:
            # TODO: horospherical only
            loss += args.online_loss * on_loss(head.point)

        return loss

    for epoch in range(args.epochs):
        epoch_loss = 0.0

        model.train()
        head.train()

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            loss = compute_loss(x, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()

        val_loss = 0.

        model.eval()
        head.eval()

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                val_loss += compute_loss(x, y)

        if args.schedule == "reducelronplateau":
            schedule.step(val_loss)
        elif schedule is not None:
            schedule.step()
        elif (args.schedule == "steplr" and epoch in lr_steps) or (args.schedule == "inat_schedule" and (epoch + 1) % 20 == 0):
            learning_rate *= 0.94 if args.schedule == "inat_schedule" else 0.1
            print(f">> StepLR: Changing LR to {learning_rate}")
            for param_group in opt.param_groups:
                param_group["lr"] = learning_rate

        model.train()
        head.train()

        print(f"[{epoch}/{args.epochs}] loss = {epoch_loss:03.02f} "
              f"val_loss = {val_loss.item():03.02f}")
        logger.log({"training_loss": epoch_loss, "validation_loss": val_loss})

        if epoch % args.save_freq == 0:
            save_model(model, head, args, epoch=epoch)

        if epoch % args.eval_freq == 0:
            model.eval()
            head.eval()

            project = (
                ball.expmap0
                if args.method in ("hyperbolic", "horospherical", "horospherical_dmm")
                else nn.Identity()
            )

            def leaf_predict_fn(x):
                if args.method == "busemann":
                    return (
                        torch.nn.functional.normalize(model(x), dim=-1)
                        @ head.prototypes.T
                    )

                logits = head.logits(project(model(x)))
                logits = logits.view(logits.size(0), logits.size(-1))
                return logits

            print("== level 3 (species)")
            print("train ", end="")
            reporter.report(
                "species", "train",
                evaluate_dataset(G, leaf_predict_fn, train_loader, device),
                epoch,
            )
            print("test  ", end="")
            val_acc = evaluate_dataset(G, leaf_predict_fn, test_loader, device)
            reporter.report("species", "test", val_acc, epoch)

            if val_acc >= best_val_acc:
                if best_val_checkpoint is not None and Path(best_val_checkpoint).is_file():
                    os.remove(best_val_checkpoint)

                best_val_acc = val_acc
                best_val_checkpoint = save_model(model, head, args, f"best_val_acc_ep_{epoch}")

            compute_acc_at_higher_levels = False
            if compute_acc_at_higher_levels and args.dataset == "cub":
                G = build_cub_hierarchy("assets/Birds_name.csv")
                lm = torch.tensor(label_map(G, 3, 2), device=device)
                def transform_label(y): return lm[y]
                print("== level 2 (families)")
                print(f"train ", end="")
                acc_at_level = evaluate_dataset(G, leaf_predict_fn, train_loader, device, transform_label)
                reporter.report("families", "train", acc_at_level, epoch)
                print(f"test  ", end="")
                acc_at_level = evaluate_dataset(G, leaf_predict_fn, test_loader, device, transform_label)
                reporter.report("families", "test", acc_at_level, epoch)

    reporter.finish()
    save_model(model, head, args)

    logger.finish()


def save_model(model, head, args, epoch=None):
    if epoch is None:
        epoch = args.epochs
    checkpoint = {"model": model.state_dict(), "head": head.state_dict()}
    desc = os.path.join(*[str(k) + "-" + str(v) for k, v in args.__dict__.items()])
    filename = Path(os.path.join("models", desc, f"epoch_{epoch}.pckl"))
    filename.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, filename)
    print(f">> saved model at {filename}")
    return filename


@torch.no_grad()
def evaluate_dataset(G, predict_fn, loader, device, transform_label=None):
    well_pred = 0
    total = 0
    ahc = AverageHierarchicalCost(G).to(device)

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).squeeze()

        preds = predict_fn(x)
        preds = preds.argmax(-1)

        if transform_label is not None:
            y = transform_label(y)
            preds = transform_label(preds)

        ahc.fit(preds, y)
        well_pred += float((preds == y).sum())
        total += y.size(0)

    acc = 100 * well_pred / total
    acc_k = "train_accuracy" if loader.dataset.train else "test_accuracy"
    logger.log({acc_k: acc})
    print(f"acc = {acc:03.02f}%")
    print(f"ahc = {ahc.score().item():03.02f}")

    return acc

if __name__ == "__main__":
    main(parse_args())
