import math
import argparse
from typing import List
import pandas as pd
import torch
import sys
import networkx as nx
import ot
import matplotlib.pyplot as plt
import numpy as np
import json


# NOTE: this code can be improved further by collapsing children onto their parent
# ....  if they are single child which means that p_parent == p_child and therefore.
# ....  d(p_parent,p_child) == 0.


def hierarchy_depth(G: nx.DiGraph) -> int:
    """
    Returns the depth of the hierarchy
    """
    return nx.dag_longest_path_length(G)


def is_decision_node(G: nx.DiGraph, node) -> bool:
    """
    Returns whether there need to be a Tree decision node at node
    False for leaf nodes and node where there is only one child.
    """
    return G.out_degree[node] > 1


def is_leaf(G: nx.DiGraph, node) -> bool:
    """
    Returns whether or not the given node is a leaf node.
    """
    return G.out_degree[node] == 0


def parent(G: nx.DiGraph, node, nth: int = 1):
    nth -= 1
    parent_ = next(G.predecessors(node))
    if nth == 0:
        return parent_
    return parent(G, parent_, nth)


def classes_at_level(G: nx.DiGraph, level: int) -> List:
    """
    Returns the class names which are the given level.
    """
    return [
        node for node in G.nodes() if nx.shortest_path_length(G, "root", node) == level
    ]


def label_map(G: nx.DiGraph, from_: int, to: int) -> List:
    """
    Returns a list `map` such that `map[i]` gives the class label at level `to`.
    Parameters
    ==========
        G: nx.DiGraph
        from_: int
        to: int
    Returns
    =======
        map: List
    """
    assert from_ > to, f"invalid map from level {from_} to {to}"
    classes_from = classes_at_level(G, from_)
    classes_to = classes_at_level(G, to)
    return [
        classes_to.index(parent(G, cls_from, from_ - to)) for cls_from in classes_from
    ]


def longest_path_length(G: nx.DiGraph) -> int:
    return nx.dag_longest_path_length(G)


def gen2d(G):
    pos = nx.nx_agraph.graphviz_layout(G, prog="twopi", args="")
    root = pos["root"]

    leaves = [node for node in G.nodes if is_leaf(G,node)]
    leaves_x = [pos[leaf][0] for leaf in leaves]
    leaves_y = [pos[leaf][1] for leaf in leaves]

    leaves_x_centered = [x - pos["root"][0] for x in leaves_x]
    leaves_y_centered = [y - pos["root"][1] for y in leaves_y]
    norms = [
        math.sqrt(x**2 + y**2) for x, y in zip(leaves_x_centered, leaves_y_centered)
    ]

    cool_protos_x = [x / norm for x, norm in zip(leaves_x_centered, norms)]
    cool_protos_y = [y / norm for y, norm in zip(leaves_y_centered, norms)]

    cool_protos = np.stack(
        (
            cool_protos_x,
            cool_protos_y,
        )
    ).T
    return cool_protos


def build_nuscenes_hierarchy() -> nx.DiGraph:
    # Order of output
    nu_scenes_classes = [
        "Ignore", # (0)
        "Barrier", # (1)
        "Bicycle", # (2)
        "Bus", # (3)
        "Car", # (4)
        "Construction Vehicle", # (5)
        "Motorcycle", # (6)
        "Pedestrian", # (7)
        "Traffic Cone", # (8)
        "Trailer", # (9)
        "Truck", # (10)
        "Driveable Surface", # (11)
        "Other Flat", # (12)
        "Sidewalk", # (13)
        "Terrain", # (14)
        "Manmade", # (15)
        "Vegetation", # (16)
    ]

    G = nx.DiGraph()
    G.add_node("root")
    for cls in nu_scenes_classes:
        G.add_node(cls)

    tree = {
        "Moveable_Object": {
            "dummy": ["Barrier", "Traffic Cone"],
        },
        "Vehicle" : {
            "4-wheeled": ["Bus", "Car", "Construction Vehicle",
                          "Truck", "Trailer"],
            "2-wheeled": ["Motorcycle", "Bicycle"],
        },
        "Pedestrian_1": {
            "Pedestrian_2": ["Pedestrian"],
            # "Adult": ["Adult", "Construction Worker", "Police Officer"],
            # "Child_p": ["Child"],
        },
        "Flat": {
            "dummy3": ["Driveable Surface", "Other Flat", "Sidewalk", "Terrain"],
        },
        "Static": {
            "dummy4": ["Manmade", "Vegetation"],
        },
        "Ignore_1": {
            "Ignore_2": ["Ignore"],
        }
    }

    def explore_dict(p, d):
        for k, v in d.items():
            G.add_edge(p,k)

            if isinstance(v, dict):
                explore_dict(k,v)
            else:
                for vv in v:
                    G.add_edge(k,vv)

    explore_dict("root",tree)

    return G

def build_inat_hierarchy(data) -> nx.DiGraph:
    G = nx.DiGraph()

    for cat in data["categories"]:

        name = cat["name"]
        genus = cat["genus"]
        family = cat["family"]
        order = cat["order"]
        cls = cat["class"]
        phylum = cat["phylum"]
        kingdom = cat["kingdom"]

        G.add_edge(genus, name)
        G.add_edge(family, genus)
        G.add_edge(order, family)
        G.add_edge(cls, order)
        G.add_edge(phylum, cls)
        G.add_edge(kingdom, phylum)

    kingdoms = [n for n in G.nodes if G.in_degree[n] == 0]
    for k in kingdoms: G.add_edge("root", k)

    return G

def build_cub_hierarchy(path_to_csv: str) -> nx.DiGraph:
    df: pd.DataFrame = pd.read_csv(path_to_csv)  # type: ignore

    ### Build graph

    G = nx.DiGraph()

    for _, row in df.iterrows():
        order = row.order
        family = row.family
        species = row.species
        G.add_edge(order, family)
        G.add_edge(family, species)

    for order in df.order.unique():
        G.add_edge("root", order)

    return G


def build_resisc_hierarchy() -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_edge("root", "transport")
    G.add_edge("root", "maritime")
    G.add_edge("transport", "bridge")
    G.add_edge("transport", "freeway")
    G.add_edge("maritime", "beach")
    G.add_edge("maritime", "island")
    return G


def build_cifar10_hierarchy() -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_edge("root", "vehicle")
    G.add_edge("root", "animal")

    G.add_edge("vehicle", "airplane")
    G.add_edge("vehicle", "automobile")

    G.add_edge("animal", "bird")
    G.add_edge("animal", "cat")
    G.add_edge("animal", "deer")
    G.add_edge("animal", "dog")
    G.add_edge("animal", "frog")
    G.add_edge("animal", "horse")

    G.add_edge("vehicle", "ship")
    G.add_edge("vehicle", "truck")

    return G


def build_cifar100_hierarchy(path_to_csv: str) -> nx.DiGraph:
    df: pd.DataFrame = pd.read_csv(path_to_csv, header=None)
    G = nx.DiGraph()
    for _, (src, dst) in df.iterrows():
        if not G.has_node(dst):
            G.add_node(dst)
        if not G.has_node(src):
            G.add_node(src)
        G.add_edge(src, dst)
    return G


def build_graph_from_dataset(dataset) -> nx.DiGraph:
    "dataset needs to be a torchvision.datasets with class_to_idx"
    G = nx.DiGraph()
    items = list(dataset.class_to_idx.keys())
    items = sorted(items, key=dataset.class_to_idx.__getitem__)
    for item in items:
        G.add_edge("root", item)
    return G


def build_cityscapes_hierarchy() -> nx.DiGraph:
    G = nx.DiGraph()

    tree = dict(
        flat=["road", "sidewalk"],
        construction=["building", "wall", "fence"],
        object=["pole", "traffic light", "traffic sign"],
        nature=["vegetation", "terrain"],
        sky=["sky_"],
        human=["person", "rider"],
        vehicle=["car", "truck", "bus", "train", "motorcycle", "bicycle"],
    )

    for k, v in tree.items():
        for cls in v:
            G.add_edge(k,cls)
        G.add_edge("root", k)

    return G


def build_pascal_voc_hierarchy():
    G = nx.DiGraph()
    VOC_CLASSES = ('__background__', # always index 0
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor') 

    G.add_node("root")
    for cls in VOC_CLASSES:
        G.add_node(cls)

    tree = dict(
        animals=("cat", "dog", "cow", "horse", "sheep", "bird"),
        household=("bottle", "chair", "sofa", "diningtable", "tvmonitor", "pottedplant"),
        vehicles=("car", "bus", "bicycle", "motorbike", "aeroplane", "boat", "train"),
        human=("person",),
        other=("__background__",),
    )

    for k, v in tree.items():
        for cls in v:
            G.add_edge(k, cls)
        G.add_edge("root", k)

    return G


def flatten_hierarchy(G: nx.DiGraph) -> nx.DiGraph:
    flattened = nx.DiGraph()
    leaves = [n for n, d in G.out_degree if d == 0]
    for leaf in leaves:
        flattened.add_edge("root", leaf)
    return flattened

def hierarchical_node_mapping(G: nx.DiGraph) -> dict[str,int]:
    """
    Returns a mapping which indicates where to position each node
    Nodes that need no prototype will have the index `-1`. The index
    will be later be used by the `NxHierarchyEvaluator`.

    Parameters
    ==========
        G: nx.DiGraph - the hierarchy
    Returns
    =======
        mapping: dict[str,int] - mapping between node and prototype index
    """
    mapping = dict()

    topo_sort = nx.topological_sort(G)
    topo_sort = list(topo_sort)

    count = 0
    for node in topo_sort:
        if not is_decision_node(G, node):
            continue

        for succ in G.successors(node):
            mapping[succ] = count
            count += 1

    return mapping

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dim", type=int, default=3)
parser.add_argument("--input-protos")
parser.add_argument("--dataset", choices=["cub", "cifar100", "cifar10",
                                          "cityscapes", "pascalvoc", "nuscenes",
                                          "inat", "inat_small"], default="cub")
parser.add_argument("--lr", type=float, default=0.5)
parser.add_argument("--save-all", action="store_true", help="save all prototypes, not only leaves")
parser.add_argument("--only-leaves", action="store_true", help="perform gw computation with only leaves")
parser.add_argument("--output-file")
args = parser.parse_args()

if args.dataset == "cub":
    G = build_cub_hierarchy("./assets/Birds_name.csv")
elif args.dataset == "cifar10":
    G = build_cifar10_hierarchy()
elif args.dataset == "cifar100":
    G = build_cifar100_hierarchy("./assets/graph_cifar100.csv")
elif args.dataset == "cityscapes":
    G = build_cityscapes_hierarchy()
elif args.dataset == "pascalvoc":
    G = build_pascal_voc_hierarchy()
elif args.dataset == "nuscenes":
    G = build_nuscenes_hierarchy()
elif "inat" in args.dataset:
    with open("assets/val2019.json") as f:
        data = json.load(f)
    G = build_inat_hierarchy(data)

    if args.dataset == "inat_small":
        G.remove_node("root")
        G.remove_node("SMHLVG")
        G.remove_node("BKJDSR")
        nx.relabel_nodes(G, {"IKZCCU": "root"}, False)

        pred = nx.predecessor(G, "root")
        for n in list(G.nodes):
            if n not in pred:
                G.remove_node(n)

else:
    raise NotImplementedError(args.dataset)

all_nodes = (
    list(G.nodes)
    if not args.only_leaves else
    [n for n in G.nodes if is_leaf(G, n)]
)

num_nodes = len(all_nodes)
dim =args.dim
depth = hierarchy_depth(G)

if dim == 2:
    protos = gen2d(G)

    num_cls = protos.shape[0]
    suffix = "ol" if args.only_leaves else ""

    if args.output_file is None:
        proto_path = f"prototypesgromov{suffix}-{dim}d-{num_cls}c.npy"
    else:
        proto_path = args.output_file
    np.save(proto_path, protos)
    print(f">> Saved prototypes at '{proto_path}'")

    sys.exit()

M_graph = torch.zeros(num_nodes, num_nodes)
uG = nx.Graph(G)
for i, n in enumerate(all_nodes):
    for j, m in enumerate(all_nodes):
        l = nx.shortest_path_length(uG, source=n, target=m)
        M_graph[i,j] = l

print(f">> loading num_nodes = {num_nodes}")
if args.input_protos is None:
    unif_protos_path = f"./prototypes/prototypesuniform-{dim}d-{num_nodes}c.npy"
else:
    unif_protos_path = args.input_protos
print(f">> loading uniform protos from {unif_protos_path}")
protos = np.load(unif_protos_path)
protos = torch.from_numpy(protos).requires_grad_(True)
M_sphere = 1 - (protos @ protos.T)

p = q = np.full((num_nodes,), 1/num_nodes)
plan = ot.gromov_wasserstein(M_graph, M_sphere, torch.from_numpy(p) , torch.from_numpy(q))
# gw = ot.gromov_wasserstein2(M_graph, M_sphere, torch.from_numpy(p) , torch.from_numpy(q))
# gw.backward()


iprotos = [
    [
        all_nodes.index(c)
        for c in classes_at_level(G,lvl)
        if c in all_nodes
    ]
    for lvl in range(1,depth+1)
]

plan_index = plan.argmax(dim=-1).numpy()

ordered_protos = protos[plan_index, :].detach()

if args.save_all:
    mapping = hierarchical_node_mapping(G)
    print(mapping)
    I = [all_nodes.index(node) for node in nx.topological_sort(G)]
    protos = ordered_protos[I,:]
else:
    print("idx -> class mapping: ", dict(enumerate(classes_at_level(G,hierarchy_depth(G)))))
    protos = ordered_protos[iprotos[-1],:]

num_cls = protos.shape[0]
suffix = "ol" if args.only_leaves else ""
if args.output_file is None:
    proto_path = f"prototypesgromov{suffix}-{dim}d-{num_cls}c.npy"
else:
    proto_path = args.output_file
np.save(proto_path, protos)
print(f">> Saved prototypes at '{proto_path}'")
