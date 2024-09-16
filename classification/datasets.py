import os
from typing import Tuple
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as FT
from torchvision.datasets import ImageFolder

CUB_MEAN = [0.485, 0.456, 0.406]
CUB_STD = [0.229, 0.224, 0.225]

TRAIN_TEST_SPLIT = 1.0


class HoldoutDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, train: bool = True, train_test_split: float = TRAIN_TEST_SPLIT):
        super().__init__()
        self.train = train
        self.dataset = dataset
        perm_file = (
            "assets/" + type(dataset).__name__  +
            "_" +
            ("train" if train else "test") +
            "_" +
            str(len(dataset)) +
            "_" +
            str(TRAIN_TEST_SPLIT) + ".npy"
        )

        if not os.path.exists(perm_file):
            print(f">> Creating perm file at {perm_file}")
            perm = torch.randperm(len(dataset)).cpu().numpy()
            np.save(perm_file, perm)

        self.perm = np.load(perm_file)
        assert len(self.perm) == len(dataset)

        assert 0.0 <= train_test_split <= 1.0

        n_train_samples = int(train_test_split * len(dataset))
        if not train:
            self.perm = self.perm[n_train_samples:]
        else:
            self.perm = self.perm[:n_train_samples]

    def __getattr__(self, attr):
        return getattr(self.dataset, attr)

    def __len__(self):
        return len(self.perm)

    def __getitem__(self, idx):
        return self.dataset[self.perm[idx]]


def identity(x):
    return x


def load_cub(root, train=False, transform=None, target_transform=None):
    return ImageFolder(
        Path(root) / ("train" if train else "test"),
        transform=transform,
        target_transform=target_transform,
    )


def smart_crop(img: Image, bb: Tuple[int,int,int,int]) -> Image:
    """
    Crop that uses the largest size to already be square. Falls back to the
    bounding box if this goes out of size.
    Parameters
    ==========
        img: Image - the image to crop.
        bb: Tuple[int,int,int,int] - the bounding box in XYWH format.
    Returns
    =======
        cropped: Image - the cropped image.
    """
    x, y, w, h = bb

    W, H = img.size

    sidew = sideh = max(w, h)

    left_offset = max((h - w) // 2, 0)
    top_offset = max((w - h) // 2, 0)

    top = y - top_offset
    left = x - left_offset

    if top + sideh > H:
        top = y
        sideh = h

    if left + sidew > W:
        left = x
        sidew = w

    return FT.crop(img, top, left, sideh, sidew)


class CUBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform=identity,
        target_transform=identity,
        crop_to_bb: bool = False,
        use_smart_crop: bool = True,
    ):
        super().__init__()
        self.root = Path(root)
        self.train = train

        self.files = ["not_a_valid_sample.png"]

        with open(self.root / "images.txt") as f:
            lines = f.readlines()

        with open(self.root / "train_test_split.txt") as f:
            tts_lines = f.readlines()

        with open(self.root / "bounding_boxes.txt") as f:
            bb_lines = f.readlines()

        def parse_bb(line):
            n, x, y, w, h = line.strip().split()
            x, y, w, h = float(x), float(y), float(w), float(h)
            return int(n), (int(x), int(y), int(w), int(h))

        self.use_smart_crop = use_smart_crop
        self.crop_to_bb = crop_to_bb
        self.bbs = {
            n: bb
            for n, bb in map(parse_bb, bb_lines)
        }

        with open(self.root / "classes.txt") as f:
            class_lines = f.readlines()

        def map_cls_line(line):
            i, cls = line.strip().split(" ")
            return int(i), cls

        self.class_to_idx = {
            cls: i
            for i, cls in map(map_cls_line, class_lines)
        }

        train_test_splits = [not train] # true for train
        for line, tts_line in zip(lines, tts_lines):
            num, filename = line.split(" ")
            num2, train_test = tts_line.split(" ")
            assert num == num2
            assert len(self.files) == int(num) == len(train_test_splits)
            train_test_splits.append(train_test.strip() == "1")
            self.files.append(filename.strip())

        self.samples = [
            i
            for i, v in enumerate(train_test_splits)
            if (v and train) or (not v and not train) ]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        index = self.samples[index]
        filename = self.root / "images" / self.files[index]
        img = Image.open(filename)

        if img.mode != "RGB": img = img.convert("RGB")

        if self.crop_to_bb:
            if self.use_smart_crop:
                img = smart_crop(img, self.bbs[index])
            else:
                x, y, w, h = self.bbs[index]
                img = FT.crop(img, y, x, h, w)

        label = self.class_to_idx[filename.parent.name] - 1

        return self.transform(img), self.target_transform(label)


class SubsetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, perm):
        super().__init__()
        self.perm = perm
        self.dataset = dataset
    def __len__(self):
        return len(self.perm)
    def __getitem__(self, index):
        return self.dataset[self.perm[index]]

