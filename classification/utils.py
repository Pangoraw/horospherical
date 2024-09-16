import os
import time
from collections import defaultdict
from pathlib import Path
import time
import json
import pandas as pd
import torch


class RunLogger:
    """
    Logs the run results to a simple json file, this replaces wandb logging.
    """

    def __init__(self, name, config):
        self.config = config
        self.name = name
        self.series = defaultdict(list)

    def log(self, data):
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                if len(v.shape) == 0:
                    v = v.item()
                else:
                    v = v.flatten().cpu().tolist()
            self.series[k].append(v)

    def finish(self):
        data = {}
        for k, v in self.series.items():
            if len(v) > 0:
                data["summary." + k] = v[-1]
        data.update({"config." + k: v for k, v in self.config.items()})
        data.update(self.series)
        data["name"] = self.name
        data["id"] = os.getenv("SLURM_JOBID", str(time.time_ns()))
        filename = f"results/runs_{self.name}.json"
        with open(filename, "w") as f:
            json.dump(data, f)
        print(f">> Saving run result to {filename}")


class Reporter:
    "Helper class to log run results"

    def __init__(self, args):
        self.args = args
        self.values = dict(epoch=[], split=[])
        self.folder = Path("results")

    def report(self, header, split, acc, epoch):
        if header not in self.values:
            self.values[header] = []

        self.values[header].append(acc)
        if header == "species":
            self.values["epoch"].append(epoch)
            self.values["split"].append(split)

    def finish(self):
        self.values.update(self.args.__dict__)
        df = pd.DataFrame(self.values)
        filename = os.path.join(
            *[f"{k}-{v}"
            for k, v in self.args.__dict__.items()]
        )
        filename = self.folder / filename / "results.csv"
        filename.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filename, index=False)
