from typing import Dict, Any, Mapping, List

import matplotlib.figure
import wandb
from matplotlib import pyplot as plt

LOGGER = None


class CustomLogger:
    def __init__(self, use_wandb: bool, config: Dict[str, Any]):
        if use_wandb:
            wandb.init(project="sensor-placement", entity="camb-mphil", config=config)
            self.config = wandb.config
        else:
            self.config = config

    def stop_run(self):
        if wandb.run is not None:
            wandb.finish()

    def _preprocess_log_dict_wandb(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = data.copy()
        for key, val in data.items():
            if isinstance(val, Mapping):
                # doesn't handle case where dict keys are not strings, so just don't use that stuff
                data[key] = self._preprocess_log_dict(val)
            elif isinstance(val, matplotlib.figure.Figure):
                data[key] = wandb.Image(val)
        return data

    def _print_and_show_log_dict(self, data: Mapping, prefix: List[str] = []):
        for key, val in data.items():
            if isinstance(val, Mapping):
                print(" " * len(prefix) + key + ":")
                self._print_and_show_log_dict(val, prefix + [key])
            elif isinstance(val, matplotlib.figure.Figure):
                val.savefig("results/" + ".".join(prefix + [key]) + ".png")
                val.show()
            else:
                print(" " * len(prefix) + key + ":" + str(val))

    def log(self, data: Dict[str, Any]):
        if wandb.run is None:
            self._print_and_show_log_dict(data)
        else:
            wandb.log(self._preprocess_log_dict_wandb(data))
