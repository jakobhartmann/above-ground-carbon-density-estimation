from typing import Dict, Any, Mapping, List, Union

import matplotlib.figure
import wandb
from matplotlib import pyplot as plt

LOGGER = None


class CustomLogger:
    def __init__(self, use_wandb: bool, config: Dict[str, Any], sweep_config: Union[Dict[str, Any], None]=None):
        self.sweep_config: Union[Dict[str, Any], None] = sweep_config
        if use_wandb:
            # wandb.init(project="sensor-placement", entity="camb-mphil", config=config)
            self._wandb_run = wandb.init(project="test-sensor-placement", entity="sepand", config=config)
            print('wandb initialized')
            self.config = wandb.config
            self._wandb_instance = wandb
        else:
            self.config = config
        print('logger started')

    def stop_run(self):
        if wandb.run is not None:
            wandb.finish()

    def _preprocess_log_dict_wandb(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = data.copy()
        for key, val in data.items():
            if isinstance(val, Mapping):
                # doesn't handle case where dict keys are not strings, so just don't use that stuff
                data[key] = self._preprocess_log_dict_wandb(val)
            # elif isinstance(val, matplotlib.figure.Figure):
            #     data[key] = wandb.Image(val)
        return data

    def _print_and_show_log_dict(self, data: Mapping, prefix: List[str] = []):
        for key, val in data.items():
            if isinstance(val, Mapping):
                print(" " * len(prefix) + key + ":")
                self._print_and_show_log_dict(val, prefix + [key])
            elif isinstance(val, matplotlib.figure.Figure):
                val.savefig("results/" + ".".join(prefix + [key]) + ".png")
                print("Showing plots")
                val.show()
            else:
                print(" " * len(prefix) + key + ":" + str(val))

    def log(self, data: Dict[str, Any]):
        if wandb.run is None:
            self._print_and_show_log_dict(data)
        else:
            print("Preprocessing data for wandb")
            data = self._preprocess_log_dict_wandb(data)
            print(data)
            wandb.log(data)
