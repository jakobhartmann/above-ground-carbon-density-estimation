from typing import Dict, Any, Mapping, List, Union

import matplotlib.figure
import wandb
from benchmarking import calc_metrics

LOGGER = None


class CustomLogger:
    def __init__(self, use_wandb: bool, config: Dict[str, Any], sweep_config: Union[Dict[str, Any], None]=None, *args, **kwargs):
        # NOTE: this is a hack to get around the fact that wandb.init() can't be called twice in the same process
        self.sweep_config: Union[Dict[str, Any], None] = sweep_config
        if use_wandb:
            wandb.init(project="sensor-placement", entity="camb-mphil", config=config)
            # self._wandb_run = wandb.init(project="test-sensor-placement", entity="sepand", config=config)
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
                # val = plt.figure(val.number)
                # val.canvas.draw()
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

    def log_metrics(self, ground_truth_reshaped, mu_plot, std_plot, mu_unseen, std_unseen, ground_truth_unseen):
        L1, L2, MSE, PSNR, SSIM, MPDF_unseen, MPDF_all, KL, ModelVariance_unseen, ModelVariance_all = calc_metrics(mu_plot, std_plot, ground_truth_reshaped, mu_unseen, std_unseen, ground_truth_unseen)

        self.log(dict(
            L1 = L1,
            L2 = L2,
            MSE = MSE,
            PSNR = PSNR,
            SSIM = SSIM,
            MPDF_unseen = MPDF_unseen,
            MPDF_all = MPDF_all,
            KL = KL,
            ModelVariance_unseen = ModelVariance_unseen,
            ModelVariance_all = ModelVariance_all
        ))

