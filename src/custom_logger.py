from typing import Dict, Any, Mapping, List, Union

import matplotlib.figure
import wandb
from benchmarking import calc_metrics

# NOTE: On import, this global variable is set to None. Ensure you online import this module once or do not import this variable directly.
LOGGER = None


class CustomLogger:
    def __init__(self, use_wandb: bool, config: Dict[str, Any], sweep_config: Union[Dict[str, Any], None]=None, *args, **kwargs):
        self.sweep_config: Union[Dict[str, Any], None] = sweep_config
        if use_wandb:
            self._wandb_run = wandb.init(project="sensor-placement", entity="camb-mphil", config=config)
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

    def log(self, data: Dict[str, Any], step=None):
        if wandb.run is None:
            self._print_and_show_log_dict(data)
        else:
            # print("Preprocessing data for wandb")
            data = self._preprocess_log_dict_wandb(data)
            # print(data)
            wandb.log(data, step=step)

    def log_metrics(self, ground_truth_reshaped, mu_plot, std_plot, mu_unseen, std_unseen, ground_truth_unseen, num_low_fidelity_samples=None, num_high_fidelity_samples=None, cost=None, step=None):
        '''Log metrics to wandb and/or stdout'''
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
        ), step)
        if num_low_fidelity_samples is not None:
            self.log(dict(
                num_low_fidelity_samples = num_low_fidelity_samples,
                num_high_fidelity_samples = num_high_fidelity_samples,
                cost = cost,
            ), step)
