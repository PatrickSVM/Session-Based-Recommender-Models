import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams


class CorrectedSummaryWriter(SummaryWriter):
    """
    Corrected summary writer for tensorboard, that does not create a new directory for every run's
    hyperparameters. Taken from https://github.com/pytorch/pytorch/issues/32651

    Usage is same as for normal SummaryWriter just that HPs run under same name as main run.
    """

    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError("hparam_dict and metric_dict should be dictionary.")
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            self.add_scalar(k, v)
