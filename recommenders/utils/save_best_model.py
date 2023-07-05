import torch


class SaveBestModel:
    """
    Class to save the best model while training by comparing
    the passed validation metric.

    If max=True, higher is better. If False, it is minimized.
    """

    def __init__(self, out_dir, max=True, metric_name=""):
        self.max = max
        self.best_valid_metric = 0
        self.out_dir = f"{out_dir}/best_model.pt"
        self.metric_name = metric_name

    def __call__(
        self,
        curr_valid_metric,
        epoch,
        model,
        model_idx=1,
    ):
        if max:
            if curr_valid_metric > self.best_valid_metric:
                self.best_valid_metric = curr_valid_metric

                torch.save(
                    {
                        "epoch": epoch,
                        "model_idx": model_idx,
                        "hidden_dim": model.hidden_dim,
                        "item_num": model.item_num,
                        "action_dim": model.action_dim,
                        "state_size": model.state_size,
                        "embedding_dim": model.embedding_dim,
                        "model_state_dict": model.state_dict(),
                    },
                    self.out_dir,
                )
