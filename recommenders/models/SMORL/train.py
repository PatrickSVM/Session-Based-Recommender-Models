from tqdm import tqdm
import wandb
import torch
import numpy as np
from recommenders.evaluate.eval_protocol import evaluate, update_train_metrics
from recommenders.evaluate.eval_dataset import EvaluationDataset
from recommenders.data_utils.replay_buffer import ReplayBuffer
from recommenders.data_utils.item_frequency import load_unpopular_items
from recommenders.evaluate.coverage import get_coverage
from recommenders.models.SMORL.smorl_gru import SMORL_GRU
from torch.utils.data import DataLoader
from recommenders.utils.save_best_model import SaveBestModel
from recommenders.utils.logging_SMORL import (
    get_logging_dict_train,
    get_logging_dict_test,
    get_logging_dict_test_second,
)


def train_SMORL(
    exp_dir,
    train_dir,
    val_dir,
    test_dir,
    num_items,
    batch_size,
    val_batch_size,
    epochs,
    lr,
    gru_layers,
    embedding_size,
    hidden_state_size,
    gamma,
    alpha,
    device,
    padding_pos,
    padding_id,
    train_pad_embed,
    use_packed_seq,
    head_idx,
    state_size,
    q_weights,
    div_emb_dir,
    unpopular_actions_path,
    topk_hr_ndcg,
    topk_cov,
    topk_div,
    topk_nov,
    nov_rew_sig=1,
    best_model_metric="Val_NDCG@10",
    seed_torch=118,
    seed_python=999,
    use_wandb=False,
    use_tensorboard=False,
    tensorboard_writer_obj=None,
    tensorboard_hparams=None,
    progress_bar=False,
):
    """
    Trains SMORL model with provided arguments.
    """

    disable_pro_bar = not progress_bar

    # Load training data
    train_set = ReplayBuffer(
        dir=train_dir,
        num_items=num_items,
        state_name="state",
        next_state_name="next_state",
        action_name="action",
        end_name="is_end",
        state_len_name="true_state_len",
        true_next_state_len_name="true_next_state_len",
        reward_name="r_act",
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=False
    )

    # Load validation data
    val_set = EvaluationDataset(
        dir=val_dir,
        padding_id=padding_id,
        pad_pos=padding_pos,
        state_len=state_size,
        session_id_name="session_id",
        action_name="item_id",
    )

    val_loader = DataLoader(
        val_set, batch_size=val_batch_size, shuffle=True, drop_last=False
    )

    # Load test set - if val_dir=test_dir, set it to val_loader
    real_test = val_dir != test_dir
    if not real_test:
        test_loader = val_loader

    else:
        # Test best model on test set
        test_set = EvaluationDataset(
            dir=test_dir,
            padding_id=padding_id,
            pad_pos=padding_pos,
            state_len=state_size,
            session_id_name="session_id",
            action_name="item_id",
        )

        test_loader = DataLoader(
            test_set, batch_size=val_batch_size, shuffle=True, drop_last=False
        )

    # Load embedding layer for divergence comp
    div_emb = torch.nn.Embedding.from_pretrained(torch.load(div_emb_dir), freeze=True)
    div_emb.to(device)

    # Load unpopular action set
    unpopular_actions_set = load_unpopular_items(data_dir=unpopular_actions_path)

    # Init SQN-class with two models
    smorl = SMORL_GRU(
        hidden_dim=hidden_state_size,
        embedding_dim=embedding_size,
        padding_pos=padding_pos,
        train_pad_embed=train_pad_embed,
        use_packed_seq=use_packed_seq,
        learning_rate=lr,
        item_num=num_items,
        state_size=state_size,
        action_dim=num_items,
        gamma=gamma,
        gru_layers=gru_layers,
        q_weights=q_weights,
        alpha=alpha,
        div_embedding=div_emb,
        unpopular_actions_set=unpopular_actions_set,
        topk_div=topk_div,
        topk_nov=topk_nov,
        nov_rew_sig=nov_rew_sig,
        device=device,
        padding_idx=padding_id,
        torch_rand_seed=seed_torch,
        python_rand_seed=seed_python,
    )

    # Setup SaveBestModel with top-10-total-ndcg
    model_saver = SaveBestModel(
        out_dir=exp_dir, max=True, metric_name=best_model_metric
    )

    # Send to device
    smorl.send_to_device()

    # Call evaluation step to init memmory (not necessary but convenient)
    # TODO: Same for train with one batch
    val_loss, val_hr, val_ndcg, val_cov, val_r_div, val_r_nov = evaluate(
        evaluation_data_loader=val_loader,
        model=smorl.SMORL_1,
        device=smorl.device,
        loss_function=smorl.cross_entropy_loss,
        padding_pos=padding_pos,
        diversity_embedding=div_emb,
        unpopular_actions_set=unpopular_actions_set,
        head_idx=head_idx,
        topk_hr_ndcg=topk_hr_ndcg,
        topk_to_consider_cov=topk_cov,
        topk_to_consider_div=topk_div,
        topk_to_consider_nov=topk_nov,
        novelty_rew_signal=nov_rew_sig,
    )

    train_sup_losses = np.zeros(epochs)
    train_q_losses = np.zeros(epochs)
    val_sup_loss = np.zeros(epochs)

    for epoch in range(epochs):
        # Init loss stats for epoch
        epoch_sup_loss = 0
        epoch_q_loss = 0

        # Initialize result arrays for total-hr/ndcg counts
        train_hr = np.zeros(shape=len(topk_hr_ndcg))
        train_ndcg = np.zeros(shape=len(topk_hr_ndcg))
        n_total_samples = 0

        # Init online reward total for nov/div
        train_nov_rew = 0
        train_div_rew = 0

        # Init dict of sets of covered actions for each k
        actions_covered_topk_dict = {k: set() for k in topk_cov}

        for n_batch, (s, a, r, s_next, s_len, s_next_len, is_end) in enumerate(
            tqdm(
                train_loader,
                desc=f"Epoch {epoch}",
                unit="batch",
                disable=disable_pro_bar,
            )
        ):
            # Set in training mode
            smorl.set_train()
            assert smorl.SMORL_1.training

            sup_loss, q_loss = smorl.train_step(
                s, a, r, s_next, s_len, s_next_len, is_end
            )

            epoch_sup_loss += sup_loss
            epoch_q_loss += q_loss

            # Get metric update for training batch
            (
                hr_batch,
                ndcg_batch,
                upd_actions_covered_topk_dict,
                batch_div_rew,
                batch_nov_rew,
            ) = update_train_metrics(
                s=s,
                a=a,
                s_len=s_len,
                model=smorl.SMORL_1,
                device=smorl.device,
                padding_pos=padding_pos,
                diversity_embedding=div_emb,
                unpopular_actions_set=unpopular_actions_set,
                actions_covered_topk_dict=actions_covered_topk_dict,
                head_idx=head_idx,
                topk_hr_ndcg=topk_hr_ndcg,
                topk_to_consider_cov=topk_cov,
                topk_to_consider_div=topk_div,
                topk_to_consider_nov=topk_nov,
                novelty_rew_signal=nov_rew_sig,
            )

            # Update metrics
            train_hr += hr_batch
            train_ndcg += ndcg_batch
            n_total_samples += len(a)
            train_div_rew += batch_div_rew
            train_nov_rew += batch_nov_rew
            actions_covered_topk_dict = upd_actions_covered_topk_dict

            # if not n_batch % 100:
            # print(f"Batch {n_batch}: {sup_loss:.2f} | {q_loss:.4f}")

        # Training losses - sum of avg per batch to avg per exmp
        train_sup_losses[epoch] = epoch_sup_loss / len(train_loader)
        train_q_losses[epoch] = epoch_q_loss / len(train_loader)

        # Calculate ratios of HR/NDCG - total sum to avg per exmp
        train_hr = train_hr / n_total_samples
        train_ndcg = train_ndcg / n_total_samples

        # Calculate per sample reward averages
        train_div_rew = train_div_rew / n_total_samples
        train_nov_rew = train_nov_rew / n_total_samples

        # Calculate coverage for epoch - k-dict with (nov, div)
        train_cov = get_coverage(
            actions_covered_topk_dict=actions_covered_topk_dict,
            unpopular_set=unpopular_actions_set,
            num_actions=smorl.SMORL_1.action_dim,
            topk=topk_cov,
        )

        # Calculate validation metrics
        val_loss, val_hr, val_ndcg, val_cov, val_r_div, val_r_nov = evaluate(
            evaluation_data_loader=val_loader,
            model=smorl.SMORL_1,
            device=smorl.device,
            loss_function=smorl.cross_entropy_loss,
            padding_pos=padding_pos,
            diversity_embedding=div_emb,
            unpopular_actions_set=unpopular_actions_set,
            head_idx=head_idx,
            topk_hr_ndcg=topk_hr_ndcg,
            topk_to_consider_cov=topk_cov,
            topk_to_consider_div=topk_div,
            topk_to_consider_nov=topk_nov,
            novelty_rew_signal=nov_rew_sig,
        )

        val_sup_loss[epoch] = val_loss

        # Prepare logging
        epoch_log_res = get_logging_dict_train(
            train_sup_loss=train_sup_losses[epoch],
            train_q_loss=train_q_losses[epoch],
            val_loss=val_loss,
            topk_hr_ndcg=topk_hr_ndcg,
            train_hr=train_hr,
            train_ndcg=train_ndcg,
            val_hr=val_hr,
            val_ndcg=val_ndcg,
            train_coverage_res=train_cov,
            val_coverage_res=val_cov,
            topk_cov=topk_cov,
            train_nov_rew=train_nov_rew,
            train_div_rew=train_div_rew,
            val_nov_rew=val_r_nov,
            val_div_rew=val_r_div,
        )

        # Save model if improvement on val-metric
        model_saver(
            epoch_log_res[model_saver.metric_name], epoch=epoch, model=smorl.SMORL_1
        )

        # Logging
        if use_wandb:
            wandb.log(epoch_log_res)
        if use_tensorboard:
            for key, val in epoch_log_res.items():
                tensorboard_writer_obj.add_scalar(key, val, epoch)

        print(
            f"Epoch {epoch}: TrainSup {train_sup_losses[epoch]:.2f} | TrainQ {train_q_losses[epoch]:.2f} | ValSup {val_sup_loss[epoch]:.2f}\n"
        )

    # Load best model
    best_model_vals = torch.load(model_saver.out_dir)
    best_epoch = best_model_vals["epoch"]
    smorl.SMORL_1.load_state_dict(best_model_vals["model_state_dict"])

    print(f"\nLoaded epoch for testing: {best_epoch}")

    # Calculate test metrics (on specified test set - could be val too)
    # First model
    test_loss, test_hr, test_ndcg, test_cov, test_r_div, test_r_nov = evaluate(
        evaluation_data_loader=test_loader,
        model=smorl.SMORL_1,
        device=smorl.device,
        loss_function=smorl.cross_entropy_loss,
        padding_pos=padding_pos,
        diversity_embedding=div_emb,
        unpopular_actions_set=unpopular_actions_set,
        head_idx=head_idx,
        topk_hr_ndcg=topk_hr_ndcg,
        topk_to_consider_cov=topk_cov,
        topk_to_consider_div=topk_div,
        topk_to_consider_nov=topk_nov,
        novelty_rew_signal=nov_rew_sig,
    )

    # Second model
    (
        test_loss_2,
        test_hr_2,
        test_ndcg_2,
        test_cov_2,
        test_r_div_2,
        test_r_nov_2,
    ) = evaluate(
        evaluation_data_loader=test_loader,
        model=smorl.SMORL_2,
        device=smorl.device,
        loss_function=smorl.cross_entropy_loss,
        padding_pos=padding_pos,
        diversity_embedding=div_emb,
        unpopular_actions_set=unpopular_actions_set,
        head_idx=head_idx,
        topk_hr_ndcg=topk_hr_ndcg,
        topk_to_consider_cov=topk_cov,
        topk_to_consider_div=topk_div,
        topk_to_consider_nov=topk_nov,
        novelty_rew_signal=nov_rew_sig,
    )

    # Compare on NDCG:
    if test_ndcg[0] < test_ndcg_2[0]:
        # Second model is best
        best_model_idx = 2

        # Write best test metrics to logging dict
        test_metrics = get_logging_dict_test(
            test_loss=test_loss_2,
            topk_hr_ndcg=topk_hr_ndcg,
            test_hr=test_hr_2,
            test_ndcg=test_ndcg_2,
            test_coverage_res=test_cov_2,
            topk_cov=topk_cov,
            test_nov_rew=test_r_nov_2,
            test_div_rew=test_r_div_2,
            real_test=real_test,
        )

        # Write metrics of first model too
        test_metrics_second = get_logging_dict_test_second(
            test_loss=test_loss,
            topk_hr_ndcg=topk_hr_ndcg,
            test_hr=test_hr,
            test_ndcg=test_ndcg,
            test_coverage_res=test_cov,
            topk_cov=topk_cov,
            test_nov_rew=test_r_nov,
            test_div_rew=test_r_div,
            real_test=real_test,
        )

    else:
        # Fist model is best
        best_model_idx = 1

        # Write best test metrics to logging dict
        test_metrics = get_logging_dict_test(
            test_loss=test_loss,
            topk_hr_ndcg=topk_hr_ndcg,
            test_hr=test_hr,
            test_ndcg=test_ndcg,
            test_coverage_res=test_cov,
            topk_cov=topk_cov,
            test_nov_rew=test_r_nov,
            test_div_rew=test_r_div,
            real_test=real_test,
        )

        # Write test metrics of second model too
        test_metrics_second = get_logging_dict_test_second(
            test_loss=test_loss_2,
            topk_hr_ndcg=topk_hr_ndcg,
            test_hr=test_hr_2,
            test_ndcg=test_ndcg_2,
            test_coverage_res=test_cov_2,
            topk_cov=topk_cov,
            test_nov_rew=test_r_nov_2,
            test_div_rew=test_r_div_2,
            real_test=real_test,
        )

    test_metrics_second["BestModelIdx"] = best_model_idx

    # Merge to one dict
    all_test_logs = dict(**test_metrics, **test_metrics_second)

    # Log Hyperparams and test metrics of run to tensorboard
    if use_tensorboard:
        tensorboard_writer_obj.add_hparams(tensorboard_hparams, all_test_logs)

    if use_wandb:
        wandb.log(all_test_logs)
