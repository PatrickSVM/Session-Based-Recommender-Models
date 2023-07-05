from tqdm import tqdm
import wandb
import torch
import numpy as np

from recommenders.evaluate.eval_protocol import evaluate, update_train_metrics
from recommenders.evaluate.eval_dataset import EvaluationDataset
from recommenders.data_utils.replay_buffer import ReplayBuffer
from recommenders.data_utils.item_frequency import load_unpopular_items
from recommenders.evaluate.coverage import get_coverage
from recommenders.models.GRU4Rec.model import GRU4Rec_trainer
from torch.utils.data import DataLoader
from recommenders.utils.save_best_model import SaveBestModel
from recommenders.utils.logging_SMORL import (
    get_logging_dict_train,
    get_logging_dict_test,
)


def train_GRU4Rec(
    exp_dir,
    train_dir,
    val_dir,
    test_dir,
    num_items,
    batch_size,
    val_batch_size,
    epochs,
    lr,
    embedding_size,
    hidden_state_size,
    device,
    padding_pos,
    padding_id,
    train_pad_embed,
    use_packed_seq,
    state_size,
    gru_layers,
    div_emb_dir,
    unpopular_actions_path,
    topk_cov,
    topk_div,
    topk_hr_ndcg,
    topk_nov,
    nov_rew_sig,
    best_model_metric="Val_NDCG@10[Click]",
    seed_torch=118,
    seed_python=999,
    save_pretrained_embeddings=False,
    use_wandb=False,
    use_tensorboard=False,
    tensorboard_writer_obj=None,
    tensorboard_hparams=None,
    progress_bar=False,
):
    """
    Trains GRU4Rec model with provided arguments.
    Also used to pretrain embedding layer for diversity rewards
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

    # Init GRU4Rec-class with two models
    gru = GRU4Rec_trainer(
        hidden_dim=hidden_state_size,
        embedding_dim=embedding_size,
        train_pad_embed=train_pad_embed,
        use_packed_seq=use_packed_seq,
        learning_rate=lr,
        item_num=num_items,
        state_size=state_size,
        action_dim=num_items,
        gru_layers=gru_layers,
        device=device,
        padding_idx=padding_id,
        torch_rand_seed=seed_torch,
        python_rand_seed=seed_python,
    )

    # Setup SaveBestModel
    model_saver = SaveBestModel(
        out_dir=exp_dir, max=True, metric_name=best_model_metric
    )

    # Send to device
    gru.send_to_device()

    # Call evaluation step to init memmory (not necessary but convenient)
    val_loss, val_hr, val_ndcg, val_cov, val_r_div, val_r_nov = evaluate(
        evaluation_data_loader=val_loader,
        model=gru.gru_model,
        device=gru.device,
        loss_function=gru.cross_entropy_loss,
        padding_pos=padding_pos,
        diversity_embedding=div_emb,
        unpopular_actions_set=unpopular_actions_set,
        head_idx=0,
        topk_hr_ndcg=topk_hr_ndcg,
        topk_to_consider_cov=topk_cov,
        topk_to_consider_div=topk_div,
        topk_to_consider_nov=topk_nov,
        novelty_rew_signal=nov_rew_sig,
    )

    train_sup_losses = np.zeros(epochs)
    val_sup_loss = np.zeros(epochs)

    for epoch in range(epochs):
        # Set in training mode
        gru.set_train()

        # Init loss stats for epoch
        epoch_sup_loss = 0

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
            gru.set_train()

            sup_loss = gru.train_step(s, a, s_len)

            epoch_sup_loss += sup_loss

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
                model=gru.gru_model,
                device=gru.device,
                padding_pos=padding_pos,
                diversity_embedding=div_emb,
                unpopular_actions_set=unpopular_actions_set,
                actions_covered_topk_dict=actions_covered_topk_dict,
                head_idx=0,
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
            num_actions=gru.gru_model.action_dim,
            topk=topk_cov,
        )

        # Calculate validation metrics
        val_loss, val_hr, val_ndcg, val_cov, val_r_div, val_r_nov = evaluate(
            evaluation_data_loader=val_loader,
            model=gru.gru_model,
            device=gru.device,
            loss_function=gru.cross_entropy_loss,
            padding_pos=padding_pos,
            diversity_embedding=div_emb,
            unpopular_actions_set=unpopular_actions_set,
            head_idx=0,
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
            train_q_loss=None,
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
            q_included=False,
        )

        # Save model if improvement on val-metric
        model_saver(
            epoch_log_res[model_saver.metric_name], epoch=epoch, model=gru.gru_model
        )

        # Logging
        if use_wandb:
            wandb.log(epoch_log_res)
        if use_tensorboard:
            for key, val in epoch_log_res.items():
                tensorboard_writer_obj.add_scalar(key, val, epoch)

        print(
            f"Epoch {epoch}: TrainSup {train_sup_losses[epoch]:.2f} | ValSup {val_sup_loss[epoch]:.2f}\n"
        )

    # Load best model
    best_model_vals = torch.load(model_saver.out_dir)
    best_epoch = best_model_vals["epoch"]
    gru.gru_model.load_state_dict(best_model_vals["model_state_dict"])

    print(f"\nLoaded epoch for testing: {best_epoch}")

    # Calculate test metrics (on specified test set - could be val too)
    test_loss, test_hr, test_ndcg, test_cov, test_r_div, test_r_nov = evaluate(
        evaluation_data_loader=test_loader,
        model=gru.gru_model,
        device=gru.device,
        loss_function=gru.cross_entropy_loss,
        padding_pos=padding_pos,
        diversity_embedding=div_emb,
        unpopular_actions_set=unpopular_actions_set,
        head_idx=0,
        topk_hr_ndcg=topk_hr_ndcg,
        topk_to_consider_cov=topk_cov,
        topk_to_consider_div=topk_div,
        topk_to_consider_nov=topk_nov,
        novelty_rew_signal=nov_rew_sig,
    )

    print("\nTest-Metrics:\n")
    print(test_loss)
    print(test_hr)
    print(test_ndcg)

    # Write test metrics to logging dict
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

    # Log Hyperparams and test metrics of run to tensorboard
    if use_tensorboard:
        tensorboard_writer_obj.add_hparams(tensorboard_hparams, test_metrics)

    if use_wandb:
        wandb.log(test_metrics)

    if save_pretrained_embeddings:
        torch.save(
            gru.gru_model.state_dict()["embedding.weight"],
            f=f"{exp_dir}/embedding_weights.pt",
        )
