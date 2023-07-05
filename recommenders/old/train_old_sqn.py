from tqdm import tqdm
import wandb
import torch
import numpy as np
from recommenders.old.sqn_evaluation_old import (
    create_metric_dicts,
    update_train_metrics,
    evaluate,
)
from recommenders.old.sqn_evaluation_old import EvaluationDataset
from recommenders.old.replay_buffer import ReplayBuffer_ActTypes
from recommenders.SQN.sqn_gru import SQN
from torch.utils.data import DataLoader
from recommenders.utils.save_best_model import SaveBestModel
from recommenders.old.logging_old import get_logging_dict_train, get_logging_dict_test

# TODO: Sort dicts
# TODO: If GRU: init_gru() ...


def train_SQN(
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
    gamma,
    device,
    padding_pos,
    padding_id,
    train_pad_embed,
    use_packed_seq,
    top_k,
    action_types,
    action_types_dict,
    action_to_rew_dict,
    state_size,
    head_idx,
    best_model_metric="Val_NDCG@10[Click]",
    seed_torch=118,
    seed_python=999,
    use_wandb=False,
    use_tensorboard=False,
    tensorboard_writer_obj=None,
    tensorboard_hparams=None,
    progress_bar=False,
):
    """
    Trains SQN model with provided arguments.
    """

    disable_pro_bar = not progress_bar

    # Init replay buffer
    map_rew_cat = np.vectorize(lambda rew_cat: action_to_rew_dict[rew_cat])

    # Load training data
    train_set = ReplayBuffer_ActTypes(
        dir=train_dir,
        map_rew_cat_func=map_rew_cat,
        num_items=num_items,
        state_name="state",
        next_state_name="next_state",
        action_name="action",
        end_name="is_done",
        action_type_name="is_buy",
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
        action_to_reward_dict=action_to_rew_dict,
        action_type_name="is_buy",
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
            action_to_reward_dict=action_to_rew_dict,
            action_type_name="is_buy",
            session_id_name="session_id",
            action_name="item_id",
        )

        test_loader = DataLoader(
            test_set, batch_size=val_batch_size, shuffle=True, drop_last=False
        )

    # Init SQN-class with two models
    # TODO: GRU, Sasrec, change names
    sqn = SQN(
        hidden_dim=hidden_state_size,
        embedding_dim=embedding_size,
        train_pad_embed=train_pad_embed,
        use_packed_seq=use_packed_seq,
        learning_rate=lr,
        item_num=num_items,
        state_size=state_size,
        action_dim=num_items,
        gamma=gamma,
        device=device,
        action_types=action_types,
        action_types_dict=action_types_dict,
        torch_rand_seed=seed_torch,
        python_rand_seed=seed_python,
    )

    # Setup SaveBestModel with top-10-total-ndcg
    model_saver = SaveBestModel(
        out_dir=exp_dir, max=True, metric_name=best_model_metric
    )

    # Send to device
    sqn.send_to_device()

    # Call evaluation step to init memmory (not necessary but convenient)
    val_loss, val_hit_ratio, val_ndcg = evaluate(
        evaluation_data_loader=val_loader,
        model=sqn.DQN_1,
        device=sqn.device,
        action_types=sqn.action_types,
        action_types_dict=sqn.action_types_dict,
        loss_function=sqn.cross_entropy_loss,
        top_k=top_k,
        head_idx=head_idx,
    )

    train_sup_losses = np.zeros(epochs)
    train_q_losses = np.zeros(epochs)
    val_sup_loss = np.zeros(epochs)

    for epoch in range(epochs):
        # Set in training mode
        sqn.DQN_1.train()

        # Init loss stats for epoch
        epoch_sup_loss = 0
        epoch_q_loss = 0

        # Initialize result dicts for hr/ndcg/action-type-count for epoch
        (
            action_types_hit_dict_train,
            action_types_ndcg_dict_train,
            action_types_count_train,
        ) = create_metric_dicts(sqn.action_types, sqn.action_types_dict, top_k)

        for n_batch, (s, a, r, s_next, a_type, s_len, s_next_len, is_end) in enumerate(
            tqdm(
                train_loader,
                desc=f"Epoch {epoch}",
                unit="batch",
                disable=disable_pro_bar,
            )
        ):
            sqn.set_train()
            sup_loss, q_loss = sqn.train_step(
                s, a, r, s_next, s_len, s_next_len, is_end
            )

            epoch_sup_loss += sup_loss
            epoch_q_loss += q_loss

            # Metric computataction_types_hit_dict_train[action_types_diction - Training batch
            # todo: Turn off action_types here. Unnecessary.?
            (
                action_types_hit_dict_train,
                action_types_ndcg_dict_train,
                action_types_count_train,
            ) = update_train_metrics(
                s,
                a,
                a_type,
                s_len,
                sqn.DQN_1,
                sqn.device,
                action_types_hit_dict_train,
                action_types_ndcg_dict_train,
                action_types_count_train,
                sqn.action_types,
                sqn.action_types_dict,
                top_k,
                head_idx,
            )

            # if not n_batch % 100:
            # print(f"Batch {n_batch}: {sup_loss:.2f} | {q_loss:.4f}")

        # Training losses
        train_sup_losses[epoch] = epoch_sup_loss / len(train_loader)
        train_q_losses[epoch] = epoch_q_loss / len(train_loader)

        # Calculate train hit ratio
        for key, val in action_types_hit_dict_train.items():
            action_types_hit_dict_train[key] = val / action_types_count_train[key]

        # Calculate train ndcg ratio
        for key, val in action_types_ndcg_dict_train.items():
            action_types_ndcg_dict_train[key] = val / action_types_count_train[key]

        # Calculate validation metrics
        val_loss, val_hit_ratio, val_ndcg = evaluate(
            evaluation_data_loader=val_loader,
            model=sqn.DQN_1,
            device=sqn.device,
            action_types=sqn.action_types,
            action_types_dict=sqn.action_types_dict,
            loss_function=sqn.cross_entropy_loss,
            top_k=top_k,
            head_idx=head_idx,
        )

        val_sup_loss[epoch] = val_loss

        # Prepare logging
        epoch_log_res = get_logging_dict_train(
            top_k=top_k,
            action_types_names=list(action_types_dict.values()),
            train_hit_ratio=action_types_hit_dict_train,
            train_ndcg=action_types_ndcg_dict_train,
            val_hit_ratio=val_hit_ratio,
            val_ndcg=val_ndcg,
        )

        epoch_log_res["Supervised Train Loss"] = train_sup_losses[epoch]
        epoch_log_res["Q-Modification-Signal"] = train_q_losses[epoch]
        epoch_log_res["Supervised Val Loss"] = val_loss

        # Save model if improvement on val-metric
        model_saver(
            epoch_log_res[model_saver.metric_name], epoch=epoch, model=sqn.DQN_1
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
        # print(f"\nTraining hr/ndcg:")
        # print(action_types_hit_dict_train)
        # print(action_types_ndcg_dict_train)
        # print(f"\Validation hr/ndcg:")
        # print(val_hit_ratio)
        # print(val_ndcg)

    # Load best model
    best_model_vals = torch.load(model_saver.out_dir)
    best_epoch = best_model_vals["epoch"]
    sqn.DQN_1.load_state_dict(best_model_vals["model_state_dict"])

    print(f"\nLoaded epoch for testing: {best_epoch}")

    # Calculate test metrics (on specified test set - could be val too)
    test_loss, test_hit_ratio, test_ndcg = evaluate(
        evaluation_data_loader=test_loader,
        model=sqn.DQN_1,
        device=sqn.device,
        action_types=sqn.action_types,
        action_types_dict=sqn.action_types_dict,
        loss_function=sqn.cross_entropy_loss,
        top_k=top_k,
        head_idx=head_idx,
    )

    print("\nTest-Metrics:\n")
    print(test_loss)
    print(test_hit_ratio)
    print(test_ndcg)

    # Write test metrics to logging dict
    test_metrics = get_logging_dict_test(
        top_k=top_k,
        action_types_names=list(action_types_dict.values()),
        test_hit_ratio=test_hit_ratio,
        test_ndcg=test_ndcg,
        real_test=real_test,
    )

    # Log Hyperparams and test metrics of run to tensorboard
    if use_tensorboard:
        tensorboard_writer_obj.add_hparams(tensorboard_hparams, test_metrics)

    if use_wandb:
        wandb.log(test_metrics)
