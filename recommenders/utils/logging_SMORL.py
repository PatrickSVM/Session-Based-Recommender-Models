def get_logging_dict_train(
    train_sup_loss,
    train_q_loss,
    val_loss,
    topk_hr_ndcg,
    train_hr,
    train_ndcg,
    val_hr,
    val_ndcg,
    train_coverage_res,
    val_coverage_res,
    topk_cov,
    train_nov_rew,
    train_div_rew,
    val_nov_rew,
    val_div_rew,
    train_reps,
    val_reps,
    q_included=True,
    prefix="",
):
    """
    Get dictionary with all metrics for the evaluation
    metrics (for each respective k):
    - Supervised Train Loss
    - Q-Modification-Signal (Train)
    - Supervised Val Loss
    - HR@k
    - NDCG@k
    - Diversity (as CV@k)
    - Novelty (as CV@k)
    - Mean Diversity Reward
    - Mean Novelty Reward
    """
    epoch_log_res = {}

    # Save losses
    epoch_log_res["Supervised Train Loss"] = train_sup_loss

    if q_included:
        epoch_log_res["Q-Modification-Signal"] = train_q_loss

    epoch_log_res[f"{prefix+' '}Supervised Val Loss"] = val_loss

    # Save HR/NDCG
    for i, k in enumerate(topk_hr_ndcg):
        epoch_log_res[f"Train_HR@{k}"] = float(train_hr[i])
        epoch_log_res[f"Train_NDCG@{k}"] = float(train_ndcg[i])
        epoch_log_res[f"{prefix}Val_HR@{k}"] = float(val_hr[i])
        epoch_log_res[f"{prefix}Val_NDCG@{k}"] = float(val_ndcg[i])
        epoch_log_res[f"{prefix}Train_R@{k}"] = float(train_reps[i])
        epoch_log_res[f"{prefix}Val_R@{k}"] = float(val_reps[i])

    # Save Novelty/Diversity(-Coverage)
    for k in topk_cov:
        epoch_log_res[f"Train_NOV_CV@{k}"] = float(train_coverage_res[k][0])
        epoch_log_res[f"Train_DIV_CV@{k}"] = float(train_coverage_res[k][1])
        epoch_log_res[f"{prefix}Val_NOV_CV@{k}"] = float(val_coverage_res[k][0])
        epoch_log_res[f"{prefix}Val_DIV_CV@{k}"] = float(val_coverage_res[k][1])

    # Save Nov/Div rewards
    epoch_log_res["Train_Nov_Reward"] = float(train_nov_rew)
    epoch_log_res["Train_Div_Reward"] = float(train_div_rew)
    epoch_log_res[f"{prefix}Val_Nov_Reward"] = float(val_nov_rew)
    epoch_log_res[f"{prefix}Val_Div_Reward"] = float(val_div_rew)

    # If prefix: Second model is logged, only include val metrics
    if prefix != "":
        for key in list(epoch_log_res.keys()):
            if "Val" not in key:
                epoch_log_res.pop(key)

    return epoch_log_res


def get_logging_dict_test(
    test_loss,
    topk_hr_ndcg,
    test_hr,
    test_ndcg,
    test_coverage_res,
    topk_cov,
    test_nov_rew,
    test_div_rew,
    test_reps,
    real_test=False,
    prefix="",
):
    """
    Get dictionary with all metrics for the evaluation
    metrics (for each respective k):
    - Supervised Train Loss
    - Q-Modification-Signal (Train)
    - Supervised Val Loss
    - HR@k
    - NDCG@k
    - Diversity (as CV@k)
    - Novelty (as CV@k)
    - Mean Diversity Reward
    - Mean Novelty Reward

    Depending on whether it is a 'real' test or not, all
    metrics get prefix 'BestModelVal' or 'Test'.
    """
    name = "Test" if real_test else "Best_Val"

    test_log_res = {}

    # Save loss
    test_log_res[f"{prefix}{name} Loss"] = test_loss

    # Save HR/NDCG
    for i, k in enumerate(topk_hr_ndcg):
        test_log_res[f"{prefix}{name}_HR@{k}"] = float(test_hr[i])
        test_log_res[f"{prefix}{name}_NDCG@{k}"] = float(test_ndcg[i])
        test_log_res[f"{prefix}Val_R@{k}"] = float(test_reps[i])

    # Save Novelty/Diversity(-Coverage)
    for k in topk_cov:
        test_log_res[f"{prefix}{name}_NOV_CV@{k}"] = float(test_coverage_res[k][0])
        test_log_res[f"{prefix}{name}_DIV_CV@{k}"] = float(test_coverage_res[k][1])

    # Save Nov/Div rewards
    test_log_res[f"{prefix}{name}_Nov_Reward"] = float(test_nov_rew)
    test_log_res[f"{prefix}{name}_Div_Reward"] = float(test_div_rew)

    return test_log_res
