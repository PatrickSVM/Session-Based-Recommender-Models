def get_logging_dict_train(
    top_k, action_types_names, train_hit_ratio, train_ndcg, val_hit_ratio, val_ndcg
):
    """
    Get dictionary with all metrics for the evaluation
    metrics, for each action (incl. "total") and k:
    - HR@k-Action
    - NDCG@k-Action
    """
    epoch_log_res = {}
    for i, k in enumerate(top_k):
        for a_t in list(action_types_names) + ["total"]:
            type_desc = a_t.capitalize()
            epoch_log_res[f"Train_HR@{k}[{type_desc}]"] = float(train_hit_ratio[a_t][i])
            epoch_log_res[f"Train_NDCG@{k}[{type_desc}]"] = float(train_ndcg[a_t][i])
            epoch_log_res[f"Val_HR@{k}[{type_desc}]"] = float(val_hit_ratio[a_t][i])
            epoch_log_res[f"Val_NDCG@{k}[{type_desc}]"] = float(val_ndcg[a_t][i])

    return epoch_log_res


def get_logging_dict_test(
    top_k, action_types_names, test_hit_ratio, test_ndcg, real_test=False
):
    """
    Get dictionary with all metrics for the etestuation
    metrics, for each action (incl. "total") and k
    for the test set:
    - HR@k-Action
    - NDCG@k-Action
    """
    name = "Test" if real_test else "BestModelVal"
    test_res = {}
    for i, k in enumerate(top_k):
        for a_t in list(action_types_names) + ["total"]:
            type_desc = a_t.capitalize()
            test_res[f"{name}_HR@{k}[{type_desc}]"] = float(test_hit_ratio[a_t][i])
            test_res[f"{name}_NDCG@{k}[{type_desc}]"] = float(test_ndcg[a_t][i])

    return test_res
