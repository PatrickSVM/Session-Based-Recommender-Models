import numpy as np
import pandas as pd


def train_test_split(
    full_df, session_key="sessionID", ratio=[0.8, 0.1, 0.1], random_seed=118
):
    """
    Splits pandas df in train, val and test dataset according
    to provided key.

    The exact percentages will not be met precisely
    but approximately. 80% of session will be taken for the trian set
    but each session has different lengths.
    """
    if type(full_df) == str:
        full_df = pd.read_csv(full_df)

    all_sessions = full_df[session_key].unique()
    n_sessions = len(all_sessions)
    n_train = round(n_sessions * ratio[0])
    n_val = round(n_sessions * ratio[1])
    n_test = n_sessions - n_train - n_val

    np.random.seed(random_seed)
    np.random.shuffle(all_sessions)

    train = full_df[full_df[session_key].isin(all_sessions[:n_train])]
    val = full_df[full_df[session_key].isin(all_sessions[n_train : (n_train + n_val)])]
    test = full_df[full_df[session_key].isin(all_sessions[-n_test:])]

    total = len(train) + len(val) + len(test)
    total_bef = len(full_df)

    print(f"Dataset successfully split according to ratio {ratio}!\n")
    print(f"Train: {len(train)} ({len(train)/total*100:.0f}%)")
    print(f"Val:   {len(val)} ({len(val)/total*100:.0f}%)")
    print(f"Test:  {len(test)} ({len(test)/total*100:.0f}%)")
    print(f"Total: {total}")

    assert total == total_bef

    return train, val, test
