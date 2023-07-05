import os
import pickle
import json
import numpy as np
import pandas as pd


def get_low_frequency_items(data_dir, quantile, item_col_name):
    # Read  data
    if "csv" in str(data_dir):
        data = pd.read_pickle(data_dir)
    elif "json" in str(data_dir):
        data = pd.read_json(data_dir, orient="records", lines=True)
    else:
        data = pd.read_pickle(data_dir)

    # Get sorted item frequency
    sorted_freq = data[item_col_name].value_counts().sort_values(ascending=False)
    freq_thresh = np.quantile(sorted_freq, q=quantile)

    # Get itemIDs with frequency smaller than thresh
    unpopular_items = sorted_freq[sorted_freq < freq_thresh].index.values

    return unpopular_items


def save_freq_to_file(data_dir, target_dir, quantile=0.9, item_col_name="item_id"):
    unpopular_items = get_low_frequency_items(
        data_dir, quantile=quantile, item_col_name=item_col_name
    )

    unpopular_items = unpopular_items.tolist()

    with open(os.path.join(target_dir, "unpopular_items.json"), "w") as f:
        json.dump(unpopular_items, f)


def load_unpopular_items(data_dir):
    """
    Loads list from directory and returns it as set.
    """
    if "pkl" in data_dir:
        with open(data_dir, "rb") as f:
            unpopular_items = pickle.load(f)
    else:
        with open(data_dir, "r") as f:
            unpopular_items = json.load(f)
    return set(unpopular_items)


if __name__ == "__main__":
    import os
    import pathlib
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Write unpopular items to file")

    parser.add_argument(
        "-f",
        "--filename",
        help="all events file path",
        metavar="FILE",
        required=True,
    )

    parser.add_argument(
        "-t",
        "--targetdir",
        help="target directory where to save file",
        required=True,
    )

    parser.add_argument(
        "-q",
        "--quantile",
        help="quantile to check - e.g. 0.9, 90% of all items included",
        required=False,
        type=float,
    )

    parser.add_argument(
        "-i",
        "--item_col_name",
        help="name of itemID column in data file",
        required=False,
    )

    # Get filepath as arg
    args = parser.parse_args()
    config_path = pathlib.Path(args.filename)
    save_freq_to_file(
        config_path.absolute(),
        quantile=args.quantile,
        target_dir=args.targetdir,
        item_col_name=args.item_col_name,
    )
