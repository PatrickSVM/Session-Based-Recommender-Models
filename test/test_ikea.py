import pandas as pd
from recommenders.ikea.data_utils.preprocessing import mark_last_inspiration_click


def test_mark_last_inspiration_click():
    example_df = {
        "sessionID": {
            0: "0_33749",
            1: "0_33749",
            2: "0_33749",
            3: "0_33749",
            4: "0_33749",
        },
        "item_id": {
            0: "20529632",
            1: "a8d6308a-be66-4140-982caf0d9308b685",
            2: "a5a00142-a2d8-4c88-8af61486f4110d24",
            3: "00529671",
            4: "80534372",
        },
        "action_type": {
            0: "view_item",
            1: "click_inspiration",
            2: "click_inspiration",
            3: "view_item",
            4: "view_item",
        },
        "market": {0: "qa", 1: "qa", 2: "qa", 3: "qa", 4: "qa"},
        "reward": {0: 0.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0},
    }

    true_last_col = [False, False, True, False, False]
    example_df = pd.DataFrame(example_df)

    res = example_df.groupby("sessionID", group_keys=False).apply(
        mark_last_inspiration_click
    )

    assert (res.values == true_last_col).all()
