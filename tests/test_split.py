import pandas as pd

from creditrisk.config import SplitConfig
from creditrisk.data.split import split_data


def test_split_reproducibility():
    df = pd.DataFrame(
        {
            "id": list(range(100)),
            "feature": list(range(100)),
            "default": [0, 1] * 50,
        }
    )
    split_cfg = SplitConfig(
        method="stratified", seed=123, train_frac=0.7, val_frac=0.15, test_frac=0.15
    )
    train_a, val_a, test_a = split_data(df, split_cfg, target="default")
    train_b, val_b, test_b = split_data(df, split_cfg, target="default")

    assert set(train_a["id"]) == set(train_b["id"])
    assert set(val_a["id"]) == set(val_b["id"])
    assert set(test_a["id"]) == set(test_b["id"])
