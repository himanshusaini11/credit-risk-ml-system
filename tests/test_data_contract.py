import pandas as pd

from creditrisk.data.load import load_csv
from creditrisk.data.schema import validate_dataframe


def test_data_schema_validation(tmp_path):
    df = pd.DataFrame(
        {
            "age": [25, 40, 35, 50],
            "income": [50000, 80000, 62000, 90000],
            "segment": ["A", "B", "A", "C"],
            "default": [0, 1, 0, 1],
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    loaded = load_csv(csv_path)
    schema = validate_dataframe(loaded, target="default")

    assert schema["n_rows"] == 4
    assert "age" in schema["features"]["numeric"]
    assert "segment" in schema["features"]["categorical"]
    assert schema["base_rate"] == 0.5
