import shutil
from pathlib import Path

import pandas as pd

from .utils import list_to_csv, strip


def join_txt_dataset(
    dataset1_file: str, dataset2_file: str, output_csv: str, filters: list
):
    """
    Join txt to create a dataset (csv file), possible to add filters apply to the
    join dataset

    """

    if not Path(dataset1_file).exists() or not Path(dataset2_file).exists():
        return

    with open(dataset1_file, "r") as file1, open(dataset2_file, "r") as file2:

        lst1 = map(strip, file1.readlines())
        lst2 = map(strip, file2.readlines())

        # remove duplicate element
        s1 = set(lst1)
        s2 = set(lst2)
        mult_set = s1 | s2

        res = sorted(mult_set)

        # filter on the dataset the keep only interesting files
        for f in filters:
            res = list(filter(f, lst1))

        return list_to_csv(res, "SOURCES", output_csv)


def create_dataset(
    input_csv: str,
    output_dir: str,
    output_csv: str | None = None,
    quantity: float = 1.0,
) -> pd.DataFrame:
    """
    params:
    - Quantity: Quantity (in normalize pourcentage) of the dataset filename (input file)
    takes to create the real dataset.
    Create a csv of the subdataset if the quantity < 1
    """

    assert quantity > 0
    assert len(output_dir) > 0 and len(input_csv) > 0

    filename = Path(input_csv)
    output = Path(output_dir)

    df: pd.DataFrame = pd.read_csv(filename)
    n_rows = 0

    if quantity <= 1:
        n_rows = int(len(df) * quantity)
        n_rows += n_rows % 2
    else:
        n_rows = int(quantity)

    df = df.iloc[:n_rows]

    left_path = output / "left"
    right_path = output / "right"

    left_path.mkdir(parents=True, exist_ok=True)
    right_path.mkdir(parents=True, exist_ok=True)

    left_df: pd.DataFrame = df.loc[df["SIDE"] == "L"]
    right_df: pd.DataFrame = df.loc[df["SIDE"] == "R"]

    def _create_class(class_df: pd.DataFrame, path: Path):
        for _, row in class_df.iterrows():

            source = row["SOURCE"]
            name = row["NAME"]

            source = source.strip('"')
            name = name.strip()

            shutil.copy(source, path / name)

    _create_class(left_df, left_path)
    _create_class(right_df, right_path)

    if quantity != 1.0 or output_csv is not None:
        basename = filename.stem

        output_csv = Path(output_csv, f"{basename}_{quantity}.csv")

        df.to_csv(output_csv)

    return df


def anonymize_dataset(csv: str) -> list:
    """
    Add or modify the name's col
    """

    df = pd.read_csv(csv)
    nb_lines = df.shape[0]

    df["NAME"] = ""
    df["NAME"] = [f"{i:04d}.png" for i in range(nb_lines)]

    df.to_csv(csv, index=False)
