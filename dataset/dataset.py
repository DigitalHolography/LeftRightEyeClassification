import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import list_to_csv, strip


def _create_class(df, output: Path, side_folder: str, side_tag: str):
    side_path = output / side_folder
    side_path.mkdir(parents=True, exist_ok=True)
    class_df: pd.DataFrame = df.loc[df["SIDE"] == side_tag]

    for _, row in class_df.iterrows():
        source = row["SOURCE"].strip('"')
        name = row["NAME"].strip()
        shutil.copy(source, side_path / name)

    return class_df


def _get_dataframe(input: str | pd.DataFrame) -> pd.DataFrame:
    # input could be 2 diff type, a dataframe or a str
    if isinstance(input, str):
        return pd.read_csv(Path(input))

    assert isinstance(input, pd.DataFrame)
    return input


def _get_numbers_rows(df: pd.DataFrame, quantity: int, is_subject_level: bool) -> int:
    n_rows = 0
    if quantity <= 1:
        n_rows = int(len(df) * quantity)
        n_rows += n_rows % 2
        return n_rows

    n_rows = int(quantity)
    assert is_subject_level and quantity % 2 == 0

    return n_rows


def _get_subdataset(
    df: pd.DataFrame,
    n_rows: int,
    quantity: int,
    is_random: bool,
    is_subject_level: bool,
) -> pd.DataFrame:

    if not is_random or quantity == 1:
        return df.iloc[:n_rows]

    total_subject = df.shape[0] // 2
    n_subject = n_rows // 2

    rng = np.random.default_rng()
    draws_right = None

    draws_left = rng.choice(
        np.arange(total_subject), size=n_subject, replace=False
    ).tolist()

    # if not subject level we possibly not have L and R of the same subject
    if is_subject_level:
        draws_right = draws_left.copy()
    else:
        draws_right = rng.choice(
            np.arange(total_subject), size=n_subject, replace=False
        ).tolist()

    draws_left = np.asarray(draws_left)
    draws_right = np.asarray(draws_right)

    left_df: pd.DataFrame = df.loc[df["SIDE"] == "L"]
    right_df: pd.DataFrame = df.loc[df["SIDE"] == "R"]

    left_df = left_df.iloc[draws_left]
    right_df = right_df.iloc[draws_right]

    return pd.concat([left_df, right_df], ignore_index=True)


def create_dataset(
    input: str | pd.DataFrame,
    output_dir: str,
    output_csv: str | None = None,
    quantity: float = 1.0,
    is_random: bool = False,
    is_subject_level: bool = False,
) -> pd.DataFrame:
    """
    params:
    - Quantity:
        Is a number line (if > 1) or a percentage (if [0, 1])
        Quantity (in normalize pourcentage) of the dataset filename (input file)
    takes to create the real dataset.
    Create a csv of the subdataset if the quantity < 1
    - random: Take random people in the dataset
    - subject: Considere a subject as an entity. Then all operation are done
    one the entity

    Informations:
        - assume that the csv is sorted by the subject name
        - assume that there is 1 Left and 1 Right
    """

    assert quantity > 0
    assert len(output_dir) > 0 and len(input) > 0

    output = Path(output_dir)
    df: pd.DataFrame = _get_dataframe(input)
    n_rows: int = _get_numbers_rows(df, quantity, is_subject_level)
    sub_df = _get_subdataset(df, n_rows, quantity, is_random, is_subject_level)

    _create_class(sub_df, output, "left", "L")
    _create_class(sub_df, output, "right", "R")

    if quantity != 1.0 or output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
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


def _generate_subdataset(
    dataset: pd.DataFrame | str,
    sizes: np.ndarray,
    random: bool = False,
    is_linked: bool = False,
) -> list[pd.DataFrame]:
    """
    Don't split at subject level it's the responsability of the user
    """

    repartition: np.ndarray = sizes
    df = _get_dataframe(dataset)
    res = []
    rng = np.random.default_rng()

    total = repartition.sum()

    assert df.shape[0] <= total * 2

    left_df: pd.DataFrame = df.loc[df["SIDE"] == "L"]
    right_df: pd.DataFrame = df.loc[df["SIDE"] == "R"]

    if not random:
        draws = list(range(total))
    else:
        draws = rng.choice(np.arange(total), size=total, replace=False).tolist()

    start = 0
    end = 0
    for size in repartition:
        end += size

        if not is_linked:
            indexes = draws[start:end]
        else:
            indexes = draws[:end]

        start = end

        left = left_df.iloc[indexes]
        right = right_df.iloc[indexes]

        L = pd.concat([left, right], ignore_index=True)
        res.append(L)

    return res


def split_dataset(
    dataset: str | pd.DataFrame,
    subject_repartition: list[int],
    random: bool = False,
):
    return _generate_subdataset(dataset, subject_repartition, random, is_linked=False)


def generate_increasing_datasets(
    dataset: str | pd.DataFrame, sizes: list[int], random: bool = False
):
    return _generate_subdataset(dataset, sizes, random, is_linked=True)


def join_dataset(*dataframes: list[pd.DataFrame | str]) -> pd.DataFrame:
    dataframes = [_get_dataframe(df) for df in dataframes]
    return pd.concat(dataframes)
