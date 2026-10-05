import _winapi
import os
from pathlib import Path

import pandas as pd


def is_left_right_pattern(file):
    LEFT_PATTERN = "_L_"
    RIGHT_PATTERN = "_R_"
    return (LEFT_PATTERN in file and RIGHT_PATTERN not in file) or (
        RIGHT_PATTERN in file and LEFT_PATTERN not in file
    )


def strip(str_: str):
    return str_.strip()


def is_exist(file):
    return Path(file).exists()


def list_to_csv(lst: list[str], col_name: str, output_csv: str):
    dict = {col_name: lst}
    df = pd.DataFrame(data=dict)
    df.to_csv(output_csv)

    return df


def text_to_csv(txt: str, col_name: str, output_csv: str):

    if not Path(txt).exists():
        return

    with open(txt, "w") as file:
        lst = file.readlines()
        return list_to_csv(lst, col_name, output_csv)


def link_dir(target, link):
    if Path(link).exists():
        return

    _winapi.CreateJunction(os.path.abspath(target), os.path.abspath(link))


def histogram_names(files: list[str]) -> dict:
    """
    Do an histogram of name with the name in key and the number of eye with this name
    """
    hist = {}

    for file in files:
        name = extract_subject_name(file)

        count = hist.get(name, 0)
        hist[name] = count + 1

    return hist
