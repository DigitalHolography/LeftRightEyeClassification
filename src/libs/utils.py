import math
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def bilinear_resize(image, height, width):
    """
    `image` is a 2-D numpy array
    `height` and `width` are the desired spatial dimension of the new 2-D array.
    """
    img_height, img_width = image.shape[:2]

    resized = np.empty([height, width])

    x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
    y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0

    for i in range(height):
        for j in range(width):

            x_l, y_l = math.floor(x_ratio * j), math.floor(y_ratio * i)
            x_h, y_h = math.ceil(x_ratio * j), math.ceil(y_ratio * i)

            x_weight = (x_ratio * j) - x_l
            y_weight = (y_ratio * i) - y_l

            a = image[y_l, x_l]
            b = image[y_l, x_h]
            c = image[y_h, x_l]
            d = image[y_h, x_h]

            pixel = (
                a * (1 - x_weight) * (1 - y_weight)
                + b * x_weight * (1 - y_weight)
                + c * y_weight * (1 - x_weight)
                + d * x_weight * y_weight
            )

            resized[i][j] = pixel

    return resized


def txt_to_list(input: str) -> list[str]:
    lst = []
    with open(input) as file:
        lst = file.readlines()

    return lst


def is_left_right_pattern(file):
    LEFT_PATTERN = "_L_"
    RIGHT_PATTERN = "_R_"
    return (LEFT_PATTERN in file and RIGHT_PATTERN not in file) or (
        RIGHT_PATTERN in file and LEFT_PATTERN not in file
    )


def strip(str_: str):
    return str_.strip()


def is_exist(file: str):
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
    target = os.path.abspath(target)
    link = os.path.abspath(link)

    # Détecte aussi les liens cassés
    if os.path.lexists(link):
        return

    os.makedirs(os.path.dirname(link), exist_ok=True)

    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        import _winapi

        _winapi.CreateJunction(target, link)


def extract_subject_name(file: str) -> str | None:
    file = file.strip()

    # from a path get the basename
    filename = Path(file).stem

    lst = filename.split("_")
    if len(lst) == 0:
        return

    subject_id = lst[1]

    # get the biggest substring of char
    sequences = re.findall(r"\D+", subject_id)
    return max(sequences, key=len, default="")


def extract_eye_laterality(file: str) -> str | None:

    LEFT_PATTERN = "_L_"
    RIGHT_PATTERN = "_R_"

    if LEFT_PATTERN in file:
        return "L"

    if RIGHT_PATTERN in file:
        return "R"


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


def find_root_repository() -> Path:
    ROOT = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=Path(__file__).resolve().parent,
            text=True,
        ).strip()
    )

    return Path(ROOT)


def logscale(start, end, num=10):
    values = np.geomspace(start, end, num=num)
    return np.unique(np.array(np.round(values) * 2, dtype=int))
