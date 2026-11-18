import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from libs.utils import (
    bilinear_resize,
    extract_eye_laterality,
    extract_subject_name,
    list_to_csv,
    strip,
    txt_to_list,
)

""" ------------------------------------------- """


def _create_class(df, output: Path, side_folder: str, side_tag: str, source_col: str):
    side_path = output / side_folder

    # remove the images of a previous generation
    if side_path.exists():
        shutil.rmtree(side_path)
    side_path.mkdir(parents=True, exist_ok=True)
    class_df: pd.DataFrame = df.loc[df["SIDE"] == side_tag]

    for _, row in class_df.iterrows():
        source = row[source_col].strip('"')

        name = row["LOCAL_REF"].strip()
        name = Path(name).name

        name_path = side_path / name  # absolute path
        shutil.copy(source, name_path)

    return class_df


def _get_dataframe(input: str | pd.DataFrame) -> pd.DataFrame:
    # input could be 2 diff type, a dataframe or a str
    if isinstance(input, str):
        return pd.read_csv(Path(input))

    assert isinstance(input, pd.DataFrame)
    return input


def _get_numbers_rows(df: pd.DataFrame, quantity: float, is_subject_level: bool) -> int:
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
    quantity: float,
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
    source_col: str = "SOURCE",
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
    - is_downloaded: take the csv and put all

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

    _create_class(sub_df, output, "left", "L", source_col)
    _create_class(sub_df, output, "right", "R", source_col)

    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        sub_df.to_csv(output_csv, index=False)

    return sub_df


""" ------------------------------------------- """


def _subjects(df: pd.DataFrame) -> np.ndarray:
    """
    List the subjects (PATIENT) of the dataframe, keep the csv order
    Check that each subject has exactly 1 Left and 1 Right
    """
    counts = df.groupby("PATIENT")["SIDE"].agg(lambda s: "".join(sorted(s)))
    invalid = counts[counts != "LR"]
    assert invalid.empty, f"subjects without exactly 1 L and 1 R: {invalid.to_dict()}"

    return df["PATIENT"].drop_duplicates().to_numpy()


def _select_subjects(df: pd.DataFrame, subjects) -> pd.DataFrame:
    sub_df = df[df["PATIENT"].isin(subjects)]
    # L first then R, like in _get_subdataset
    return pd.concat(
        [sub_df.loc[sub_df["SIDE"] == "L"], sub_df.loc[sub_df["SIDE"] == "R"]],
        ignore_index=True,
    )


def check_no_collision(*dataframes: pd.DataFrame):
    """
    Assert that no subject and no image are shared between the dataframes
    """
    seen_patients: set = set()
    seen_sources: set = set()

    for i, df in enumerate(dataframes):
        patients = set(df["PATIENT"])
        sources = set(df["SOURCE"])

        assert not df["SOURCE"].duplicated().any(), f"duplicated image in df {i}"
        assert not (
            patients & seen_patients
        ), f"subject collision in df {i}: {patients & seen_patients}"
        assert not (
            sources & seen_sources
        ), f"image collision in df {i}: {sources & seen_sources}"

        seen_patients |= patients
        seen_sources |= sources


def _generate_subdataset(
    dataset: pd.DataFrame | str,
    sizes: Sequence[int] | np.ndarray,
    random: bool = False,
    is_linked: bool = False,
) -> list[pd.DataFrame]:
    """
    sizes are in number of images (2 images per subject, so must be even)
    The draws are done at subject level (a subject is never split)
    - not linked: disjoint subdatasets, sum(sizes) <= dataset size
    - linked: nested subdatasets (each one contains the previous),
    max(sizes) <= dataset size
    """

    sizes = np.asarray(sizes, dtype=int)
    assert (sizes % 2 == 0).all(), f"sizes must be even: {sizes}"

    df = _get_dataframe(dataset)
    subjects = _subjects(df)
    n_subjects_needed = sizes.max() if is_linked else sizes.sum()
    assert (
        n_subjects_needed // 2 <= subjects.size
    ), f"need {n_subjects_needed} images but the dataset has {subjects.size * 2}"

    if random:
        subjects = np.random.default_rng().permutation(subjects)

    res = []
    start = 0
    for size in sizes:
        n_subject = size // 2

        if is_linked:
            draws = subjects[:n_subject]
        else:
            draws = subjects[start : start + n_subject]
            start += n_subject

        sub_df = _select_subjects(df, draws)
        assert sub_df.shape[0] == size
        res.append(sub_df)

    if not is_linked:
        check_no_collision(*res)

    return res


def split_dataset(
    dataset: str | pd.DataFrame,
    subject_repartition: Sequence[int] | np.ndarray,
    random: bool = False,
):
    return _generate_subdataset(dataset, subject_repartition, random, is_linked=False)


def generate_increasing_datasets(
    dataset: str | pd.DataFrame, sizes: Sequence[int] | np.ndarray, random: bool = False
):
    return _generate_subdataset(dataset, sizes, random, is_linked=True)


""" ------------------------------------------- """


def join_dataframe(*dataframes: pd.DataFrame | str) -> pd.DataFrame:
    frames = [_get_dataframe(df).copy() for df in dataframes]
    return pd.concat(frames, ignore_index=True)


def source_to_dataframe(input: list[str]) -> pd.DataFrame:
    """
    From a list of sources build a dataframe with {sources, patient, side}
    The basename should contain all information required
    params:
        input: Is a list with all sources
    """
    data = {"SOURCE": [], "PATIENT": [], "TYPE": [], "SIDE": [], "PATHO": []}
    for source in input:
        patient: str | None = extract_subject_name(source)
        side: str | None = extract_eye_laterality(source)

        if side == None or patient == None:
            continue

        data["SOURCE"].append(source.strip())
        data["PATIENT"].append(patient)
        data["TYPE"].append("retinal seg")
        data["SIDE"].append(side)
        data["PATHO"].append(None)

    return pd.DataFrame(data=data)


def anonymize_dataset(csv: str):
    """
    Add or modify the name's col
    """

    df = pd.read_csv(csv)
    nb_lines = df.shape[0]

    df["LOCAL_REF"] = ""
    df["LOCAL_REF"] = [f"{i:04d}.png" for i in range(nb_lines)]

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


""" -------------------------------------------------- """


def _get_moment0ff_path(holo: str):
    holo_path: Path = Path(holo)
    basename: str = holo_path.stem
    hd_path: Path = holo_path.parent / basename / f"{basename}_HD"

    return str(hd_path / "png/moment0ff.png")


def create_retinal_segmentation_testset(
    input: str, train_csv: str, output_dir: str, output_csv: str
) -> pd.DataFrame:
    """
    !!! This function is specific to this experiment !!!
    - get all patient not in the train_csv
    - do bilineare resize
    - concat artery and vein segmentation
    - save all in output folder
    """

    df_input: pd.DataFrame = source_to_dataframe(txt_to_list(input))
    df_train: pd.DataFrame = pd.read_csv(train_csv)

    df_input["LOCAL_REF"] = [f"{i:04d}.png" for i in range(df_input.shape[0])]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    # pathes
    seg_ground_truth = "ground_truths"
    vein_mask_name = "retina_vein_mask.png"
    artery_mask_name = "retina_artery_mask.png"

    # filter
    mask = df_input["PATIENT"].isin(df_train["PATIENT"])
    df = df_input[~mask]
    df.to_csv(output_csv, index=False)

    for _, row in df.iterrows():

        # concat two bilineare resizes images
        source = row["SOURCE"].strip('"')
        name = row["LOCAL_REF"]

        folder = Path(source).stem
        parent = Path(source).parent
        seg_path = Path(parent, folder) / seg_ground_truth

        vein_path = seg_path / vein_mask_name
        artery_path = seg_path / artery_mask_name

        if not vein_path.exists() or not artery_path.exists():
            print(f"Failed to find segmentation: {vein_path} or {artery_path}")
            continue

        vein = np.asarray(Image.open(vein_path).convert("L"))
        artery = np.asarray(Image.open(artery_path).convert("L"))
        seg_mask = np.maximum(vein, artery)
        resize_seg_mask = bilinear_resize(seg_mask, 512, 512)

        # get the segmentation (with the mask)
        moment0ff_path = _get_moment0ff_path(source)
        moment0ff = np.asarray(Image.open(moment0ff_path).convert("L"))

        assert moment0ff.shape == mask.shape, f"{moment0ff.shape} vs {mask.shape}"

        mask = resize_seg_mask > 0
        image = np.where(mask, moment0ff, 0)

        # save
        Image.fromarray(image.astype(np.uint8)).save(Path(output_dir) / name)

    return df


""" -------------------------------------------------- """


def download_images(input: str, output_dir: str):
    """
    Download from the source -> output_dir and change the local_ref to point on the
    right local path.
    """

    df: pd.DataFrame = _get_dataframe(input)
    output_dir_path: Path = Path(output_dir)

    for index, row in df.iterrows():
        source = row["SOURCE"].strip('"')
        name = str(output_dir_path / row["LOCAL_REF"].strip())

        df.loc[index, "LOCAL_REF"] = name

        Path(name).parent.mkdir(parents=True, exist_ok=True)

        shutil.copy(source, name)

    df.to_csv(input, index=False)
