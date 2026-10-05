import ntpath
import re
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd


def _fill_partition(images: list[str], output_file: str, subdataset: np.ndarray[str]):

    def find_name(subject, filename):
        basename = ntpath.basename(filename)
        return re.search(subject, basename) is not None

    with open(output_file, "w") as file:

        for subject in subdataset:
            subject_files = list(filter(partial(find_name, subject), images))
            file.writelines(subject_files)


def _log_subject_eye(train, valid, test):

    subject_train, nb_image_train = train
    subject_valid, nb_image_valid = valid
    subject_test, nb_image_test = test

    nb_subject_train = len(subject_train)
    nb_subject_valid = len(subject_valid)
    nb_subject_test = len(subject_test)

    total_subject = nb_subject_train + nb_subject_valid + nb_subject_test
    total_eyes = nb_image_train + nb_image_valid + nb_image_test

    s = f"train : subject {nb_subject_train}, eye {nb_image_train}\n"
    s += f"valid : subject {nb_subject_valid}, eye {nb_image_valid}\n"
    s += f"test  : subject {nb_subject_test}, eye {nb_image_test}\n"
    s += f"total : subject {total_subject}, eye {total_eyes}\n"

    print(s)


def split_dataset_source(
    csv: str,
    output_dir: str,
    ratio_train=0.70,
    ratio_valid=0.15,
    ratio_test=0.15,
    margin_train=0.05,
    margin_test=0.02,
):

    if not Path(csv).exists():
        return

    df = pd.read_csv(csv)

    with open(dataset, "r") as file:
        images = file.readlines()

        hist = histogram_subject(images)

        train, valid, test = _repeated_weighted_draw(
            hist, ratio_train, ratio_test, margin_train, margin_test
        )

        _log_subject_eye(train, valid, test)

        # split into 3 folder
        _fill_file(images, ntpath.join(output_dir, "train.txt"), train[0])
        _fill_file(images, ntpath.join(output_dir, "valid.txt"), valid[0])
        _fill_file(images, ntpath.join(output_dir, "test.txt"), test[0])


def _fill_classes(
    rng,
    hist: dict,
    values: np.ndarray,
    weights: np.ndarray,
    total_images: int,
    quantity_goal: int,
    interval: int,
    timeout=2000,
):
    """
    Use a timetout because greedy doesn't garanty a global convergence
    """

    values = values.copy()
    weights = weights.copy()

    result = []
    count_result = 0
    invalid_draw = 0

    while invalid_draw < timeout and values.size > 0:
        # normalisation (garenti that the sum is 1)
        p = weights / weights.sum()
        idx = rng.choice(values.size, p=p)

        draw = values[idx]
        ratio = ((count_result + hist[draw]) / total_images) - quantity_goal

        # if we add the patient we add is to big we look for a better one
        if ratio > interval:
            invalid_draw += 1
            continue

        invalid_draw = 0
        result.append(draw)
        count_result += hist[draw]

        # remove
        values = np.delete(values, idx)
        weights = np.delete(weights, idx)

        # => 0 <= ratio < interval
        if ratio >= 0:
            break

    return result, count_result, values


def _probabilistic_greedy_function(
    train,
    ratio_train: int,
    valid,
    ratio_valid: int,
    test,
    ratio_test: int,
    lambda_size=10,
    lambda_diversity=1,
):

    # E_size
    [_, n_image_train] = train
    [_, n_image_valid] = valid
    [subjects_test, n_image_test] = test

    n_image = n_image_train + n_image_test + n_image_valid

    E_size = (
        abs((n_image_train / n_image) - ratio_train)
        + abs((n_image_valid / n_image) - ratio_valid)
        + abs((n_image_test / n_image) - ratio_test)
    )

    # E_diversity
    E_diversity = len(subjects_test) / n_image_test if n_image_test > 0 else 0.0

    return lambda_size * E_size - lambda_diversity * E_diversity


def _repeated_weighted_draw(
    hist: dict[str, int],
    ratio_train: float,
    ratio_test: float,
    itrain: float,
    itest: float,
    draw_nb=1000,
):
    """
    :params qtrain, qvalid, qtest:
        are the quantity of the dataset in each category and
    :params itrain, itest:
        are the degree of liberty around the quantity's goal
    """

    rng = np.random.default_rng()

    best = None
    best_cost = np.inf

    for _ in range(draw_nb):
        subjects = np.asarray(list(hist.keys()))
        weights = np.asarray(list(hist.values()))

        total_images_nb = weights.sum()

        # draw for train
        train, train_images_nb, remaining = _fill_classes(
            rng, hist, subjects, weights, total_images_nb, ratio_train, itrain
        )

        subjects = remaining.copy()
        weights = np.asarray([hist[s] for s in subjects])

        # draw for test
        test, test_images_nb, remaining = _fill_classes(
            rng, hist, subjects, 1.0 / weights, total_images_nb, ratio_test, itest
        )

        # the remaining goes to valid
        valid = remaining.copy()
        ratio_valid = 1 - ratio_train - ratio_test
        valid_images_nb = total_images_nb - train_images_nb - test_images_nb

        # eval the quality of the split
        cost = _probabilistic_greedy_function(
            [train, train_images_nb],
            ratio_train,
            [valid, valid_images_nb],
            ratio_valid,
            [test, test_images_nb],
            ratio_test,
        )

        # maximise the quality
        if cost < best_cost:
            best = (
                [train, train_images_nb],
                [valid, valid_images_nb],
                [test, test_images_nb],
            )
            best_cost = cost

    return best


def _copy_anonymized_files(
    dict_anonymize: dict[str], files: list[str], output_dir: str
):

    for file in files:
        name = dict_anonymize.get(file)
        if name is None:
            print(f"Failed to copy and anonymized {file}")
            continue

        total_name = ntpath.join(output_dir, name)
        shutil.copy(file, total_name)


def _split_left_right(files: list[str]) -> tuple[list[str], list[str]]:
    LEFT_PATTERN = "_L_"
    RIGHT_PATTERN = "_R_"

    left = []
    right = []
    unknown = []

    for file in files:

        basename = ntpath.basename(file)

        is_left = LEFT_PATTERN in basename
        is_right = RIGHT_PATTERN in basename

        if is_left and not is_right:
            left.append(file)
        elif is_right and not is_left:
            right.append(file)
        else:
            unknown.append(file)

    # a sort of safeguard (we can do better)
    if len(unknown) > 0:
        return

    return left, right


def _anonymize(filenames: list[str]) -> dict[str, str]:

    filenames = sorted(filenames)

    subject_ids: dict[str, int] = {}
    file_counters: dict[str, int] = defaultdict(int)

    res = {}

    for file in filenames:
        name = extract_subject_name(file)

        if name not in subject_ids:
            subject_ids[name] = len(subject_ids)
        subject_id = subject_ids[name]

        file_idx = file_counters[name]
        file_counters[name] += 1

        ext = Path(file).suffix
        new_name = f"{subject_id:03d}{file_idx:03d}{ext}"

        res[file] = new_name

    return res


def split_dataset_images(sources_file: str, output_dir: str, output_csv: str):

    # copy files into right or left folder
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    left_path = ntpath.join(output_dir, "left")
    right_path = ntpath.join(output_dir, "right")

    Path(left_path).mkdir(exist_ok=True, parents=True)
    Path(right_path).mkdir(exist_ok=True, parents=True)

    with open(sources_file) as sources:

        filenames = sources.readlines()
        filenames = [s.strip() for s in filenames]

        # split left and right
        split = _split_left_right(filenames)

        if split is None:
            print(f"Failed to split {sources_file} into left and right")
            return

        left_files, right_files = split

        dict_anonymize = _anonymize(filenames)

        # copy files
        _copy_anonymized_files(dict_anonymize, left_files, left_path)
        _copy_anonymized_files(dict_anonymize, right_files, right_path)

        # register into a csv
