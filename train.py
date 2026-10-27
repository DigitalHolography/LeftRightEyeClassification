import os
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO


def train(model_name: str, dataset_path: str, config: dict):

    model = YOLO(config["model"])

    model.train(
        data=dataset_path,
        project="train/",
        name=model_name,
        exist_ok=True,
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        device=0,  # GPU
        optimizer=config["optimizer"],
        verbose=False,
        fliplr=0.0,
        val=False,
        patience=0,
    )

    return model


def test(model, dataset_path: str):
    print(f"dataset for test: {dataset_path}")
    metrics = model.val(
        data=dataset_path, split="test", imgsz=512, device=0, verbose=False
    )

    cm = metrics.confusion_matrix.matrix.T.astype(int)
    tp = np.diag(cm)

    accuracy = metrics.top1
    precision = tp / np.maximum(cm.sum(axis=0), 1)
    recall = tp / np.maximum(cm.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)

    names = [model.names[i] for i in range(len(model.names))]

    # Log
    for i, name in enumerate(names):
        print(
            f"{name:>10} | precision {precision[i]:.3f} | recall {recall[i]:.3f} | F1 {f1[i]:.3f}"
        )

    print(
        f"{'macro':>10} | precision {precision.mean():.3f} | recall {recall.mean():.3f} | F1 {f1.mean():.3f}"
    )
    print(f"Accuracy : {metrics.top1:.3f}")

    print("Confusion matrix :\n", cm)

    return metrics, [accuracy, precision.mean(), recall.mean(), f1.mean()]


def predict_segmentations_folder(model: YOLO, input_dir: str, csv: str):
    dir_list = os.listdir(input_dir)
    df: pd.DataFrame = pd.read_csv(csv)

    right_true = 0
    left_true = 0
    right_false = 0
    left_false = 0

    def from_name_to_tag(name):
        if "left" == name:
            return "L"
        elif "right" == name:
            return "R"
        return None

    side_by_name = df.set_index("NAME")["SIDE"].to_dict()

    for image in dir_list:
        image_path = Path(input_dir) / image

        r = model.predict(image_path, imgsz=512, device=0, verbose=False)[0]

        truth = side_by_name.get(image)
        if truth is None:
            continue

        pred = model.names[r.probs.top1]
        conf = r.probs.top1conf.item()
        tag_pred = from_name_to_tag(pred)

        if truth == "L":
            if tag_pred == "L":
                left_true += 1
            elif tag_pred == "R":
                print(
                    f"Failed to predict {image} - pred: {tag_pred}, truth: {truth}, confidence: {conf:.3f}"
                )
                left_false += 1
        elif truth == "R":
            if tag_pred == "R":
                right_true += 1
            elif tag_pred == "L":
                print(
                    f"Failed to predict {image} - pred: {tag_pred}, truth: {truth}, confidence: {conf:.3f}"
                )
                right_false += 1
        else:
            print(
                f"Failed to predict {image} - pred: {tag_pred}, truth: {truth}, confidence: {conf:.3f}"
            )

    print("Stats:")
    # print(f"- Successful prediction / Total: {count} / {len(dir_list)}")
    print(f"right true {right_true} | left true {left_true}")
    print(f"right false {right_false} | left false {left_false}")
    print(
        f"{(right_true + left_true) / (right_false + right_true + left_false + left_true) }"
    )
