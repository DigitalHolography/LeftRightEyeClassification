import math
import os
from pathlib import Path

from PIL import Image
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from ultralytics import YOLO

from libs.utils import find_root_repository


def train(
    model_name: str,
    dataset_path: str | Path,
    config: dict,
    project_name: str = "train/",
):
    model = YOLO(config["model"])

    model.train(
        data=str(dataset_path),
        project=project_name,
        name=model_name,
        exist_ok=True,
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        device=0,  # GPU
        optimizer=config["optimizer"],
        verbose=False,
        val=False,
        patience=0,
        fliplr=0.0,
        deterministic=True,
        seed=42,
        amp=False,
        workers=8,
    )

    return model


def test(model, dataset_path: str | Path):
    print(f"dataset for test: {dataset_path}")
    metrics = model.val(
        data=str(dataset_path), split="test", imgsz=512, device=0, verbose=False
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

    side_by_name = df.set_index("LOCAL_REF")["SIDE"].to_dict()

    for image in dir_list:
        image_path = find_root_repository() / Path(input_dir) / image

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


def kfold_cross_validation(n: int, dataset: str, config: dict):
    dataset_path = Path(dataset)
    histories = []
    metrics = []

    for i in range(n):
        model_path = f"fold_{i}"

        # training
        model = train(model_path, str(dataset_path / model_path), config)

        # save results
        save_dir = model.trainer.save_dir
        df = pd.read_csv(save_dir / "results.csv")
        df.columns = df.columns.str.strip()
        histories.append(df)
        last_model = YOLO(str(model.trainer.last))

        metric = last_model.val(name=f"fold_{i}")

        # get accuracy recal and F1 scores
        cm = metric.confusion_matrix.matrix.T.astype(int)
        n = len(last_model.names)
        names = [last_model.names[i] for i in range(n)]

        idx = np.arange(len(names))
        y_true = np.repeat(idx, cm.sum(axis=1))
        y_pred = np.concatenate([np.repeat(idx, row) for row in cm])

        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        metrics.append((metric.top1, p, r, f1))

    return histories, metrics


def layercam_load_folder(model: YOLO, input_folder: str, output_folder: str):

    if (not Path(input_folder).exists()) or (not Path(input_folder).is_dir()):
        raise ValueError(
            f"Input folder {input_folder} does not exist or is not a directory."
        )

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    for image in os.listdir(input_folder):
        if not image.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        image = find_root_repository() / Path(input_folder) / image
        predict_folder = Path(image).stem
        path_prediction = find_root_repository() / Path(
            "runs", "classify", predict_folder
        )

        name = f"{Path(image).stem}_cam.jpg"
        r = model.predict(
            image, name=predict_folder, verbose=False, visualize=True, exist_ok=True
        )[0]

        path = Path(path_prediction, name)
        if not path.exists():
            print(f"Failed to predict {path}")
            return

        image = np.asarray(Image.open(path))

        label_idx = r.probs.top1
        label = r.names[label_idx]

        # Extract label and confidence
        if hasattr(r, "probs") and r.probs is not None:
            probs = r.probs.data.cpu().numpy()
            label_idx = np.argmax(probs)
            label = r.names[label_idx]
            confidence = probs[label_idx]
        else:
            # Fallback for different YOLO versions
            label = "Unknown"
            confidence = 0.0

        # Choose color based on label
        if label.lower() == "left":
            color = (255, 0, 0)  # Red
        elif label.lower() == "right":
            color = (0, 255, 0)  # Green
        else:
            color = (255, 255, 0)  # Yellow

        # Draw border inside the image array
        border_thickness = 10
        img_with_border = cv2.copyMakeBorder(
            image,
            border_thickness,
            border_thickness,
            border_thickness,
            border_thickness,
            cv2.BORDER_CONSTANT,
            value=color,
        )

        # Add text label

        conf_trunc = math.floor(confidence * 100 + 1e-9) / 100
        label_text = f"{label.upper()} ({conf_trunc:.2f})"
        cv2.putText(
            img_with_border,
            label_text,
            (25, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        img_with_border = cv2.cvtColor(img_with_border, cv2.COLOR_RGB2BGR)

        # save the image with border and label
        output_path = Path(output_folder, name)
        cv2.imwrite(str(output_path), img_with_border)
