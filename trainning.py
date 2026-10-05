import numpy as np
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
    )

    return model


def test(model, dataset_path: str):
    print(f"dataset for test: {dataset_path}")
    metrics = model.val(
        data=dataset_path, split="test", imgsz=512, device=0, verbose=False
    )

    cm = metrics.confusion_matrix.matrix.T.astype(int)
    tp = np.diag(cm)

    accuracy = tp / np.maximum(cm.sum(axis=0), 1)
    recall = tp / np.maximum(cm.sum(axis=1), 1)
    f1 = 2 * accuracy * recall / np.maximum(accuracy + recall, 1e-12)

    names = [model.names[i] for i in range(len(model.names))]

    # Log
    for i, name in enumerate(names):
        print(
            f"{name:>10} | accuracy {accuracy[i]:.3f} | recall {recall[i]:.3f} | F1 {f1[i]:.3f}"
        )

    print(
        f"{'macro':>10} | accuracy {accuracy.mean():.3f} | recall {recall.mean():.3f} | F1 {f1.mean():.3f}"
    )
    print(f"Accuracy : {metrics.top1:.3f}")

    print("Confusion matrix :\n", cm)

    return metrics
