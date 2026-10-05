import base64
import html
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from ultralytics import YOLO


def plot_kfold_report(
    histories,
    scores,
    other_metrics,
    names: list[str],
    titre: str | None = None,
    show: bool = True,
):
    curve_metrics = [
        ("train/loss", "Loss d'entraînement"),
        ("val/loss", "Loss de validation"),
        ("metrics/accuracy_top1", "Accuracy top1 (validation)"),
    ]

    # ---------- Données des courbes ----------
    dfs = []
    for i, df in enumerate(histories):
        df = df.copy()
        df.columns = df.columns.str.strip()
        df["fold"] = i
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)

    # ---------- Données de la matrice ----------
    n_folds = len(scores)
    row_names = [f"Fold {i}" for i in range(n_folds)] + ["mean", "std"]
    folds = np.column_stack([np.asarray(scores), np.asarray(other_metrics)])
    M = np.vstack([folds, folds.mean(axis=0), folds.std(axis=0)])

    # ---------- Grille ----------
    h_bas = max(3.5, 0.45 * len(row_names))
    fig = plt.figure(figsize=(18, 4.5 + h_bas + 1), layout="constrained")
    gs = fig.add_gridspec(2, 6, height_ratios=[4.5, h_bas])

    curve_axes = [fig.add_subplot(gs[0, 2 * k : 2 * k + 2]) for k in range(3)]
    ax_heat = fig.add_subplot(gs[1, :3])

    # ---------- Ligne 1 : courbes ----------
    colors = plt.cm.tab10.colors
    for ax, (col, title) in zip(curve_axes, curve_metrics):
        if col not in all_df.columns:
            ax.set_title(f"{title}\n(colonne '{col}' absente)")
            continue

        for i, df in enumerate(dfs):
            ax.plot(
                df["epoch"],
                df[col],
                color=colors[i % 10],
                alpha=0.6,
                linewidth=1.2,
                label=f"fold {i}",
            )

        stats = all_df.groupby("epoch")[col].agg(["mean", "std"])
        ax.plot(
            stats.index, stats["mean"], color="black", linewidth=2.5, label="moyenne"
        )
        ax.fill_between(
            stats.index,
            stats["mean"] - stats["std"],
            stats["mean"] + stats["std"],
            color="gray",
            alpha=0.2,
            label="± écart-type",
        )

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)

    curve_axes[0].set_ylabel("Loss")
    curve_axes[-1].set_ylabel("Accuracy")
    handles, labels = curve_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside upper center", ncol=len(labels))

    # ---------- Ligne 2 gauche : heatmap ----------
    im = ax_heat.imshow(M, cmap="viridis", aspect="auto")
    ax_heat.set_xticks(np.arange(M.shape[1]), labels=names)
    ax_heat.set_yticks(np.arange(M.shape[0]), labels=row_names)
    plt.setp(ax_heat.get_xticklabels(), rotation=45, ha="right")
    ax_heat.axhline(n_folds - 0.5, color="white", linewidth=2)  # sépare folds / stats

    seuil = (M.max() + M.min()) / 2
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            couleur = "white" if M[i, j] < seuil else "black"
            ax_heat.text(
                j, i, f"{M[i, j]:.3f}", ha="center", va="center", color=couleur
            )
    ax_heat.set_title("Scores par fold")
    fig.colorbar(im, ax=ax_heat, shrink=0.8)

    if titre:
        fig.suptitle(titre, fontsize=15, fontweight="bold")
    if show:
        plt.show()

    return fig


def _html_text(texte: str) -> str:
    paragraphes = [p.strip() for p in texte.strip().split("\n\n") if p.strip()]
    return "\n".join(
        f"<p>{html.escape(p).replace(chr(10), '<br>')}</p>" for p in paragraphes
    )


def add_text_to_report(
    title: str = "", text: str = "", level: int = 2, path: str = "report.html"
):
    with open(path, "a", encoding="utf-8") as f:
        if title:
            f.write(f"<h{level}>{html.escape(title)}</h{level}>\n")
        if text:
            f.write(_html_text(text) + "\n")


def add_fig_to_report(fig, titre="", path="report.html"):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    img = base64.b64encode(buf.getvalue()).decode()
    with open(path, "a", encoding="utf-8") as f:
        f.write(f'<h2>{titre}</h2>\n<img src="data:image/png;base64,{img}"><br>\n')


def _macro_scores(m):
    cm = m.confusion_matrix.matrix.T.astype(int)
    idx = np.arange(cm.shape[0])
    y_true = np.repeat(idx, cm.sum(axis=1))
    y_pred = np.concatenate([np.repeat(idx, row) for row in cm])
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=idx, average="macro", zero_division=0
    )
    return p, r, f1


def plot_metrics(results, model_names: list[str], ncols: int = 3):
    """results     : liste d'objets renvoyés par model.val(), un par modèle
    model_names : nom de chaque modèle (même ordre que results)"""
    assert len(results) == len(
        model_names
    ), "results et model_names doivent avoir la même taille"

    metric_names = ["top1", "top5", "precision", "recall", "F1"]
    M = np.array(
        [[m.top1, m.top5, *_macro_scores(m)] for m in results]
    )  # (modèles, métriques)

    n_metrics = len(metric_names)
    ncols = min(ncols, n_metrics)
    nrows = int(np.ceil(n_metrics / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
    )
    axes = axes.ravel()

    x = np.arange(len(model_names))
    for j, (ax, name) in enumerate(zip(axes, metric_names)):
        v = M[:, j]
        ax.plot(x, v, marker="o", lw=2)
        for xi, vi in zip(x, v):
            ax.annotate(
                f"{vi:.3f}",
                (xi, vi),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=9,
            )
        ax.set_title(name)
        ax.set_xticks(x, labels=model_names, rotation=30, ha="right")
        marge = max(0.02, (v.max() - v.min()) * 0.3)
        ax.set_ylim(v.min() - marge, min(1.02, v.max() + marge))
        ax.grid(alpha=0.3)

    for ax in axes[n_metrics:]:  # cache les cases vides de la grille
        ax.set_visible(False)

    fig.suptitle("Comparaison des modèles", fontsize=14)
    fig.tight_layout()
    plt.show()


def plot_LayerCAM(model: YOLO, data: str, class_name: str):

    predict_folder = f"predict_LayerCAM_{class_name}"
    path_prediction = Path("runs", "classify", predict_folder)

    lst = []

    for img in Path(data).iterdir():
        name = f"{Path(img).stem}_cam.jpg"
        r = model.predict(
            img, name=predict_folder, verbose=False, visualize=True, exist_ok=True
        )[0]

        path = Path(path_prediction, name)
        if not path.exists():
            print(f"Failed to predict {path}")
            continue

        # skip wrong result
        if model.names[r.probs.top1] != class_name:
            continue

        lst.append(np.asarray(Image.open(path)))

    n_image = len(lst)
    n_cols = 3
    n_rows = n_image // n_cols

    _, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False
    )

    for ax, arr in zip(axes.flat, lst):
        ax.imshow(arr)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
