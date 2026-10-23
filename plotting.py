import base64
import html
import io
import numbers
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from ultralytics import YOLO


def plot_kfold_report(
    histories,
    scores,
    names: list[str],
    titre: str | None = None,
    show: bool = True,
):
    """scores : une ligne par fold, une valeur par métrique (même ordre que names).
    Les colonnes en trop sont ignorées."""

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
    rows = [
        list(r) if isinstance(r, (list, tuple, np.ndarray)) else [r] for r in scores
    ]
    n_folds = len(rows)
    n_cols = min(min(len(r) for r in rows), len(names))
    assert n_cols > 0, "scores et names doivent contenir au moins une métrique"
    names = list(names)[:n_cols]

    folds = np.array([[float(v) for v in r[:n_cols]] for r in rows])
    row_names = [f"Fold {i}" for i in range(n_folds)] + ["mean", "std"]
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
        ax.plot(stats.index, stats["mean"], color="black", linewidth=2.5, label="mean")
        ax.fill_between(
            stats.index,
            stats["mean"] - stats["std"],
            stats["mean"] + stats["std"],
            color="gray",
            alpha=0.2,
            label="± std",
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


def plot_metrics(
    results,
    model_names: list,
    metric_names: list[str] = ("accuracy", "recall", "f1"),
    metrics: list | None = None,
    ncols: int = 3,
    title: str | None = None,
    show_band: bool = True,
    show_points: bool = True,
    savename: str | None = None,
):
    """Plot one subplot per entry of `metrics`, with mean ± std.

    results      : list of rows, one per model; each row holds one cell per
                   metric, in the order given by metric_names. A cell is:
                   - (mean, std)  -> mean with standard deviation
                   - mean         -> scalar, std treated as 0
                   A row may also be an array of shape (n_metrics, 2).
    model_names  : name of each model (same order as results).
                   If ALL names are numbers (e.g. [2, 3, 4, 80]), they are used
                   as real x positions: spacing is proportional to the values
                   and the axis extends to the max.
                   Otherwise names are placed at equal spacing as labels.
    metric_names : names of the columns of each row, in order
    metrics      : list of entries, each one being:
                   - "y_metric"            -> y_metric per model
                   - ("y_metric", "x")     -> if x is a metric: y vs x scatter,
                                              one point per model (means only);
                                              otherwise: y_metric per model,
                                              x used as the x-axis label
                   None = every metric present in all rows, plotted per model
    title        : optional global figure title
    show_band    : per-model plots: shade the ±std band around the line
    show_points  : per-model plots: if False, draw only the line (+ band),
                   without markers, error bars or value labels
    """
    assert len(results) == len(
        model_names
    ), "results and model_names must have the same length"

    # ---- Normalize cells into (mean, std) ---------------------------------
    def to_cell(c):
        if isinstance(c, (list, tuple, np.ndarray)):
            c = np.asarray(c, dtype=float).ravel()
            assert c.size == 2, f"a cell must be a scalar or (mean, std), got {c}"
            return float(c[0]), abs(float(c[1]))
        return float(c), 0.0

    def to_row(r):
        if isinstance(r, np.ndarray) and r.ndim == 2:  # shape (n_metrics, 2)
            return [to_cell(c) for c in r]
        if isinstance(r, (list, tuple, np.ndarray)):
            return [to_cell(c) for c in r]
        return [to_cell(r)]  # single scalar metric

    rows = [to_row(r) for r in results]
    metric_names = list(metric_names)

    # Columns available in every row
    n_available = min(min(len(r) for r in rows), len(metric_names))
    available = metric_names[:n_available]

    def column(name):
        """Return (means, stds) arrays for one metric across models."""
        i = metric_names.index(name)
        mean = np.array([r[i][0] for r in rows])
        std = np.array([r[i][1] for r in rows])
        return mean, std

    def limits(mean, std):
        lo, hi = (mean - std).min(), (mean + std).max()
        margin = max(0.02, (hi - lo) * 0.3)
        return lo - margin, min(1.02, hi + margin)

    def fmt(m, s):
        return f"{m:.2f}" if s == 0 else f"{m:.2f}\n± {s:.2f}"

    # ---- Numeric model names -> real x positions --------------------------
    numeric_x = all(
        isinstance(n, numbers.Real) and not isinstance(n, bool) for n in model_names
    )
    if numeric_x:
        xpos = np.asarray(model_names, dtype=float)
        order = np.argsort(xpos)  # draw the line from left to right
        all_int = all(float(n).is_integer() for n in model_names)
    else:
        xpos = np.arange(len(model_names), dtype=float)
        order = np.arange(len(model_names))
        all_int = True

    # ---- Normalize metrics into (y, x) pairs ------------------------------
    if metrics is None:
        metrics = available
    elif isinstance(metrics, (str, tuple)):
        metrics = [metrics]
    specs = [(m, None) if isinstance(m, str) else tuple(m) for m in metrics]

    for y, x in specs:
        assert (
            y in available
        ), f"metric '{y}' not available in every row (available: {available})"
        if x in metric_names:
            assert (
                x in available
            ), f"metric '{x}' not available in every row (available: {available})"

    # ---- Figure -----------------------------------------------------------
    n_plots = len(specs)
    ncols = min(ncols, n_plots)
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
    )
    axes = axes.ravel()

    for ax, (y, x) in zip(axes, specs):
        ym, ys = column(y)

        if x in available:
            # Metric vs metric: one point per model (means only)
            xm, _ = column(x)
            ys = np.zeros_like(ym)  # std not shown here
            ax.scatter(xm, ym, s=50, zorder=3)
            for name, xi, yi in zip(model_names, xm, ym):
                ax.annotate(
                    str(name),
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center",
                    fontsize=9,
                )
            ax.set_xlim(*limits(xm, np.zeros_like(xm)))
            ax.set_xlabel(f"{x} (mean)")
        else:
            # Metric per model: line + error bars (+ optional band)
            xs, my, sy = xpos[order], ym[order], ys[order]
            if show_points:
                ax.errorbar(xs, my, yerr=sy, marker="o", lw=2, capsize=4, zorder=3)
            else:
                ax.plot(xs, my, lw=2, zorder=3)
            if show_band and sy.any():
                ax.fill_between(xs, my - sy, my + sy, alpha=0.15, zorder=2)
            for xi, yi, si in zip(xs, my, sy) if show_points else []:
                ax.annotate(
                    fmt(yi, si),
                    (xi, yi + si),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                    fontsize=8,
                )

            if numeric_x:
                # Real numeric axis: automatic ticks, up to the max value
                span = xs.max() - xs.min()
                pad = 0.05 * span if span > 0 else 0.5
                ax.set_xlim(xs.min() - pad, xs.max() + pad)
                ax.xaxis.set_major_locator(MaxNLocator(integer=all_int))
            else:
                ax.set_xticks(
                    xs, labels=[model_names[i] for i in order], rotation=30, ha="right"
                )
                ax.set_xlim(-0.5, len(xs) - 0.5)  # room for edge annotations
            if x:
                ax.set_xlabel(x)

        ax.set_ylabel(f"{y} (mean ± std)" if ys.any() else f"{y} (mean)")
        ax.set_ylim(*limits(ym, ys))
        ax.grid(alpha=0.3)

    for ax in axes[n_plots:]:  # hide empty grid cells
        ax.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if savename is not None:
        fig.savefig(savename, format="eps", bbox_inches="tight")

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

        """
        # skip wrong result
        if model.names[r.probs.top1] != class_name:
            continue
        """

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
