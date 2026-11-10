import base64
import html
import io
import math
import numbers
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.ticker import MaxNLocator, NullFormatter
from matplotlib.transforms import Bbox
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from ultralytics import YOLO

from libs.utils import find_root_repository


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
    logx: bool = False,
    savename: str | None = None,
    xscale: str = "linear",
    xpow: float = 2.0,
    width: float | None = None,
    height: float = 4.0,
    xticks="auto",
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
    logx         : numeric names only: logarithmic x axis. If a name is <= 0
                   (e.g. 0), a symlog axis is used (linear between 0 and 1).
                   0 -> 1 then takes the same width as one decade.
                   Labels follow the same pattern in every decade: 1, 2, 5 x 10^d
                   (only powers of ten beyond 3 decades), with a marked grid line.
                   A light grid line marks every unit of each decade
                   (1..9, 10..90, 100..), at its true log position.
                   Same as xscale="log".
    xscale       : numeric names only: "linear", "log", "sqrt", "square" or "pow".
                   Power scales: the x position is value ** p.
                   - "sqrt"   (p = 0.5): small values get more room.
                   - "square" (p = 2)  : large values get more room (inverse
                     of "sqrt").
                   - "pow"    (p = xpow): any exponent > 0.
                   Labels are round values (multiples of 10, then 1-2-5, ...)
                   spread evenly on screen. The light grid marks each data point.
    xpow         : exponent used by xscale="pow".

    width        : width of each subplot, in inches. A wider (rectangular)
                   plot spreads out points that are close together, e.g. the
                   many small values at the start of a learning curve, while
                   keeping the axis to scale (0-10-20-30 equally spaced).
                   None = automatic: starts at 5 in and widens (up to 12 in)
                   until every value label fits right next to its point.
    height       : height of each subplot, in inches.
    xticks       : numeric names only: values written on the x axis.
                   - "auto" : regular graduations (0, 10, 20, ...)
                   - "data" : the x value of every point (2, 4, 6, ..., 80)
                   - list   : those values, e.g. [2, 10, 22, 48, 80]
                   The grid stays on the regular graduations. Labels stay
                   horizontal.

    Font size: the x graduations, the y graduations and the values written
    on the plot always share ONE size. The biggest size (10 pt down to 3 pt)
    where nothing touches is used: in auto width, the figure is widened
    first, then the size is reduced. Value labels sit as close as possible
    to their point without touching the curve, a point or another label.
    Markers shrink when points are close.
    """
    if logx:
        xscale = "log"
    powers = {"sqrt": 0.5, "square": 2.0, "pow": float(xpow)}
    assert xscale in ("linear", "log", *powers), f"unknown xscale '{xscale}'"
    p = powers.get(xscale)
    if isinstance(xticks, str):
        assert xticks in (
            "auto",
            "data",
        ), f"xticks must be 'auto', 'data' or a list of values, got '{xticks}'"
    assert p is None or p > 0, "xpow must be > 0"
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

    def nice_ticks(lo, hi, fwd, n_target=7):
        """Round values in [lo, hi] spread evenly once transformed by fwd.

        Candidates, from the roundest: multiples of 10^k (k = order of
        magnitude of hi), then the 1-2-5 series (1, 2, 5, 10, 20, 50, ...),
        then multiples of 10^(k-1). A value is kept when far enough, on
        screen, from the values already kept.
        """
        span = fwd(hi) - fwd(lo)
        gap = span / (n_target + 1)
        mag = 10 ** np.floor(np.log10(max(hi, 1e-12)))

        def multiples(s):
            return np.arange(np.ceil(lo / s) * s, hi + 1e-9, s)

        d_lo = int(np.floor(np.log10(max(lo, mag / 1000))))
        series = [
            k * 10.0**d for d in range(d_lo, int(np.log10(mag)) + 1) for k in (1, 2, 5)
        ]
        tiers = [multiples(mag), [v for v in series if lo <= v <= hi]]
        if mag / 10 >= (1 if all_int else 0):
            tiers.append(multiples(mag / 10))
        # Start from the side the scale stretches (where ticks are far apart)
        mid = (lo + hi) / 2
        stretched_right = fwd(mid) - fwd(lo) < fwd(hi) - fwd(mid)
        kept = []
        for tier in tiers:
            for v in sorted(tier, reverse=stretched_right):
                if all(abs(fwd(v) - fwd(k)) >= gap for k in kept):
                    kept.append(float(v))
        return np.array(sorted(kept))

    def log_grid(lo, hi, with_zero=False):
        """Grid values k * 10**d (k = 1..9) inside [lo, hi], plus 0 if asked."""
        start = max(lo, 1) if with_zero else lo
        out = [0.0] if with_zero else []
        for d in range(int(np.floor(np.log10(start))), int(np.ceil(np.log10(hi))) + 1):
            for k in range(1, 10):
                v = k * 10.0**d
                if lo <= v <= hi:
                    out.append(v)
        return np.array(out)

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
        nrows,
        ncols,
        figsize=((width or 5) * ncols, height * nrows),
        squeeze=False,
    )
    axes = axes.ravel()

    fit_jobs = []  # (ax, numeric_ticks, value_annotations) to de-overlap later
    custom_tick_axes = []  # axes with xticks="data" / list: shrink if overlapping

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
            ax.set_xlabel(f"{x}")
        else:
            # Metric per model: line + error bars (+ optional band)
            xs, my, sy = xpos[order], ym[order], ys[order]
            if show_points:
                eb = ax.errorbar(xs, my, yerr=sy, marker="o", lw=2, capsize=4, zorder=3)
                line = eb.lines[0]
            else:
                (line,) = ax.plot(xs, my, lw=2, zorder=3)
            if show_band and sy.any():
                ax.fill_between(xs, my - sy, my + sy, alpha=0.15, zorder=2)
            anns = []
            for xi, yi, si in zip(xs, my, sy) if show_points else []:
                ann = ax.annotate(
                    fmt(yi, si),
                    (xi, yi + si),
                    textcoords="offset points",
                    xytext=(0, 2),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
                ann.pt = (xi, yi)  # the point itself
                ann.high_xy = (xi, yi + si)  # anchor used if placed above
                ann.low_xy = (xi, yi - si)  # anchor used if placed below
                anns.append(ann)
            fit_jobs.append((ax, numeric_x, anns, line, xs, my))

            if numeric_x and p is not None:
                assert xs.min() >= 0, f"xscale='{xscale}' needs values >= 0"

                def fwd(v, p=p):
                    return np.power(np.clip(v, 0, None), p)

                def inv(t, p=p):
                    return np.power(np.clip(t, 0, None), 1 / p)

                ax.set_xscale("function", functions=(fwd, inv))
                t_lo, t_hi = fwd(xs.min()), fwd(xs.max())
                pad = 0.04 * (t_hi - t_lo) if t_hi > t_lo else 0.5
                x_lo, x_hi = inv(max(0.0, t_lo - pad)), inv(t_hi + pad)
                ax.set_xlim(x_lo, x_hi)
                major = nice_ticks(x_lo, x_hi, fwd)
                ax.set_xticks(major, labels=[f"{v:g}" for v in major])
                # Light grid: one line per data point
                ax.set_xticks(np.setdiff1d(np.unique(xs), major), minor=True)
                ax.xaxis.set_minor_formatter(NullFormatter())
                ax.grid(axis="x", which="major", alpha=0.45)
                ax.grid(axis="x", which="minor", alpha=0.15)
                ax.grid(axis="y", which="major", alpha=0.3)
            elif numeric_x and xscale == "log":
                vals = np.unique(xs)
                with_zero = bool(vals.min() <= 0)
                pos = vals[vals > 0]
                if with_zero:  # log(0) undefined -> symlog, [0, 1] as wide as a decade
                    # matplotlib widens the linear part by 1/(1 - 1/base):
                    # linscale=0.9 -> exactly one decade for [0, 1]
                    ax.set_xscale("symlog", linthresh=1, linscale=0.9)
                    x_lo = -0.1
                else:
                    ax.set_xscale("log")
                    x_lo = pos.min() / 1.1
                x_hi = pos.max() * 1.1 if pos.size else 1.1
                ax.set_xlim(x_lo, x_hi)

                # Labels: same pattern in every decade (1-2-5, or only
                # powers of ten when the axis covers many decades)
                n_dec = np.log10(x_hi / max(x_lo, 1 if with_zero else x_lo))
                subs = (1, 2, 5) if n_dec <= 3 else (1,)
                major = log_grid(x_lo, x_hi, with_zero=with_zero)
                major = np.array(
                    [
                        v
                        for v in major
                        if v == 0 or round(v / 10 ** np.floor(np.log10(v))) in subs
                    ]
                )
                # Fine grid: every unit of each decade (1..9, 10..90, ...)
                minor = log_grid(x_lo, x_hi, with_zero=with_zero)
                minor = np.setdiff1d(minor, major)

                ax.set_xticks(major, labels=[f"{v:g}" for v in major])
                ax.set_xticks(minor, minor=True)
                ax.xaxis.set_minor_formatter(NullFormatter())
                ax.grid(axis="x", which="major", alpha=0.45)
                ax.grid(axis="x", which="minor", alpha=0.15)
                ax.grid(axis="y", which="major", alpha=0.3)
            elif numeric_x:
                # Real numeric axis: automatic ticks, up to the max value
                span = xs.max() - xs.min()
                pad = 0.05 * span if span > 0 else 0.5
                ax.set_xlim(xs.min() - pad, xs.max() + pad)
                # Labels: regular graduations (tens, ...) chosen automatically
                ax.xaxis.set_major_locator(MaxNLocator(integer=all_int))
                # Linear grid: one line per graduation, evenly spaced
                ax.grid(axis="both", which="major", alpha=0.3)
            else:
                ax.set_xticks(
                    xs, labels=[model_names[i] for i in order], rotation=30, ha="right"
                )
                ax.set_xlim(-0.5, len(xs) - 0.5)  # room for edge annotations
            if numeric_x and not (isinstance(xticks, str) and xticks == "auto"):
                custom = (
                    np.unique(xs)
                    if isinstance(xticks, str)  # "data"
                    else np.asarray(xticks, dtype=float)
                )
                if xscale == "linear":
                    # Grid keeps the regular graduations (as unlabelled minor ticks)
                    ax.xaxis.set_minor_locator(MaxNLocator(integer=all_int))
                    # matplotlib drops minor ticks that fall on a major one
                    # (e.g. 10 when 10 is a data value): keep them all so the
                    # grid stays evenly spaced
                    ax.xaxis.remove_overlapping_locs = False
                    ax.grid(axis="x", which="major", visible=False)
                    ax.grid(axis="x", which="minor", alpha=0.3)
                    ax.tick_params(axis="x", which="minor", length=0)
                ax.set_xticks(custom, labels=[f"{v:g}" for v in custom])
                custom_tick_axes.append(ax)
            if x:
                ax.set_xlabel(x)

        ax.set_ylabel(f"{y} (mean ± std)" if ys.any() else f"{y}")
        ax.set_ylim(*limits(ym, ys))
        if not (numeric_x and x not in available):  # numeric grid set above
            ax.grid(alpha=0.3)

    for ax in axes[n_plots:]:  # hide empty grid cells
        ax.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14)

    # ---- One font size for x/y graduations and value labels ---------------
    jobs = [j for j in fit_jobs if j[2]]  # axes that carry value labels
    if fit_jobs:
        renderer = fig.canvas.get_renderer()
        px = fig.dpi / 72  # pixels per point
        gap = 1 * px  # min space between two boxes

        def collide(a, b):
            return not (
                a.x1 + gap <= b.x0
                or b.x1 + gap <= a.x0
                or a.y1 + gap <= b.y0
                or b.y1 + gap <= a.y0
            )

        def priority(v, lo, hi):
            if v in (lo, hi) or v == 0:
                return 0  # the ends first
            if float(v).is_integer():
                iv = int(v)
                if iv % 10 == 0:
                    return 1
                if iv & (iv - 1) == 0:  # power of two
                    return 2
                if iv % 5 == 0:
                    return 3
            return 4

        def curve_pixels(ax, anns, xs, my):
            """Pixel samples of the curve and of the error bars, 1 px apart."""

            def seg(a, b):
                n = max(2, int(np.hypot(*(b - a))) + 1)
                return np.linspace(a, b, n)

            pts = ax.transData.transform(np.column_stack([xs, my]))
            parts = [seg(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
            for a in anns:  # vertical error bars
                lo_ = ax.transData.transform(a.low_xy)
                hi_ = ax.transData.transform(a.high_xy)
                parts.append(seg(lo_, hi_))
            return np.vstack(parts) if parts else pts

        line_pad = 1.5 * px  # half the line width + a little air

        def hits(box, pix):
            return np.any(
                (pix[:, 0] > box.x0 - line_pad)
                & (pix[:, 0] < box.x1 + line_pad)
                & (pix[:, 1] > box.y0 - line_pad)
                & (pix[:, 1] < box.y1 + line_pad)
            )

        # Directions around the point, in order of preference:
        # (dx, dy, ha, va, anchor)
        s2 = np.sqrt(0.5)
        directions = [
            (0, 1, "center", "bottom", "high_xy"),  # above
            (0, -1, "center", "top", "low_xy"),  # below
            (-s2, s2, "right", "bottom", "high_xy"),  # above left
            (s2, s2, "left", "bottom", "high_xy"),  # above right
            (-s2, -s2, "right", "top", "low_xy"),  # below left
            (s2, -s2, "left", "top", "low_xy"),  # below right
            (-1, 0, "right", "center", "pt"),  # left
            (1, 0, "left", "center", "pt"),  # right
        ]
        near_extra = 4  # pt: a label at most this far beyond its marker is "near"

        def place(ax, numeric, anns, line, xs, my, size, far):
            """Place every label of one axis.

            Each label goes as close as possible to its point, in any of the
            8 directions around it, without touching the curve, a point or
            another label (and inside the axes when possible). With
            far=False, only positions right next to the point are tried.
            A label is never hidden.
            Returns True when every label fits right next to its point.
            """
            # Markers: shrink them when points are close, so they never touch
            pts = ax.transData.transform(np.column_stack([xs, my]))
            d = np.hypot(*np.diff(pts, axis=0).T).min() if len(pts) > 1 else 1e9
            ms = float(np.clip(0.8 * d / px, 2.5, 6.0))  # diameter in pt
            line.set_markersize(ms)
            r = ms * px / 2

            pix = curve_pixels(ax, anns, xs, my)
            obstacles = []
            for a in anns:
                cx, cy = ax.transData.transform(a.pt)
                obstacles.append(Bbox([[cx - r, cy - r], [cx + r, cy + r]]))

            if numeric:
                xv = [a.pt[0] for a in anns]
                lo, hi = min(xv), max(xv)
                anns = sorted(anns, key=lambda a: (priority(a.pt[0], lo, hi), a.pt[0]))

            frame = ax.get_window_extent(renderer)
            base = r / px + 2.5  # pt: just outside the marker
            dists = np.arange(0, (near_extra if not far else 40) + 0.1, 1.0) + base

            def put(a, dist, d):
                dx, dy, ha, va, anchor = d
                a.xy = getattr(a, anchor)
                a.xyann = (dx * dist, dy * dist)
                a.set_ha(ha)
                a.set_va(va)
                return a.get_window_extent(renderer)

            def inside(box):
                return (
                    box.x0 >= frame.x0
                    and box.x1 <= frame.x1
                    and box.y0 >= frame.y0
                    and box.y1 <= frame.y1
                )

            all_near = True
            for a in anns:
                found = None
                # Closest free position first, inside the axes if possible
                for need_inside in (True, False):
                    for dist in dists:
                        for d in directions:
                            box = put(a, dist, d)
                            if need_inside and not inside(box):
                                continue
                            if any(collide(box, o) for o in obstacles):
                                continue
                            if hits(box, pix):
                                continue
                            found = (dist, d)
                            break
                        if found:
                            break
                    if found:
                        break
                if found is None:  # nothing free: right above, never hidden
                    found = (base, directions[0])
                    all_near = False
                elif found[0] > base + near_extra:
                    all_near = False
                obstacles.append(put(a, *found))
            return all_near

        space = 2 * px  # min gap between two x graduation labels

        def set_size(size):
            """Same font size for the x and y graduations and the values."""
            for ax, *_ in fit_jobs:
                ax.tick_params(axis="both", which="major", labelsize=size)
            for _, _, anns, *_ in jobs:
                for a in anns:
                    a.set_fontsize(size)

        def xticks_fit():
            """True when no two x graduation labels touch (kept horizontal)."""
            for ax, numeric, *_ in fit_jobs:
                if not numeric:  # model names: already tilted
                    continue
                boxes = sorted(
                    (
                        t.get_window_extent(renderer)
                        for t in ax.get_xticklabels()
                        if t.get_text()
                    ),
                    key=lambda bb: bb.x0,
                )
                if any(b2.x0 < b1.x1 + space for b1, b2 in zip(boxes, boxes[1:])):
                    return False
            return True

        def layout(w, size, far=False):
            fig.set_size_inches(w * ncols, height * nrows)
            set_size(size)
            fig.tight_layout()
            fig.canvas.draw()
            ok = xticks_fit()
            for j in jobs:
                ok &= place(*j, size, far)
            return ok

        # Best display: biggest common font first (10 pt = matplotlib's
        # default for graduations), then the narrowest figure (5 in per
        # subplot, up to 12 in) where nothing touches. A width given by the
        # caller is kept as is: only the font adapts.
        widths = [width] if width else [5, 6, 7, 8, 9, 10, 12]
        sizes = [10, 9, 8, 7, 6, 5, 4, 3.5, 3]
        best = next(
            ((w, s) for s in sizes for w in widths if layout(w, s)),
            (widths[-1], sizes[-1]),
        )
        layout(*best, far=True)
        fig.tight_layout()  # labels may stick out of the axes
        fig.canvas.draw()
        for j in jobs:
            place(*j, best[1], far=True)

    if savename is not None:
        fig.savefig(savename, format="eps", bbox_inches="tight")

    plt.show()


def plot_LayerCAM(model: YOLO, data: str, class_name: str):

    predict_folder = f"predict_LayerCAM_{class_name}"
    path_prediction = find_root_repository() / Path("runs", "classify", predict_folder)

    lst = []

    for img in Path(data).iterdir():
        name = f"{Path(img).stem}_cam.jpg"
        _ = model.predict(
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


def plot_LayerCAM_image(model, image, ground_truth: str):

    predict_folder = Path(image).stem
    path_prediction = find_root_repository() / Path("runs", "classify", predict_folder)

    name = f"{Path(image).stem}_cam.jpg"
    r = model.predict(
        image, name=predict_folder, verbose=False, visualize=True, exist_ok=True
    )[0]

    path = Path(path_prediction, name)
    if not path.exists():
        print(f"Failed to predict {path}")
        return

    """
    # skip wrong result
    if model.names[r.probs.top1] != ground_truth:
        print(f"Failed to predict {path}, wrong prediction")
        return
    """

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

    display(Image.fromarray(img_with_border))
