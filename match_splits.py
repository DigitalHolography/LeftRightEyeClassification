#!/usr/bin/env python3
"""
Retrouve, pour chaque image du dataset numéroté (datasets/classification/
{train,valid,test}/{left,right}/0001.png), le chemin de l'image originale
listée dans un fichier texte (pngs.txt).

Le rapprochement se fait par contenu, en trois passes de plus en plus tolérantes :
  1. md5 des octets du fichier          -> copie bit à bit (shutil.copyfile)
  2. md5 du tableau de pixels décodé    -> ré-encodage PNG différent
  3. dHash 16x16 (distance de Hamming)  -> redimensionnement / recompression

Sortie : train.txt, valid.txt, test.txt (chemins originaux, un par ligne)
         + mapping.csv (détail complet) + unmatched.txt
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

SPLITS = ("train", "valid", "test")
CLASSES = ("left", "right")
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# _L_ ou _R_ dans le nom de fichier original, pour le contrôle de cohérence
EYE_PATTERN = re.compile(r"_(L|R)_", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Empreintes
# --------------------------------------------------------------------------- #
def file_md5(path: Path, chunk: int = 1 << 20) -> str | None:
    """md5 des octets bruts du fichier."""
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            while block := f.read(chunk):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def _imread(path: Path) -> np.ndarray | None:
    """Lecture robuste (gère les chemins non-ASCII sous Windows)."""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None


def pixel_md5(path: Path) -> str | None:
    """md5 du tableau de pixels décodé : insensible au ré-encodage."""
    img = _imread(path)
    if img is None:
        return None
    return hashlib.md5(np.ascontiguousarray(img).tobytes()).hexdigest()


def dhash(path: Path, size: int = 16) -> int | None:
    """
    Hash perceptuel (difference hash) sur size*(size+1) bits.
    Insensible au redimensionnement et à la compression légère.
    """
    img = _imread(path)
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    small = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# --------------------------------------------------------------------------- #
# Collecte
# --------------------------------------------------------------------------- #
def read_original_paths(list_files: list[Path]) -> list[Path]:
    """Lit un ou plusieurs fichiers .txt de chemins, ignore vides/doublons/absents."""
    seen: set[str] = set()
    paths: list[Path] = []
    missing = 0
    for lf in list_files:
        with open(lf, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                raw = line.strip().strip('"')
                if not raw:
                    continue
                p = Path(raw)
                key = str(p).lower()
                if key in seen:
                    continue
                seen.add(key)
                if not p.is_file():
                    missing += 1
                    continue
                paths.append(p)
    if missing:
        print(f"[!] {missing} chemin(s) de la liste introuvable(s) sur le disque.")
    return paths


def collect_dataset_images(dataset_dir: Path) -> list[tuple[Path, str, str]]:
    """Retourne [(chemin, split, classe), ...] pour le dataset numéroté."""
    items = []
    for split in SPLITS:
        for cls in CLASSES:
            folder = dataset_dir / split / cls
            if not folder.is_dir():
                print(f"[!] Dossier absent : {folder}")
                continue
            for p in sorted(folder.iterdir()):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    items.append((p, split, cls))
    return items


# --------------------------------------------------------------------------- #
# Rapprochement
# --------------------------------------------------------------------------- #
def build_index(originals: list[Path], fn, label: str) -> dict:
    """Construit empreinte -> [chemins originaux] (plusieurs si doublons)."""
    index: dict = defaultdict(list)
    for i, p in enumerate(originals, 1):
        if i % 200 == 0:
            print(f"    {label}: {i}/{len(originals)}", end="\r", file=sys.stderr)
        key = fn(p)
        if key is not None:
            index[key].append(p)
    print(f"    {label}: {len(originals)}/{len(originals)} indexées.", file=sys.stderr)
    return index


def match_dataset(
    dataset_dir: Path,
    list_files: list[Path],
    out_dir: Path,
    dhash_threshold: int = 6,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    originals = read_original_paths(list_files)
    dataset = collect_dataset_images(dataset_dir)
    print(f"\n{len(originals)} images originales lisibles, "
          f"{len(dataset)} images dans le dataset numéroté.\n")

    print("Indexation des originales...")
    idx_bytes = build_index(originals, file_md5, "md5 octets")
    idx_pixels = build_index(originals, pixel_md5, "md5 pixels")
    print("    dHash...", file=sys.stderr)
    dhash_list = [(dhash(p), p) for p in originals]
    dhash_list = [(h, p) for h, p in dhash_list if h is not None]

    used: set[str] = set()          # originales déjà attribuées
    rows: list[dict] = []
    unmatched: list[tuple[Path, str, str]] = []

    def take(candidates: list[Path]) -> tuple[Path | None, bool]:
        """Prend un candidat non encore utilisé. Retourne (chemin, ambigu?)."""
        free = [c for c in candidates if str(c) not in used]
        if not free:
            # tous déjà pris : on réutilise le premier (doublon exact du dataset)
            return (candidates[0] if candidates else None, len(candidates) > 1)
        used.add(str(free[0]))
        return free[0], len(candidates) > 1

    print("\nRapprochement...")
    for n, (img, split, cls) in enumerate(dataset, 1):
        if n % 100 == 0:
            print(f"    {n}/{len(dataset)}", end="\r", file=sys.stderr)

        original, method, ambiguous, score = None, None, False, ""

        key = file_md5(img)
        if key and key in idx_bytes:
            original, ambiguous = take(idx_bytes[key])
            method = "md5_octets"

        if original is None:
            key = pixel_md5(img)
            if key and key in idx_pixels:
                original, ambiguous = take(idx_pixels[key])
                method = "md5_pixels"

        if original is None:
            h = dhash(img)
            if h is not None and dhash_list:
                dists = [(hamming(h, oh), op) for oh, op in dhash_list]
                dists.sort(key=lambda t: t[0])
                best_d, best_p = dists[0]
                if best_d <= dhash_threshold:
                    # ambigu si un 2e candidat est aussi proche
                    ambiguous = len(dists) > 1 and dists[1][0] <= best_d + 1
                    original, _ = take([best_p])
                    method = "dhash"
                    score = str(best_d)

        if original is None:
            unmatched.append((img, split, cls))
            continue

        # contrôle de cohérence label dossier <-> _L_/_R_ du nom original
        m = EYE_PATTERN.search(original.name)
        side_in_name = m.group(1).upper() if m else ""
        expected = {"left": "L", "right": "R"}[cls]
        coherent = "" if not side_in_name else ("ok" if side_in_name == expected else "MISMATCH")

        rows.append({
            "split": split,
            "class": cls,
            "numbered": str(img),
            "original": str(original),
            "method": method,
            "score": score,
            "ambiguous": "yes" if ambiguous else "",
            "label_check": coherent,
        })

    # ----------------------------------------------------------------- sorties
    for split in SPLITS:
        lines = [r["original"] for r in rows if r["split"] == split]
        (out_dir / f"{split}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

    with open(out_dir / "mapping.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["split", "class", "numbered", "original",
                            "method", "score", "ambiguous", "label_check"])
        w.writeheader()
        w.writerows(rows)

    if unmatched:
        (out_dir / "unmatched.txt").write_text(
            "\n".join(f"{p}\t{s}\t{c}" for p, s, c in unmatched) + "\n",
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ résumé
    print("\n\n=== Résumé ===")
    for split in SPLITS:
        sub = [r for r in rows if r["split"] == split]
        left = sum(1 for r in sub if r["class"] == "left")
        right = sum(1 for r in sub if r["class"] == "right")
        print(f"  {split:<6} : {len(sub):>5} images  (left {left}, right {right})")
    print(f"  {'TOTAL':<6} : {len(rows):>5} appariées / {len(dataset)}")

    by_method = defaultdict(int)
    for r in rows:
        by_method[r["method"]] += 1
    print("\n  Méthodes :", dict(by_method))

    amb = [r for r in rows if r["ambiguous"]]
    if amb:
        print(f"  [!] {len(amb)} appariement(s) ambigu(s) "
              f"(plusieurs originales identiques) — voir colonne 'ambiguous'.")

    bad = [r for r in rows if r["label_check"] == "MISMATCH"]
    if bad:
        print(f"  [!] {len(bad)} image(s) dont le dossier left/right contredit "
              f"le _L_/_R_ du nom original :")
        for r in bad[:10]:
            print(f"      {r['numbered']}  ->  {r['original']}")
        if len(bad) > 10:
            print(f"      ... et {len(bad) - 10} autre(s), voir mapping.csv")

    if unmatched:
        print(f"  [!] {len(unmatched)} image(s) non appariée(s) -> unmatched.txt")
        print("      (essaie d'augmenter --dhash-threshold, ou ajoute pngs2.txt)")

    print(f"\nFichiers écrits dans : {out_dir.resolve()}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Retrouve les chemins originaux des images du dataset "
                    "numéroté, par split (train/valid/test)."
    )
    ap.add_argument("dataset", type=Path,
                    help="dossier contenant train/, valid/, test/ "
                         "(ex: datasets/classification)")
    ap.add_argument("lists", type=Path, nargs="+",
                    help="fichier(s) .txt listant les chemins originaux "
                         "(ex: pngs.txt pngs2.txt)")
    ap.add_argument("-o", "--out", type=Path, default=Path("splits_out"),
                    help="dossier de sortie (défaut: splits_out)")
    ap.add_argument("--dhash-threshold", type=int, default=6,
                    help="distance de Hamming max pour la passe perceptuelle "
                         "(défaut: 6 ; mets 0 pour la désactiver)")
    args = ap.parse_args()

    match_dataset(args.dataset, args.lists, args.out, args.dhash_threshold)


if __name__ == "__main__":
    main()
