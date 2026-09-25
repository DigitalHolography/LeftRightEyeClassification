#!/usr/bin/env python3
"""
Retrouve, pour chaque image du dataset numéroté
(datasets/classification/{train,valid,test}/{left,right}/0001.png),
l'image originale listée dans un ou plusieurs fichiers texte (pngs.txt...),
puis signale les fuites entre splits.

Rapprochement par contenu, en trois passes de plus en plus tolérantes :
  1. md5 des octets du fichier          -> copie bit à bit
  2. md5 du tableau de pixels décodé    -> ré-encodage différent
  3. dHash 16x16 (distance de Hamming)  -> redimensionnement / recompression,
                                           attribution à vérifier (statut review)

Lecture seule : le dataset, les originales et les listes ne sont jamais modifiés.
Chaque exécution écrit dans un NOUVEAU sous-dossier horodaté du dossier de sortie :
  train.txt, valid.txt, test.txt   originale retenue par image, ordre du dataset
  mapping.csv                      détail : méthode, candidates, statut, patient
  patient_overlap.csv              patients présents dans plusieurs splits
  same_image_across_splits.csv     même image dans plusieurs splits (fuite certaine)
  unmatched.txt, missing_originals.txt, unreadable_originals.txt

Usage (dans un terminal bash, écrire les chemins avec des /) :
    python match_splits.py datasets/classification pngs.txt pngs2.txt
    python match_splits.py dataset/classification_old pngs.txt --dhash-threshold 0

L'identité patient est inférée du nom original (date et côté ignorés,
BL/BL2/AS/apne traités comme suffixes d'acquisition) et reste à valider.
Les identifiants X, XB et XH sont regroupés comme alias possibles.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path, PureWindowsPath

import cv2
import numpy as np

SPLITS = ("train", "valid", "test")
CLASSES = ("left", "right")
EXPECTED_SIDE = {"left": "L", "right": "R"}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Côté dans le nom original : _L_, _R_, _L2_, _R1_..., aussi en fin de nom.
SIDE_PATTERN = re.compile(r"_([LR])\d*(?=_|$)")
PATIENT_PATTERN = re.compile(
    r"^(?P<patient>[A-Z]+(?:_?\d{3,})?)(?P<variant>[BH]?)(?=_+"
    r"(?:[LR]\d*|HD|BL\d*|AS|APNE|\d+)_)",
    re.IGNORECASE,
)

MAPPING_FIELDS = [
    "split",
    "class",
    "numbered",
    "original",
    "method",
    "score",
    "candidate_count",
    "alternatives",
    "reused",
    "label_check",
    "patient_id",
    "patient_note",
    "source_lists",
    "status",
    "review_reason",
]
SHARED_FIELDS = ["group", "split", "class", "numbered", "original"]
OVERLAP_FIELDS = (
    ["patient_id", "status", "splits", "note"]
    + [f"{s}_images" for s in SPLITS]
    + [f"{s}_originals" for s in SPLITS]
)


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


def pixel_md5(img: np.ndarray) -> str:
    """md5 des pixels décodés, forme et type compris : insensible au ré-encodage."""
    header = f"{img.shape}:{img.dtype}:".encode("ascii")
    return hashlib.md5(header + np.ascontiguousarray(img).tobytes()).hexdigest()


def dhash(img: np.ndarray, size: int = 16) -> int:
    """Hash perceptuel (difference hash) sur size*size bits."""
    if img.ndim == 3:
        img = (
            cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
            if img.shape[2] >= 3
            else img[:, :, 0]
        )
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    small = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def decoded_keys(path: Path, want_dhash: bool) -> tuple[str | None, int | None]:
    """md5 des pixels et dHash calculés sur un seul décodage."""
    img = _imread(path)
    if img is None:
        return None, None
    try:
        value = dhash(img) if want_dhash else None
    except cv2.error:
        value = None
    return pixel_md5(img), value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# --------------------------------------------------------------------------- #
# Identité patient et côté
# --------------------------------------------------------------------------- #
def patient_info(path: Path) -> tuple[str, str]:
    """Identifiant inféré et réserves éventuelles ; ne déduit pas d'alias certains."""
    name = PureWindowsPath(str(path)).stem.upper()
    name = re.sub(r"^\d{6}_", "", name)
    iredo = re.match(r"IREDO\s*-?\s*(\d+)(?=[\s_-])", name)
    if iredo:
        # Le numéro 40 apparaît avec HA et VS : ne pas certifier ces identités.
        return f"IREDO{iredo[1]}", "Identité IREDO à valider (numéros parfois ambigus)"
    notes = []
    if re.match(r"^\d{6}[A-Z]", name):
        name = name[6:]
        notes.append("Date accolée à l'identifiant retirée : alias à valider")
    if name.startswith("JUSTINEPRESSURE_"):
        notes.append("Nom de protocole : identité patient à valider")
        return "JUSTINEPRESSURE", " ; ".join(notes)
    match = PATIENT_PATTERN.match(name)
    if not match:
        notes.append("Format d'identifiant non reconnu")
        return "", " ; ".join(notes)
    if match["variant"]:
        notes.append("Suffixe b/h retiré : alias patient à valider")
    return match["patient"].replace("_", ""), " ; ".join(notes)


def alias_base(patient_id: str) -> str | None:
    """XB ou XH peut être un alias de X (suffixe non retiré sans chiffres)."""
    if len(patient_id) > 1 and patient_id.isalpha() and patient_id[-1] in "BH":
        return patient_id[:-1]
    return None


def alias_groups(patient_ids) -> list[list[str]]:
    """Regroupe X, XB et XH (y compris XB et XH sans X) par union-find."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ids = list(patient_ids)
    for pid in ids:
        base = alias_base(pid)
        root = find(pid)
        if base is not None:
            parent[root] = find(base)
    groups = defaultdict(list)
    for pid in ids:
        groups[find(pid)].append(pid)
    return sorted(sorted(group) for group in groups.values())


def side_check(original: Path, cls: str) -> str:
    """ok, MISMATCH, conflit (L et R dans le nom) ou vide (côté illisible)."""
    sides = set(SIDE_PATTERN.findall(PureWindowsPath(str(original)).stem.upper()))
    if not sides:
        return ""
    if len(sides) > 1:
        return "conflit"
    return "ok" if sides == {EXPECTED_SIDE[cls]} else "MISMATCH"


# --------------------------------------------------------------------------- #
# Collecte
# --------------------------------------------------------------------------- #
def path_key(path: Path) -> str:
    """Clé insensible à la casse (chemins Windows)."""
    return str(path).lower()


def read_original_paths(list_files: list[Path]):
    """Lit les listes en UTF-8 strict ; doublons ignorés sans tenir compte de la casse."""
    paths, missing = [], []
    sources: dict[str, list[str]] = defaultdict(list)
    for list_file in list_files:
        try:
            lines = list_file.read_text(encoding="utf-8-sig").splitlines()
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{list_file} n'est pas en UTF-8 : le réenregistrer en "
                "UTF-8 (sinon des chemins accentués seraient perdus)."
            ) from exc
        for line in lines:
            raw = line.strip().strip('"')
            if not raw:
                continue
            path = Path(raw)
            key = path_key(path)
            first = key not in sources
            sources[key].append(str(list_file))
            if not first:
                continue
            if path.is_file():
                paths.append(path)
            else:
                missing.append(raw)
    return paths, missing, sources


def collect_dataset_images(dataset_dir: Path):
    """Retourne [(chemin, split, classe)] et les anomalies de structure."""
    if not dataset_dir.is_dir():
        raise ValueError(f"Dossier du dataset introuvable : {dataset_dir}")
    items, warnings = [], []
    for split in SPLITS:
        split_dir = dataset_dir / split
        if not split_dir.is_dir():
            warnings.append(f"Split absent, non audité : {split_dir}")
            continue
        extra_dirs = [
            p.name
            for p in split_dir.iterdir()
            if p.is_dir() and p.name.lower() not in CLASSES
        ]
        stray = [
            p
            for p in split_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ]
        if extra_dirs:
            warnings.append(
                f"Dossiers non audités dans {split_dir} : "
                f"{', '.join(sorted(extra_dirs))}"
            )
        if stray:
            warnings.append(
                f"{len(stray)} image(s) hors left/right, non auditée(s), "
                f"dans {split_dir}"
            )
        for cls in CLASSES:
            folder = split_dir / cls
            if not folder.is_dir():
                warnings.append(f"Dossier absent, non audité : {folder}")
                continue
            if any(p.is_dir() for p in folder.iterdir()):
                warnings.append(f"Sous-dossiers ignorés dans {folder}")
            images = sorted(
                p
                for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in IMG_EXTS
            )
            items.extend((p, split, cls) for p in images)
    if not items:
        raise ValueError(
            f"Aucune image trouvée dans {dataset_dir}/"
            f"{{{','.join(SPLITS)}}}/{{{','.join(CLASSES)}}}. "
            "Vérifier le chemin (avec des / dans bash)."
        )
    return items, warnings


def create_run_dir(out_dir: Path, dataset_dir: Path) -> Path:
    """Crée un dossier neuf : aucun fichier existant ne peut être écrasé."""
    output, dataset = out_dir.resolve(), dataset_dir.resolve()
    if output == dataset or dataset in output.parents:
        raise ValueError("Le dossier de sortie doit être en dehors du dataset.")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for n in range(1, 100):
        run_dir = out_dir / (stamp if n == 1 else f"{stamp}_{n}")
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise ValueError(f"Impossible de créer un dossier de sortie neuf dans {out_dir}")


def write_lines(path: Path, lines) -> None:
    lines = list(lines)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def progress(label: str, n: int, total: int) -> None:
    if n % 200 == 0 or n == total:
        print(f"    {label} : {n}/{total}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Rapprochement
# --------------------------------------------------------------------------- #
def choose(
    candidates: list[Path], distances: dict, used: set[str]
) -> tuple[Path, bool]:
    """Meilleure candidate, en préférant une originale pas encore attribuée."""
    best = min(distances.values()) if distances else None
    pool = [p for p in candidates if best is None or distances[p] == best]
    free = [p for p in pool if path_key(p) not in used]
    chosen = (free or pool)[0]
    used.add(path_key(chosen))
    return chosen, not free


def match_dataset(
    dataset_dir: Path, list_files: list[Path], out_dir: Path, dhash_threshold: int = 6
) -> None:
    # Toutes les vérifications d'entrée ont lieu avant la moindre écriture.
    dataset, warnings = collect_dataset_images(dataset_dir)
    originals, missing, sources = read_original_paths(list_files)
    if not originals:
        raise ValueError(
            "Aucune originale accessible. Vérifier les listes et "
            "la connexion au NAS. Aucun audit effectué."
        )
    run_dir = create_run_dir(out_dir, dataset_dir)
    want_dhash = dhash_threshold > 0

    for warning in warnings:
        print(f"[!] {warning}")
    print(
        f"\n{len(originals)} originales accessibles, {len(missing)} chemins absents, "
        f"{len(dataset)} images dans le dataset numéroté.\n"
    )

    print("Indexation des originales (un seul décodage par image)...")
    idx_bytes, idx_pixels = defaultdict(list), defaultdict(list)
    pixel_of: dict[str, str] = {}
    perceptual, unreadable = [], []
    for n, path in enumerate(originals, 1):
        byte_key = file_md5(path)
        pixel_key, value = decoded_keys(path, want_dhash)
        if byte_key is not None:
            idx_bytes[byte_key].append(path)
        if pixel_key is not None:
            idx_pixels[pixel_key].append(path)
            pixel_of[path_key(path)] = pixel_key
        if value is not None:
            perceptual.append((value, path))
        if byte_key is None or pixel_key is None:
            unreadable.append(str(path))
        progress("originales", n, len(originals))

    print("\nRapprochement...")
    used: set[str] = set()
    rows, unmatched = [], []
    for n, (img, split, cls) in enumerate(dataset, 1):
        progress("dataset", n, len(dataset))
        method, candidates, distances = "", [], {}
        byte_key = file_md5(img)
        if byte_key in idx_bytes:
            method, candidates = "md5_octets", idx_bytes[byte_key]
        else:
            pixel_key, value = decoded_keys(img, want_dhash)
            if pixel_key is None:
                unmatched.append(f"{img}\t{split}\t{cls}\timage illisible")
                continue
            if pixel_key in idx_pixels:
                method, candidates = "md5_pixels", idx_pixels[pixel_key]
            elif value is not None:
                # Toutes les candidates sous le seuil sont conservées.
                distances = {
                    p: d
                    for h, p in perceptual
                    if (d := hamming(value, h)) <= dhash_threshold
                }
                candidates = sorted(distances, key=lambda p: (distances[p], str(p)))
                method = "dhash"
        if not candidates:
            unmatched.append(f"{img}\t{split}\t{cls}\taucune correspondance")
            continue

        chosen, reused = choose(candidates, distances, used)
        info = {p: patient_info(p) for p in candidates}
        identities = {identity for identity, _ in info.values()}
        reasons = []
        if method == "dhash":
            reasons.append("appariement perceptuel (dHash)")
        if len(identities) > 1:
            reasons.append("candidates de patients différents")
        if "" in identities:
            reasons.append("identifiant non reconnu")
        if any(note for identity, note in info.values() if identity):
            reasons.append("réserve sur l'identité")
        patient_id, patient_note = info[chosen]
        by_id = defaultdict(list)
        for p in candidates:
            by_id[info[p][0]].append(str(p))
        content = (
            None
            if method == "dhash"
            else pixel_of.get(path_key(chosen)) or f"octets:{byte_key}"
        )
        rows.append(
            {
                "split": split,
                "class": cls,
                "numbered": str(img),
                "original": str(chosen),
                "method": method,
                "score": distances.get(chosen, ""),
                "candidate_count": len(candidates),
                "alternatives": "\n".join(str(p) for p in candidates if p != chosen),
                "reused": "yes" if reused else "",
                "label_check": side_check(chosen, cls),
                "patient_id": patient_id,
                "patient_note": patient_note,
                "source_lists": "\n".join(sorted(set(sources[path_key(chosen)]))),
                "status": "review" if reasons else "exact",
                "review_reason": " ; ".join(reasons),
                "_by_id": by_id,
                "_content": content,
            }
        )

    # --------------------------------------------------------------- fuites
    # Même image (octets ou pixels identiques) dans plusieurs splits.
    by_content = defaultdict(list)
    for row in rows:
        if row["_content"]:
            by_content[row["_content"]].append(row)
    shared, group_no = [], 0
    for key in sorted(by_content):
        group = by_content[key]
        if len({r["split"] for r in group}) < 2:
            continue
        group_no += 1
        shared.extend({"group": group_no, **r} for r in group)

    # Même patient (identifiant inféré, alias b/h regroupés) dans plusieurs splits.
    by_patient = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for pid in row["_by_id"]:
            if pid:
                by_patient[pid][row["split"]].append(row)

    def confirmed_splits(pid: str) -> set[str]:
        return {
            s
            for s, rs in by_patient[pid].items()
            if any(r["status"] == "exact" for r in rs)
        }

    overlaps = []
    for ids in alias_groups(by_patient):
        present = [s for s in SPLITS if any(by_patient[pid].get(s) for pid in ids)]
        if len(present) < 2:
            continue
        # Exact seulement si un même identifiant est confirmé dans deux splits.
        is_exact = any(len(confirmed_splits(pid)) >= 2 for pid in ids)
        entry = {
            "patient_id": " / ".join(ids),
            "status": "exact_overlap" if is_exact else "possible_overlap",
            "splits": "+".join(present),
            "note": (
                f"Alias possibles (suffixe b/h) à valider : {', '.join(ids)}"
                if len(ids) > 1
                else ""
            ),
        }
        for s in SPLITS:
            pairs = [(r, pid) for pid in ids for r in by_patient[pid].get(s, [])]
            entry[f"{s}_images"] = "\n".join(sorted({r["numbered"] for r, _ in pairs}))
            entry[f"{s}_originals"] = "\n".join(
                sorted({o for r, pid in pairs for o in r["_by_id"][pid]})
            )
        overlaps.append(entry)

    # ------------------------------------------------------------- sorties
    for split in SPLITS:
        write_lines(
            run_dir / f"{split}.txt",
            [r["original"] for r in rows if r["split"] == split],
        )
    write_csv(run_dir / "mapping.csv", MAPPING_FIELDS, rows)
    write_csv(run_dir / "patient_overlap.csv", OVERLAP_FIELDS, overlaps)
    write_csv(run_dir / "same_image_across_splits.csv", SHARED_FIELDS, shared)
    write_lines(run_dir / "unmatched.txt", unmatched)
    write_lines(run_dir / "missing_originals.txt", missing)
    write_lines(run_dir / "unreadable_originals.txt", unreadable)

    # -------------------------------------------------------------- résumé
    print("\n=== Résumé ===")
    for split in SPLITS:
        sub = [r for r in rows if r["split"] == split]
        left = sum(r["class"] == "left" for r in sub)
        print(
            f"  {split:<6} : {len(sub):>5} images  (left {left}, right {len(sub) - left})"
        )
    print(f"  {'TOTAL':<6} : {len(rows):>5} appariées / {len(dataset)}")
    by_method = defaultdict(int)
    for r in rows:
        by_method[r["method"]] += 1
    print("\n  Méthodes :", dict(by_method))

    review = [r for r in rows if r["status"] == "review"]
    multi = sum(r["candidate_count"] > 1 for r in rows)
    reused = sum(r["reused"] == "yes" for r in rows)
    if review:
        print(
            f"  [!] {len(review)} appariement(s) à vérifier (status=review, "
            "voir review_reason). Ils figurent aussi dans train/valid/test.txt."
        )
    if multi:
        print(
            f"  [!] {multi} image(s) avec plusieurs originales candidates "
            "(voir alternatives)."
        )
    if reused:
        print(f"  [!] {reused} image(s) appariée(s) à une originale déjà attribuée.")

    bad = [r for r in rows if r["label_check"] == "MISMATCH"]
    if bad:
        print(
            f"  [!] {len(bad)} image(s) dont le dossier left/right contredit "
            "le côté du nom original :"
        )
        for r in bad[:10]:
            print(f"      {r['numbered']}  ->  {r['original']}")
        if len(bad) > 10:
            print(f"      ... et {len(bad) - 10} autre(s), voir mapping.csv")
    conflicts = sum(r["label_check"] == "conflit" for r in rows)
    unchecked = sum(r["label_check"] == "" for r in rows)
    if conflicts:
        print(f"  [!] {conflicts} nom(s) original(aux) contenant à la fois L et R.")
    if unchecked:
        print(
            f"  [!] {unchecked} image(s) sans côté lisible dans le nom original : "
            "left/right non vérifié."
        )

    exact = sum(o["status"] == "exact_overlap" for o in overlaps)
    print(
        f"\n  Fuites : {group_no} image(s) présente(s) dans plusieurs splits ; "
        f"patients communs {exact} exacts, {len(overlaps) - exact} possibles."
    )

    if unmatched:
        print(f"  [!] {len(unmatched)} image(s) non appariée(s) -> unmatched.txt")
        if want_dhash:
            print("      (vérifier missing_originals.txt, ou ajouter une autre liste)")
        else:
            print("      (dHash désactivé : essayer --dhash-threshold 6)")
    if missing:
        print(
            f"  [!] {len(missing)} chemin(s) des listes absent(s) du disque "
            "-> missing_originals.txt"
        )
    if unreadable:
        print(
            f"  [!] {len(unreadable)} originale(s) illisible(s) "
            "-> unreadable_originals.txt"
        )
    if warnings:
        print(f"  [!] {len(warnings)} anomalie(s) de structure du dataset :")
        for warning in warnings:
            print(f"      {warning}")
    if warnings or unmatched or missing or unreadable or review:
        print("  [!] Audit incomplet : consulter les rapports avant toute conclusion.")
    print(
        "  L'absence de fuite détectée ne la garantit pas : les identifiants "
        "patient sont inférés des noms et doivent être validés."
    )
    print(f"\nFichiers écrits dans : {run_dir.resolve()}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "dataset",
        type=Path,
        help="dossier contenant train/, valid/, test/ " "(ex: datasets/classification)",
    )
    ap.add_argument(
        "lists",
        type=Path,
        nargs="+",
        help="fichier(s) .txt listant les chemins originaux "
        "(ex: pngs.txt pngs2.txt)",
    )
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("splits_out"),
        help="dossier parent (défaut: splits_out) ; chaque exécution "
        "y crée un sous-dossier horodaté",
    )
    ap.add_argument(
        "--dhash-threshold",
        type=int,
        default=6,
        help="distance de Hamming max pour la passe perceptuelle "
        "(défaut: 6 ; 0 pour la désactiver)",
    )
    args = ap.parse_args()
    if not 0 <= args.dhash_threshold <= 256:
        ap.error("--dhash-threshold doit être compris entre 0 et 256.")
    try:
        match_dataset(args.dataset, args.lists, args.out, args.dhash_threshold)
    except (OSError, ValueError) as exc:
        ap.exit(1, f"Erreur : {exc}\n")


if __name__ == "__main__":
    main()
