"""Paint the clusters back onto the assemblies, and score whether they transfer.

Two jobs, one honest and one cosmetic.

The cosmetic one is :func:`write_cluster_beds`: one BED9 per assembly, in that
assembly's own contig coordinates, so a cluster found in a 464-haplotype k-mer
matrix can be dragged into IGV or a genome browser next to the assembly it came
from.  Colours are stable per cluster id -- the same cluster is the same colour
in every file of a run -- because the first thing anyone does is open two
assemblies side by side.

The honest one is :func:`cluster_transfer_report`.  The clusters were learned
with no alignment and no reference: bins from CHM13 and bins from 464 HPRC
haplotypes went into the same matrix as anonymous bags of k-mers.  The question
that makes the whole exercise worth doing is whether a cluster whose *reference*
bins are, say, active alpha-satellite HOR also lands on active HOR in the
assemblies -- annotated independently, by a different pipeline, in different
coordinates.  ``asm_agreement`` is that number.  It is deliberately computed
only over *annotated* assembly bins: an assembly with no cenSat track is
evidence about nothing, and folding it in as a miss would flatter or punish the
result at random.  Clusters with no reference bins at all are reported with
``asm_agreement = NaN`` rather than 0 -- there is nothing to transfer from.
"""

from __future__ import annotations

import colorsys
import gzip
import io
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from kmer_dust import schemas
from kmer_dust.config import Config
from kmer_dust.enrich import NOISE_CLUSTER, NOISE_NAME
from kmer_dust.log import get_logger

logger = get_logger(__name__)

__all__ = [
    "TRANSFER_COLUMNS",
    "NOISE_RGB",
    "cluster_colors",
    "write_cluster_beds",
    "cluster_transfer_report",
    "load_transfer_report",
]

#: Grey, and deliberately dull: noise should recede in a browser window.
NOISE_RGB = "190,190,190"

#: Contract for :func:`cluster_transfer_report`.
TRANSFER_COLUMNS: dict[str, str] = {
    "cluster": "int32",
    "name": "string",
    "n_ref_bins": "int64",
    "n_asm_bins": "int64",
    "ref_top_feature": "string",
    "asm_top_feature": "string",
    "asm_agreement": "float64",
    "asm_annotated_frac": "float64",
}

#: Golden-ratio conjugate: successive hues are maximally far apart, so adjacent
#: cluster ids never get near-identical colours however many clusters there are.
_PHI = 0.6180339887498949


def cluster_colors(cluster_ids: Iterable[int], seed: int = 7) -> dict[int, str]:
    """Stable ``cluster -> "R,G,B"`` palette.

    The hue of cluster ``i`` depends only on ``i`` and ``seed``, so re-running
    with more or fewer clusters does not reshuffle the ones that survived, and
    two runs of the same config produce byte-identical BEDs.
    """
    offset = float(np.random.default_rng(int(seed)).random())
    colors: dict[int, str] = {}
    for cid in cluster_ids:
        cid = int(cid)
        if cid < 0:
            colors[cid] = NOISE_RGB
            continue
        hue = (offset + cid * _PHI) % 1.0
        # Alternate value/saturation so that even a long hue walk stays legible.
        value = 0.95 if cid % 2 == 0 else 0.75
        saturation = 0.85 if cid % 3 else 0.60
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors[cid] = f"{int(round(r * 255))},{int(round(g * 255))},{int(round(b * 255))}"
    return colors


def _bed_frame(
    rows: pd.DataFrame, clusters: pd.DataFrame, names: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Join bins, labels and names into the columns a BED9 line needs."""
    meta = rows[["bin_uid", "assembly", "contig", "start", "end"]].copy()
    meta["bin_uid"] = meta["bin_uid"].astype("string")
    lab = clusters[["bin_uid", "cluster", "probability"]].copy()
    lab["bin_uid"] = lab["bin_uid"].astype("string")
    frame = lab.merge(meta, on="bin_uid", how="inner")
    if frame.empty:
        return frame

    if names is not None and len(names):
        lookup = dict(
            zip(names["cluster"].astype(int), names["name"].astype("string").fillna(""))
        )
    else:
        lookup = {}
    ids = frame["cluster"].astype(int)
    frame["name"] = [
        lookup.get(cid) or (NOISE_NAME if cid == NOISE_CLUSTER else f"C{cid}") for cid in ids
    ]
    prob = pd.to_numeric(frame["probability"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    frame["score"] = np.clip(np.rint(prob * 1000.0), 0, 1000).astype(np.int64)
    palette = cluster_colors(sorted(set(ids)), seed=cfg.seed)
    frame["rgb"] = [palette[cid] for cid in ids]
    frame["assembly"] = frame["assembly"].astype("string")
    frame["contig"] = frame["contig"].astype("string")
    return frame.sort_values(["assembly", "contig", "start", "end"], kind="stable").reset_index(
        drop=True
    )


def _bed_lines(frame: pd.DataFrame) -> list[str]:
    contig = frame["contig"].astype(str).to_numpy()
    start = frame["start"].to_numpy(dtype=np.int64)
    end = frame["end"].to_numpy(dtype=np.int64)
    name = frame["name"].astype(str).to_numpy()
    score = frame["score"].to_numpy(dtype=np.int64)
    rgb = frame["rgb"].astype(str).to_numpy()
    return [
        f"{contig[i]}\t{start[i]}\t{end[i]}\t{name[i]}\t{score[i]}\t."
        f"\t{start[i]}\t{end[i]}\t{rgb[i]}\n"
        for i in range(len(frame))
    ]


def _track_line(cfg: Config, label: str) -> str:
    run = str(cfg.run_name or "kmer-dust")
    return (
        f'track name="{run} {label}" '
        f'description="kmer-dust cluster assignments ({run}, {label})" '
        'itemRgb="On" visibility=1\n'
    )


def _write_gzip(path: Path, lines: Iterable[str]) -> None:
    """Write ``lines`` bgzip-compressed via pysam, or plain gzip if pysam cannot.

    bgzip is worth the try because it makes the file tabix-indexable.  The gzip
    fallback pins ``mtime=0`` and writes no filename into the header: a
    timestamp there would make two runs of the same config differ byte for byte.
    """
    tmp = path.with_suffix(".gz.tmp")
    try:
        import pysam

        handle = pysam.BGZFile(str(tmp), "wb")
    except Exception as exc:  # noqa: BLE001 - pysam is optional here, gzip always works
        logger.debug("bgzip unavailable (%s); falling back to gzip", exc)
        raw = open(tmp, "wb")
        try:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
                for line in lines:
                    gz.write(line.encode("utf-8"))
        finally:
            raw.close()
    else:
        try:
            for line in lines:
                handle.write(line.encode("utf-8"))
        finally:
            handle.close()
    tmp.replace(path)


def write_cluster_beds(
    rows: pd.DataFrame,
    clusters: pd.DataFrame,
    names: pd.DataFrame,
    cfg: Config,
    outdir: Path,
    *,
    force: bool = False,
) -> list[Path]:
    """Write ``<assembly>.clusters.bed`` per assembly plus ``clusters.all.bed.gz``.

    Returns the per-assembly paths followed by the concatenated file.  Each BED
    is BED9 (``chrom start end name score strand thickStart thickEnd itemRgb``)
    behind a UCSC ``track`` line naming the run, sorted by contig then start.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    combined = outdir / "clusters.all.bed.gz"

    if rows is None or len(rows) == 0 or clusters is None or len(clusters) == 0:
        logger.warning("backprop: nothing to write (no bins or no clusters)")
        if force or not combined.exists():
            _write_gzip(combined, [_track_line(cfg, "all assemblies")])
        return [combined]

    frame = _bed_frame(rows, clusters, names, cfg)
    if frame.empty:
        logger.warning("backprop: no bin survived the rows/clusters join")
        if force or not combined.exists():
            _write_gzip(combined, [_track_line(cfg, "all assemblies")])
        return [combined]

    written: list[Path] = []
    for assembly, group in frame.groupby(frame["assembly"].astype(str), sort=True):
        path = outdir / f"{_safe_name(assembly)}.clusters.bed"
        if path.exists() and not force:
            logger.debug("backprop: reusing %s", path)
            written.append(path)
            continue
        tmp = path.with_suffix(".bed.tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(_track_line(cfg, assembly))
            handle.writelines(_bed_lines(group))
        tmp.replace(path)
        written.append(path)

    if force or not combined.exists():
        # Globally sorted by contig then start: contig names are unique across
        # assemblies (PanSN prefixes), so this is a usable single-file view.
        ordered = frame.sort_values(["contig", "start", "end"], kind="stable")
        _write_gzip(combined, [_track_line(cfg, "all assemblies"), *_bed_lines(ordered)])
    written.append(combined)
    logger.info("backprop: wrote %d BED file(s) under %s", len(written), outdir)
    return written


def _safe_name(assembly: str) -> str:
    """Assembly ids are filename-safe in practice; be sure anyway."""
    return "".join(c if c.isalnum() or c in "._-+" else "_" for c in str(assembly)) or "assembly"


# --------------------------------------------------------------------------
# does a cluster mean the same thing in the assemblies as in the reference?
# --------------------------------------------------------------------------


def _dominant_mode(values: np.ndarray) -> str:
    """Most frequent non-``unannotated`` label, ties broken alphabetically."""
    counts: dict[str, int] = {}
    for value in values:
        text = str(value)
        if not text or text == "unannotated":
            continue
        counts[text] = counts.get(text, 0) + 1
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _comparable_features(cfg: Config) -> tuple[str, ...]:
    """Features both the reference and the assemblies could have been labelled with.

    The reference carries a gene annotation and a telomere track that the HPRC
    per-assembly track set does not, so without this a euchromatic reference bin
    is ``gene`` while the very same locus in an assembly is ``line`` -- and the
    transfer score collapses to zero for reasons that have nothing to do with
    the clustering.  Restricting both sides to the intersection is the only
    comparison that measures what the report claims to measure.
    """
    ref = schemas.features_for_tracks(list(cfg.annotate.reference_tracks))
    asm = schemas.features_for_tracks(list(cfg.annotate.assembly_tracks))
    both = set(ref) & set(asm)
    return tuple(f for f in schemas.FEATURE_VOCAB if f in both)


def _restricted_dominant(
    joined: pd.DataFrame, comparable: Sequence[str], cfg: Config
) -> np.ndarray:
    """Per-bin dominant feature recomputed over ``comparable`` only.

    Falls back to the stored ``dominant_feature`` when the fraction columns are
    absent (an annotations table written by an older run) or when nothing was
    excluded, so the cheap path stays cheap.
    """
    stored = joined["dominant_feature"].astype(str).to_numpy(dtype=object)
    comparable = [f for f in comparable if schemas.feature_column(f) in joined.columns]
    if not comparable or len(comparable) == len(schemas.FEATURE_VOCAB):
        return stored
    cols = [schemas.feature_column(f) for f in comparable]
    fracs = joined[cols].to_numpy(dtype=np.float32, copy=False)
    if fracs.size == 0:
        return stored
    best = np.argmax(fracs, axis=1)
    best_frac = fracs[np.arange(fracs.shape[0]), best]
    threshold = max(float(cfg.annotate.min_frac_for_dominant), float(np.nextafter(0.0, 1.0)))
    names = np.asarray(comparable, dtype=object)
    return np.where(best_frac >= threshold, names[best], "unannotated")


def _write_transfer_features(comparable: Sequence[str], cfg: Config, outdir: Path) -> None:
    """Record which vocabulary the transfer score was computed over."""
    payload = {
        "reference_tracks": list(cfg.annotate.reference_tracks),
        "assembly_tracks": list(cfg.annotate.assembly_tracks),
        "comparable_features": list(comparable),
        "excluded_features": [
            f for f in schemas.FEATURE_VOCAB if f not in set(comparable)
        ],
        "why": (
            "asm_agreement is computed only over features both sides could have "
            "been labelled with; a feature only one track set can produce would "
            "otherwise score as disagreement."
        ),
    }
    path = Path(outdir) / "transfer_features.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def cluster_transfer_report(
    rows: pd.DataFrame,
    clusters: pd.DataFrame,
    annotations: pd.DataFrame,
    names: pd.DataFrame,
    cfg: Config,
    outdir: Path,
) -> pd.DataFrame:
    """Per cluster: does its reference annotation survive into the assemblies?

    Columns are ``cluster, name, n_ref_bins, n_asm_bins, ref_top_feature,
    asm_top_feature, asm_agreement, asm_annotated_frac``.  ``asm_agreement`` is
    the fraction of *annotated* assembly bins in the cluster whose dominant
    feature equals ``ref_top_feature``; it is ``NaN`` when the cluster has no
    annotated reference bins (nothing to transfer) or no annotated assembly bins
    (nothing to transfer to).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if clusters is None or len(clusters) == 0 or rows is None or len(rows) == 0:
        logger.warning("backprop: no clusters to score")
        report = schemas.empty_frame(TRANSFER_COLUMNS)
        _write_transfer(report, outdir)
        return report

    joined = _transfer_join(rows, clusters, annotations, names)
    if joined.empty:
        logger.warning("backprop: no bins survived the transfer join")
        report = schemas.empty_frame(TRANSFER_COLUMNS)
        _write_transfer(report, outdir)
        return report

    is_ref = joined["is_reference"].to_numpy(dtype=bool)
    annotated = joined["annotated"].to_numpy(dtype=bool)
    comparable = _comparable_features(cfg)
    dominant = _restricted_dominant(joined, comparable, cfg)
    _write_transfer_features(comparable, cfg, outdir)
    labels = joined["cluster"].to_numpy(dtype=np.int64)
    name_of = dict(zip(joined["cluster"].astype(int), joined["name"].astype(str)))

    records: list[dict[str, object]] = []
    for cid in np.unique(labels):
        mask = labels == cid
        ref_mask = mask & is_ref
        asm_mask = mask & ~is_ref
        ref_ann = ref_mask & annotated
        asm_ann = asm_mask & annotated

        ref_top = _dominant_mode(dominant[ref_ann])
        asm_top = _dominant_mode(dominant[asm_ann])
        n_asm_ann = int(asm_ann.sum())
        if ref_top and n_asm_ann:
            agreement = float(np.sum(dominant[asm_ann] == ref_top) / n_asm_ann)
        else:
            agreement = float("nan")
        n_asm = int(asm_mask.sum())
        records.append(
            {
                "cluster": int(cid),
                "name": name_of.get(int(cid), f"C{int(cid)}"),
                "n_ref_bins": int(ref_mask.sum()),
                "n_asm_bins": n_asm,
                "ref_top_feature": ref_top,
                "asm_top_feature": asm_top,
                "asm_agreement": agreement,
                "asm_annotated_frac": (n_asm_ann / n_asm) if n_asm else 0.0,
            }
        )

    report = schemas.enforce(
        pd.DataFrame(records).sort_values("cluster").reset_index(drop=True), TRANSFER_COLUMNS
    )
    _write_transfer(report, outdir)
    scored = report["asm_agreement"].notna()
    if scored.any():
        logger.info(
            "backprop: %d/%d cluster(s) testable, median assembly agreement %.3f",
            int(scored.sum()),
            len(report),
            float(report.loc[scored, "asm_agreement"].median()),
        )
    else:
        logger.warning("backprop: no cluster had both reference and assembly annotation")
    return report


def _transfer_join(
    rows: pd.DataFrame,
    clusters: pd.DataFrame,
    annotations: pd.DataFrame,
    names: pd.DataFrame,
) -> pd.DataFrame:
    frame = clusters[["bin_uid", "cluster"]].copy()
    frame["bin_uid"] = frame["bin_uid"].astype("string")
    frame["cluster"] = frame["cluster"].astype("int32")

    meta = rows[["bin_uid", "source", "haplotype"]].copy()
    meta["bin_uid"] = meta["bin_uid"].astype("string")
    frame = frame.merge(meta, on="bin_uid", how="inner")
    if frame.empty:
        return frame
    # The reference is whichever bins came from T2T; ``haplotype == "ref"`` is a
    # second spelling of the same thing in hand-written manifests.
    frame["is_reference"] = frame["source"].astype(str).eq("t2t") | frame["haplotype"].astype(
        str
    ).eq("ref")

    if annotations is not None and len(annotations):
        keep = ["bin_uid", "dominant_feature", "annotated"]
        keep += [c for c in schemas.FEATURE_COLUMNS if c in annotations.columns]
        ann = annotations[keep].copy()
        ann["bin_uid"] = ann["bin_uid"].astype("string")
        frame = frame.merge(ann, on="bin_uid", how="left")
    for col, default in (("dominant_feature", "unannotated"), ("annotated", False)):
        if col not in frame.columns:
            frame[col] = default
    frame["dominant_feature"] = frame["dominant_feature"].astype("string").fillna("unannotated")
    frame["annotated"] = frame["annotated"].fillna(False).astype(bool)

    if names is not None and len(names):
        lookup: Mapping[int, str] = dict(
            zip(names["cluster"].astype(int), names["name"].astype(str))
        )
    else:
        lookup = {}
    frame["name"] = [
        lookup.get(int(c)) or (NOISE_NAME if int(c) == NOISE_CLUSTER else f"C{int(c)}")
        for c in frame["cluster"]
    ]
    return frame


def _write_transfer(report: pd.DataFrame, outdir: Path) -> None:
    parquet = outdir / "cluster_transfer.parquet"
    tsv = outdir / "cluster_transfer.tsv"
    tmp = parquet.with_suffix(".parquet.tmp")
    report.to_parquet(tmp, index=False)
    tmp.replace(parquet)
    buffer = io.StringIO()
    report.to_csv(buffer, sep="\t", index=False, na_rep="NA", lineterminator="\n")
    tmp_tsv = tsv.with_suffix(".tsv.tmp")
    tmp_tsv.write_text(buffer.getvalue(), encoding="utf-8")
    tmp_tsv.replace(tsv)


def load_transfer_report(outdir: Path) -> pd.DataFrame:
    """Read back the frame written by :func:`cluster_transfer_report`."""
    path = Path(outdir) / "cluster_transfer.parquet"
    if not path.exists():
        raise FileNotFoundError(f"no transfer report at {path}")
    frame = pd.read_parquet(path)
    if frame.empty:
        return schemas.empty_frame(TRANSFER_COLUMNS)
    return schemas.enforce(frame, TRANSFER_COLUMNS)
