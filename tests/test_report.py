"""The HTML report: it must be buildable from a finished run directory alone.

The report is the only output most readers will ever open, and it is explicitly
meant to work offline (``report.embed_plotlyjs``), so the tests check that it is
self-contained rather than checking what it looks like.
"""

from __future__ import annotations

import json
from html.parser import HTMLParser

import pandas as pd
import pytest

from kmer_dust.viz.report import build_report, collect_report_frame

pytestmark = pytest.mark.slow


class _AssetScan(HTMLParser):
    """Collect every tag that would make the browser fetch a remote asset.

    A regex is useless here: once plotly.js is inlined the file contains
    megabytes of minified JavaScript with https:// literals (including
    ``cdn.plot.ly`` itself, as a topojson default).  HTMLParser treats script
    bodies as raw text, so only real markup is inspected.
    """

    ASSET_TAGS = {"script", "link", "img", "iframe", "source", "audio", "video"}

    def __init__(self) -> None:
        super().__init__()
        self.external: list[tuple[str, str]] = []

    def handle_starttag(self, tag, attrs):
        if tag not in self.ASSET_TAGS:
            return
        for key, value in attrs:
            if key in {"src", "href"} and str(value or "").startswith(("http://", "https://")):
                self.external.append((tag, value))


def test_collect_report_frame_has_one_row_per_bin(smoke_run):
    cfg, _, _ = smoke_run
    rows = pd.read_parquet(cfg.path("matrix", "rows.parquet"))
    frame = collect_report_frame(cfg, cfg.out)
    assert len(frame) == len(rows)
    assert "bin_uid" in frame.columns
    assert "cluster" in frame.columns
    assert set(frame["bin_uid"]) == set(rows["bin_uid"])
    # the embedding coordinates have to be in there for the plot to exist
    assert sum(c in frame.columns for c in ("x", "y", "umap_1", "umap_2", "umap1", "umap2")) >= 2


def test_build_report_writes_a_self_contained_html(smoke_run):
    cfg, _, _ = smoke_run
    path = build_report(cfg, cfg.out, force=True)
    assert path.exists()
    assert path.name == "kmer_dust_report.html"
    assert path.parent.name == "report"
    text = path.read_text(errors="replace")
    assert "<html" in text.lower()
    # report.embed_plotlyjs is True in the smoke config, so the ~4.8 MB plotly
    # bundle must really be in the file rather than pulled from the CDN.
    assert len(text) > 1_000_000
    scan = _AssetScan()
    scan.feed(text)
    assert scan.external == [], f"the report must work with no network: {scan.external}"


def _walk(obj):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _walk(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk(value)
    else:
        yield obj


def test_summary_json(smoke_run):
    """The headline numbers must actually be the run's numbers."""
    cfg, manifest, _ = smoke_run
    build_report(cfg, cfg.out, force=True)
    summary = json.loads(cfg.path("report", "summary.json").read_text())
    assert isinstance(summary, dict) and summary
    rows = pd.read_parquet(cfg.path("matrix", "rows.parquet"))
    values = list(_walk(summary))
    assert len(rows) in values, "the bin count is missing from summary.json"
    assert len(manifest) in values, "the assembly count is missing from summary.json"
    assert json.dumps(summary)  # plain JSON, no numpy scalars survived


def test_rebuild_is_stable(smoke_run):
    cfg, _, _ = smoke_run
    first = build_report(cfg, cfg.out, force=True).read_bytes()
    second = build_report(cfg, cfg.out, force=True).read_bytes()
    assert len(first) == len(second)


def test_cdn_url_uses_the_plotlyjs_version_not_the_plotly_py_version():
    """The CDN indexes plotly.js, which is a different project from plotly.py.

    plotly.py 6.9.0 ships plotly.js 3.7.0. Building the CDN URL from
    ``plotly.__version__`` yields ``plotly-6.9.0.min.js``, which does not exist:
    the CDN answers 403, the script never loads, and the page renders as a blank
    white rectangle with no error that names the cause. This bit exactly once.
    """
    from kmer_dust.config import Config
    from kmer_dust.viz.report import _plotly_block, _plotlyjs_version

    js_version = _plotlyjs_version()
    assert js_version and js_version[0].isdigit()

    import plotly

    cfg = Config()
    cfg.report.embed_plotlyjs = False
    tag, reported = _plotly_block(cfg)
    assert f"plotly-{js_version}.min.js" in tag
    assert reported == js_version
    # The bug was using the Python package version here.
    if plotly.__version__ != js_version:
        assert f"plotly-{plotly.__version__}.min.js" not in tag
