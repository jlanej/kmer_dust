"""Reporting layer: everything that turns finished stage output into a picture.

Kept behind a subpackage because it is the only part of kmer-dust that depends
on presentation choices (plotly, colour vocabulary, HTML assets) rather than on
the data contracts in :mod:`kmer_dust.schemas`.  Nothing else in the package
imports from here, so a headless run never pays for it.
"""

from __future__ import annotations

from .report import REPORT_COLUMNS, build_report, collect_report_frame

__all__ = ["build_report", "collect_report_frame", "REPORT_COLUMNS"]
