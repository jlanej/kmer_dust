"""Environment checks that turn a confusing failure into a one-line answer.

Everything in this pipeline that reads a remote assembly goes through htslib,
via pysam.  htslib can only open an ``https://`` URL if it was *built* against
libcurl -- and a pysam installed from the sdist rather than from a manylinux
wheel is routinely built without it.  When that happens every remote open fails
instantly with ``error when opening file <url>``, which names the URL and
nothing else: the discriminating text (``Connection refused`` vs ``Protocol not
supported``) is written by htslib straight to file descriptor 2 from C, never
passes through :mod:`logging`, and does not interleave predictably with Python
output under CI log buffering.

The symptom is a run that downloads nothing, blames no one, and takes 0.0
seconds to do it.  This module exists so that never costs anyone an afternoon
again: :func:`check_remote_access` answers "can this interpreter read an https
FASTA at all" before a single byte is attempted, and :func:`capture_c_stderr`
recovers the C-level message when something does go wrong anyway.
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
import tempfile
from collections.abc import Iterator
from typing import NamedTuple

from .log import get_logger

logger = get_logger(__name__)

__all__ = [
    "HtslibInfo",
    "htslib_info",
    "check_remote_access",
    "require_remote_access",
    "capture_c_stderr",
    "RemoteAccessError",
]

_FEATURE_RE = re.compile(r"(\w+)=(\S+)")


class RemoteAccessError(RuntimeError):
    """htslib in this interpreter cannot open remote URLs."""


class HtslibInfo(NamedTuple):
    version: str
    features: dict[str, str]
    schemes: tuple[str, ...]
    raw: str

    @property
    def libcurl(self) -> bool:
        return self.features.get("libcurl", "no").lower() == "yes"

    @property
    def https(self) -> bool:
        """True when https is usable.

        Prefer the explicit URL-scheme-handler block when htslib prints one;
        fall back to the libcurl build flag, which is what actually decides it.
        """
        if self.schemes:
            return "https" in self.schemes
        return self.libcurl

    def summary(self) -> str:
        flags = " ".join(f"{k}={v}" for k, v in sorted(self.features.items()))
        return f"htslib {self.version or '?'} [{flags or 'unknown build'}]"


def htslib_info() -> HtslibInfo:
    """Parse ``samtools --version --verbose`` for htslib's build configuration.

    Returns an empty-but-valid record rather than raising: a preflight check
    that itself explodes is worse than the problem it screens for.
    """
    try:
        import pysam

        raw = str(pysam.samtools.version("--verbose"))
    except Exception as exc:  # noqa: BLE001 - diagnostics must never be fatal
        logger.debug("cannot query htslib build details: %s", exc)
        return HtslibInfo("", {}, (), "")

    version = ""
    match = re.search(r"Using htslib (\S+)", raw)
    if match:
        version = match.group(1)

    features: dict[str, str] = {}
    block = raw.find("HTSlib compilation details")
    if block >= 0:
        line = re.search(r"Features:\s*(.+)", raw[block:])
        if line:
            features = dict(_FEATURE_RE.findall(line.group(1)))

    schemes: list[str] = []
    handlers = raw.find("URL scheme handlers")
    if handlers >= 0:
        for token in re.findall(r"[A-Za-z][\w+.-]*", raw[handlers:]):
            schemes.append(token.lower())

    return HtslibInfo(version, features, tuple(schemes), raw)


def check_remote_access() -> tuple[bool, str]:
    """``(ok, explanation)`` for "can htslib open an https URL in this process".

    This is a *build* check, not a connectivity check -- it is deliberately
    offline and instant, so it can run before any network work and cleanly
    separate "this install cannot do https" from "this network cannot reach S3".
    """
    info = htslib_info()
    if not info.raw:
        return True, "could not determine htslib build details; proceeding"
    if info.https:
        return True, info.summary()
    return False, (
        f"{info.summary()}: this htslib was built WITHOUT libcurl, so it cannot open "
        "https:// URLs at all. Every remote assembly read will fail instantly. This "
        "usually means pysam was built from source instead of installed from a "
        "manylinux wheel -- reinstall with "
        "`pip install --force-reinstall --only-binary=:all: pysam`."
    )


def require_remote_access() -> None:
    """Raise :class:`RemoteAccessError` when remote reads cannot possibly work."""
    ok, explanation = check_remote_access()
    if not ok:
        raise RemoteAccessError(explanation)
    logger.debug("remote access preflight: %s", explanation)


@contextlib.contextmanager
def capture_c_stderr() -> Iterator[list[str]]:
    """Collect what htslib writes to fd 2 while the block runs.

    pysam surfaces ``OSError('error when opening file <url>')`` and drops the
    reason on the floor, because htslib writes its ``[E::...]`` diagnostic
    directly to the file descriptor.  Redirecting fd 2 to a temp file for the
    duration of the call is the only way to get it back and attach it to the
    exception the caller reports.

    The yielded list is populated on exit.  If anything about the redirection
    fails, the block still runs and the list is simply empty -- diagnostics must
    never break the thing they are diagnosing.
    """
    captured: list[str] = []
    try:
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass
    try:
        saved = os.dup(2)
    except OSError:
        yield captured
        return
    tmp = tempfile.TemporaryFile(mode="w+b")
    try:
        os.dup2(tmp.fileno(), 2)
        yield captured
    finally:
        try:
            os.dup2(saved, 2)
        finally:
            os.close(saved)
        try:
            tmp.seek(0)
            text = tmp.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            text = ""
        finally:
            tmp.close()
        for line in text.splitlines():
            line = line.strip()
            if line:
                captured.append(line)
                # Keep it visible: it was on its way to the terminal anyway.
                logger.debug("htslib: %s", line)


def describe_open_failure(url: str, exc: BaseException, captured: list[str]) -> str:
    """Best available one-line explanation for a failed remote open."""
    detail = "; ".join(captured[-3:]) if captured else ""
    ok, explanation = check_remote_access()
    parts = [f"{type(exc).__name__}: {exc}"]
    if detail:
        parts.append(detail)
    if not ok:
        parts.append(explanation)
    elif not detail:
        parts.append(f"(htslib reported nothing further; {explanation})")
    return " | ".join(parts)
