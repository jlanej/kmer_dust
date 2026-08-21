"""Preflight checks: the difference between a 40-minute mystery and one line.

These exist because a real CI run fetched zero assemblies, took 0.0 s to do it,
and reported only the symptom. The reason -- htslib built without libcurl
cannot open https:// at all -- was recoverable in one call, and the discarded
C-level error message was recoverable in one more.
"""

from __future__ import annotations

from unittest import mock

import pytest

from kmer_dust import preflight


def _info(**features):
    return preflight.HtslibInfo("1.23.1", dict(features), (), "raw text")


def test_libcurl_yes_means_https_is_available():
    assert _info(libcurl="yes").https is True
    assert _info(libcurl="yes").libcurl is True


def test_libcurl_no_means_https_is_not_available():
    assert _info(libcurl="no").https is False


def test_an_explicit_scheme_list_wins_over_the_build_flag():
    """htslib can be built with libcurl and still not register https."""
    with_schemes = preflight.HtslibInfo("1.23.1", {"libcurl": "yes"}, ("file", "ftp"), "raw")
    assert with_schemes.https is False
    both = preflight.HtslibInfo("1.23.1", {"libcurl": "no"}, ("file", "https"), "raw")
    assert both.https is True


def test_check_remote_access_explains_a_missing_libcurl():
    with mock.patch.object(preflight, "htslib_info", return_value=_info(libcurl="no")):
        ok, why = preflight.check_remote_access()
    assert ok is False
    # The message has to name the cause AND the fix, because whoever reads it is
    # looking at a log, not at this source file.
    assert "libcurl" in why
    assert "only-binary" in why


def test_require_remote_access_raises_with_the_explanation():
    with mock.patch.object(preflight, "htslib_info", return_value=_info(libcurl="no")):
        with pytest.raises(preflight.RemoteAccessError, match="libcurl"):
            preflight.require_remote_access()


def test_require_remote_access_passes_on_a_healthy_build():
    with mock.patch.object(preflight, "htslib_info", return_value=_info(libcurl="yes")):
        preflight.require_remote_access()  # must not raise


def test_an_unparseable_htslib_never_blocks_the_run():
    """A preflight that itself explodes is worse than the problem it screens for."""
    empty = preflight.HtslibInfo("", {}, (), "")
    with mock.patch.object(preflight, "htslib_info", return_value=empty):
        ok, why = preflight.check_remote_access()
    assert ok is True
    assert "could not determine" in why


def test_htslib_info_parses_the_real_installed_build():
    info = preflight.htslib_info()
    # pysam is a hard dependency, so this must produce something real.
    assert info.version, "could not read the htslib version from pysam"
    assert "libcurl" in info.features


def test_capture_c_stderr_recovers_what_pysam_throws_away():
    """The whole point: pysam's OSError names only the URL.

    htslib writes the actual reason to fd 2 from C, so without this capture a
    connection refusal and an unsupported protocol are indistinguishable.
    """
    pysam = pytest.importorskip("pysam")
    url = "https://127.0.0.1:9/definitely-not-there.fa"  # port 9 = discard
    with preflight.capture_c_stderr() as captured:
        with pytest.raises(Exception) as excinfo:
            pysam.FastaFile(url)

    assert url in str(excinfo.value)
    assert captured, "htslib wrote nothing to fd 2, or the capture missed it"
    joined = " ".join(captured)
    assert "E::" in joined

    combined = preflight.describe_open_failure(url, excinfo.value, captured)
    assert url in combined
    assert any(part in combined for part in ("Connection refused", "E::"))


def test_capture_c_stderr_restores_the_descriptor():
    import sys

    before = sys.stderr
    with preflight.capture_c_stderr():
        pass
    assert sys.stderr is before
    # and fd 2 still works
    print("", file=sys.stderr)
