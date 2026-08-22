"""Config round-tripping, seed propagation and every validation rule.

``config.resolved.yaml`` is the provenance record for a run, so the round trip
has to be exact: loading what we dumped must give a config that compares equal
field-for-field, or a published run cannot be reproduced from its own output.
"""

from __future__ import annotations

import dataclasses

import pytest
import yaml

from kmer_dust.config import Config, SketchConfig, default_config


def test_defaults_are_valid():
    cfg = default_config()
    cfg.validate()  # must not raise
    assert cfg.sketch.k == 31
    assert cfg.sketch.bin_size == 10_000


def test_yaml_round_trip_is_exact(tmp_path):
    cfg = Config.from_dict(
        {
            "run_name": "rt",
            "outdir": str(tmp_path / "out"),
            "seed": 99,
            "sketch": {"k": 21, "scaled": 500, "bin_size": 5_000},
            "manifest": {"chroms": ["chr21", "chrX"], "samples": ["HG00408"]},
            "matrix": {"weighting": "log", "row_norm": "l1"},
        }
    )
    path = tmp_path / "config.yaml"
    cfg.dump(path)
    again = Config.load(path)
    assert dataclasses.asdict(again) == dataclasses.asdict(cfg)
    # ...and the dumped file is plain, readable YAML with no Python tags.
    text = path.read_text()
    assert "!!python" not in text
    assert yaml.safe_load(text)["sketch"]["k"] == 21


def test_dump_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "deep" / "nested" / "config.yaml"
    default_config().dump(path)
    assert path.exists()


def test_load_of_an_empty_yaml_gives_defaults(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    assert dataclasses.asdict(Config.load(path)) == dataclasses.asdict(default_config())


def test_to_dict_is_json_like():
    data = default_config().to_dict()
    assert isinstance(data["sketch"], dict)
    assert data["manifest"]["chroms"] == ["chr21"]


# --------------------------------------------------------------------------
# unknown keys
# --------------------------------------------------------------------------


def test_unknown_top_level_key_raises():
    with pytest.raises(ValueError, match="unknown top-level"):
        Config.from_dict({"nonsense": 1})


def test_unknown_section_key_raises():
    with pytest.raises(ValueError, match="unknown key"):
        Config.from_dict({"sketch": {"kk": 31}})


def test_null_section_is_treated_as_empty():
    cfg = Config.from_dict({"sketch": None})
    assert cfg.sketch == SketchConfig()


# --------------------------------------------------------------------------
# seed propagation
# --------------------------------------------------------------------------


def test_seed_propagates_to_every_seeded_section():
    cfg = Config.from_dict({"seed": 4242})
    for section in ("manifest", "select", "decompose", "embed", "cluster", "report"):
        assert getattr(cfg, section).seed == 4242, section


def test_explicit_section_seed_survives_propagation():
    cfg = Config.from_dict({"seed": 4242, "embed": {"seed": 11}})
    assert cfg.embed.seed == 11
    assert cfg.cluster.seed == 4242


def test_default_top_level_seed_leaves_sections_alone():
    cfg = Config.from_dict({"cluster": {"seed": 3}})
    assert cfg.seed == 7
    assert cfg.cluster.seed == 3
    assert cfg.embed.seed == 7


# --------------------------------------------------------------------------
# derived paths
# --------------------------------------------------------------------------


def test_stage_dir_creates_and_is_under_outdir(tmp_path):
    cfg = Config.from_dict({"outdir": str(tmp_path / "run")})
    stage = cfg.stage_dir("sketch")
    assert stage.is_dir()
    assert stage == cfg.out / "sketch"
    assert cfg.path("sketch", "a.parquet") == cfg.out / "sketch" / "a.parquet"


def test_max_hash_matches_scaled():
    cfg = Config.from_dict({"sketch": {"scaled": 200}})
    assert cfg.max_hash == ((1 << 64) - 1) // 200
    assert Config.from_dict({"sketch": {"scaled": 1}}).max_hash == (1 << 64) - 1


# --------------------------------------------------------------------------
# validation -- one case per rule in Config.validate
# --------------------------------------------------------------------------


BAD_CONFIGS = {
    "k_too_large": ({"sketch": {"k": 33}}, "sketch.k"),
    "k_zero": ({"sketch": {"k": 0}}, "sketch.k"),
    "k_even": ({"sketch": {"k": 30}}, "odd"),
    "bin_smaller_than_k": ({"sketch": {"k": 31, "bin_size": 30}}, "bin_size"),
    "scaled_zero": ({"sketch": {"scaled": 0}}, "scaled"),
    "acgt_frac_high": ({"sketch": {"min_bin_acgt_frac": 1.5}}, "min_bin_acgt_frac"),
    "acgt_frac_negative": ({"sketch": {"min_bin_acgt_frac": -0.1}}, "min_bin_acgt_frac"),
    "prevalence_inverted": (
        {"select": {"min_sample_prevalence": 0.9, "max_sample_prevalence": 0.1}},
        "prevalence",
    ),
    "prevalence_above_one": ({"select": {"max_sample_prevalence": 1.5}}, "prevalence"),
    "buckets_not_power_of_two": ({"select": {"n_buckets": 12}}, "power of two"),
    "buckets_zero": ({"select": {"n_buckets": 0}}, "power of two"),
    "bad_weighting": ({"matrix": {"weighting": "tfidf"}}, "weighting"),
    "bad_row_norm": ({"matrix": {"row_norm": "max"}}, "row_norm"),
    "too_few_components": ({"decompose": {"n_components": 1}}, "n_components"),
    "drop_first_too_large": (
        {"decompose": {"n_components": 4, "drop_first": 4}},
        "drop_first",
    ),
    "embed_dims": ({"embed": {"n_components": 4}}, "embed.n_components"),
    "bad_cluster_method": ({"cluster": {"method": "kmeans"}}, "cluster.method"),
    "bad_cluster_space": ({"cluster": {"space": "matrix"}}, "cluster.space"),
    "bad_manifest_source": ({"manifest": {"source": "hprc_release1"}}, "manifest.source"),
    "manifest_path_required_file": ({"manifest": {"source": "file"}}, "manifest.path"),
    "manifest_path_required_dir": ({"manifest": {"source": "local_dir"}}, "manifest.path"),
}


@pytest.mark.parametrize("payload,needle", list(BAD_CONFIGS.values()), ids=list(BAD_CONFIGS))
def test_validation_error_fires(payload, needle):
    with pytest.raises(ValueError, match="invalid configuration"):
        Config.from_dict(payload)
    try:
        Config.from_dict(payload)
    except ValueError as exc:
        assert needle in str(exc), f"expected {needle!r} in:\n{exc}"


def test_validation_reports_every_problem_at_once():
    with pytest.raises(ValueError) as exc:
        Config.from_dict({"sketch": {"k": 30}, "matrix": {"weighting": "nope"}})
    message = str(exc.value)
    assert "odd" in message and "weighting" in message


def test_boundary_values_are_accepted():
    Config.from_dict({"sketch": {"k": 31, "bin_size": 31, "scaled": 1}})
    Config.from_dict({"sketch": {"min_bin_acgt_frac": 0.0}})
    Config.from_dict({"sketch": {"min_bin_acgt_frac": 1.0}})
    Config.from_dict({"select": {"min_sample_prevalence": 0.5, "max_sample_prevalence": 0.5}})
    Config.from_dict({"select": {"n_buckets": 1}})
    Config.from_dict({"embed": {"n_components": 3}})
    Config.from_dict({"decompose": {"n_components": 2, "drop_first": 1}})


# --------------------------------------------------------------------------
# CLI-level contracts that only surface when the package is invoked as a
# program -- which is how the container runs it.
# --------------------------------------------------------------------------


def test_version_flag_needs_no_subcommand():
    """`docker run <image> --version` must work.

    Typer/click refuse a bare root option with "Missing command" unless the app
    is built with ``invoke_without_command=True``; the container's CI step runs
    exactly this, so it is worth pinning.
    """
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "kmer_dust", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("kmer-dust ")


def test_bare_invocation_shows_help():
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "kmer_dust"], capture_output=True, text=True, check=False
    )
    assert "Usage" in (proc.stdout + proc.stderr)
    assert "sketch" in (proc.stdout + proc.stderr)


def test_a_partial_rerun_keeps_the_timings_of_stages_it_skipped(tmp_path):
    """Re-running one cheap stage must not erase the record of the expensive ones.

    A skipped stage short-circuits on its existing output and reports ~0 s, so
    overwriting the summary with only this invocation's stages destroys exactly
    the evidence you wanted -- what the run actually cost.
    """
    import json

    from kmer_dust.config import Config
    from kmer_dust.pipeline import StageResult, _write_run_summary

    cfg = Config(outdir=str(tmp_path))
    _write_run_summary(
        cfg,
        [
            StageResult("sketch", 702.0, {"bins": 100}),
            StageResult("cluster", 4819.0, {"clusters": 3021}),
        ],
    )
    # ... now a partial re-run touches only `report`
    _write_run_summary(cfg, [StageResult("report", 2.1, {"path": "x.html"})])

    stages = {s["name"]: s for s in json.loads((tmp_path / "run_summary.json").read_text())["stages"]}
    assert stages["sketch"]["seconds"] == 702.0, "the expensive stage's timing survived"
    assert stages["cluster"]["clusters"] == 3021
    assert stages["report"]["seconds"] == 2.1
    # and re-running a stage updates it rather than duplicating it
    _write_run_summary(cfg, [StageResult("cluster", 5.0, {"clusters": 42})])
    stages = {s["name"]: s for s in json.loads((tmp_path / "run_summary.json").read_text())["stages"]}
    assert stages["cluster"]["seconds"] == 5.0
    assert stages["cluster"]["clusters"] == 42
    assert stages["sketch"]["seconds"] == 702.0


def test_run_summary_keeps_pipeline_order(tmp_path):
    import json

    from kmer_dust.config import Config
    from kmer_dust.pipeline import STAGES, StageResult, _write_run_summary

    cfg = Config(outdir=str(tmp_path))
    _write_run_summary(cfg, [StageResult("report", 1.0, {})])
    _write_run_summary(cfg, [StageResult("sketch", 2.0, {})])
    names = [s["name"] for s in json.loads((tmp_path / "run_summary.json").read_text())["stages"]]
    assert names == [n for n in STAGES if n in set(names)]
