"""Presence featurization: the reader's grammar, the artifact contract, and the model's own predicates.

Three layers, all dependency-free (no meds-torch-data, no model), so they run in every generated repo
whatever its ``data_backend``:

- **Grammar** tests use inline YAML/frames: the grammar belongs to the template, so literals are correct
  here — everywhere else the suite reads the repository's own ``predicates.yaml``, never a test copy.
- **The file** is validated strictly: runtime skips unsupported forms in *foreign* files, but your own
  declaration containing forms featurization cannot use is a bug in the file, surfaced here directly
  rather than through a warning nobody reads.
- **The artifact** tests assert the published contract (columns, ``features.json``, manifest,
  split-layout-as-authority) against whatever predicates the file declares — they hardcode none of its
  names, so they keep passing when you replace the starter content with your real concepts.

The equivalence guard at the bottom is the anti-drift test between the two representations' cohort
readers; it needs an MTD workspace, so it self-skips in a ``custom_featurization`` repo.
"""

import json
from pathlib import Path

import polars as pl
import pytest
import yaml
from meds_model_base.featurize import (
    FEATURES_FILENAME,
    PREDICATE_COLUMN_PREFIX,
    PredicatesError,
    featurize_frame,
    load_features,
    parse_predicates,
    read_predicates_file,
)
from meds_model_base.manifest import read_manifest
from meds_model_base.tasks import cohort_subjects, featurized_cohort, materialize_labels

PREDICATES_FILE = Path(__file__).resolve().parent.parent / "predicates.yaml"


# --- Grammar (inline literals: the grammar is the template's, not the model's) -------------------------


def _frame():
    return pl.DataFrame(
        {
            "subject_id": [1, 1, 2, 2],
            "code": ["HR", "ADMISSION//CARDIAC", "TEMP", "DISCHARGE"],
            "numeric_value": [99.0, None, 37.0, None],
        }
    )


def test_every_supported_form_matches_what_it_says(tmp_path):
    parsed = parse_predicates(
        {
            "hr": {"code": "HR"},
            "adm": {"code": {"regex": "^ADMISSION//.*"}},
            "vitals": {"code": {"any": ["HR", "TEMP"]}},
            "hr_or_adm": {"expr": "or(hr, adm)"},
        }
    )
    out = featurize_frame(_frame(), parsed)
    assert out["predicate//hr"].to_list() == [1, 0, 0, 0]
    assert out["predicate//adm"].to_list() == [0, 1, 0, 0]
    assert out["predicate//vitals"].to_list() == [1, 0, 1, 0]
    assert out["predicate//hr_or_adm"].to_list() == [1, 1, 0, 0]
    assert all(out[c].dtype == pl.Int8 for c in parsed.columns)


def test_original_columns_are_untouched():
    df = _frame()
    out = featurize_frame(df, parse_predicates({"hr": {"code": "HR"}}))
    assert out.select(df.columns).equals(df), "featurization is pure augmentation"


def test_unsupported_forms_skip_with_a_cascade():
    """Value bounds are model-side semantics; the or() referencing them must fall with them."""
    parsed = parse_predicates(
        {
            "hr": {"code": "HR"},
            "high_hr": {"code": "HR", "value_min": 110},
            "high_or_hr": {"expr": "or(high_hr, hr)"},
        }
    )
    assert parsed.names == ["hr"]
    assert set(parsed.skipped) == {"high_hr", "high_or_hr"}


def test_strict_mode_errors_listing_the_offenders():
    with pytest.raises(PredicatesError, match="high_hr"):
        parse_predicates({"hr": {"code": "HR"}, "high_hr": {"code": "HR", "value_min": 110}}, strict=True)


def test_an_empty_feature_space_is_always_an_error():
    """Skipping everything must not degrade into publishing a featureless artifact."""
    with pytest.raises(PredicatesError, match="No featurizable predicate remains"):
        parse_predicates({"high_hr": {"code": "HR", "value_min": 110}})


def test_refeaturizing_featurized_data_is_refused():
    parsed = parse_predicates({"hr": {"code": "HR"}})
    with pytest.raises(PredicatesError, match="already carries predicate column"):
        featurize_frame(featurize_frame(_frame(), parsed), parsed)


def test_a_missing_predicates_file_names_itself(tmp_path):
    with pytest.raises(PredicatesError, match="does not exist"):
        read_predicates_file(tmp_path / "nope.yaml")


# --- The model's own predicates file -------------------------------------------------------------------


def test_the_repos_predicates_file_parses_with_zero_skips():
    """Strictness lives here, not at runtime: skip-with-warning is for *foreign* files (a raw dataset
    predicates.yaml in a MEDS-DEV run); your own declaration should never contain forms featurization
    cannot use — that is a model silently narrower than its author believes."""
    parsed = read_predicates_file(PREDICATES_FILE)
    assert parsed.skipped == {}, (
        f"predicates.yaml contains unsupported predicate(s): {parsed.skipped}. Rewrite them in the "
        "supported presence subset (exact / regex / any-of / or)."
    )
    assert parsed.names, "predicates.yaml must declare at least one predicate"


# --- The published artifact (generic over whatever the file declares) ----------------------------------


@pytest.fixture(scope="session")
def patients(featurized_data_dir):
    return featurized_data_dir / "patients"


def test_manifest_declares_the_representation_and_the_provenance(patients):
    manifest = read_manifest(patients, require_type="data")
    assert manifest["representation"] == "predicates"
    feat = manifest["featurization"]
    assert feat["predicates_digest"], "the digest is what makes a drifted predicates file detectable"
    assert feat["n_features"] == len(load_features(patients))
    assert feat["skipped"] == [], "the shipped file parses clean, so nothing may be skipped"


def test_features_json_is_the_feature_space(patients):
    """Order and spelling come from features.json — never from parquet column order or the YAML."""
    features = load_features(patients)
    declared = set(yaml.safe_load(PREDICATES_FILE.read_text())["predicates"])
    for f in features:
        assert f["name"] in declared
        assert f["column"] == f"{PREDICATE_COLUMN_PREFIX}{f['name']}"

    columns = [f["column"] for f in features]
    for shard in sorted((patients / "data").rglob("*.parquet")):
        df = pl.read_parquet(shard)
        for c in columns:
            assert df[c].dtype == pl.Int8, f"{c} in {shard.name}"
            assert set(df[c].unique().to_list()) <= {0, 1}
    raw = json.loads((patients / FEATURES_FILENAME).read_text())
    assert raw["version"] == 1


def test_match_counts_in_the_manifest_are_the_column_sums(patients):
    counts = read_manifest(patients)["featurization"]["match_counts"]
    shards = sorted((patients / "data").rglob("*.parquet"))
    for f in load_features(patients):
        total = sum(int(pl.read_parquet(s)[f["column"]].sum()) for s in shards)
        assert counts[f["name"]] == total


def test_split_membership_travels_as_shard_layout_only(patients):
    """The featurized artifact keeps the MTD branch's invariant: no subject_splits.parquet copy."""
    assert not (patients / "metadata" / "subject_splits.parquet").exists()
    cohort = featurized_cohort(patients)
    assert set(cohort) and all(cohort.values())
    assert cohort == cohort_subjects(patients), "the manifest dispatch must land on the shard reader"
    assert read_manifest(patients)["splits"] == {s: len(m) for s, m in sorted(cohort.items())}


def test_labels_materialize_against_the_featurized_cohort(patients, labels_dir, tmp_path):
    dest, summary = materialize_labels(labels_dir, patients, tmp_path / "labels")
    assert summary, "at least one split must receive labels"
    for split in summary:
        assert (dest / f"{split}.parquet").is_file()


# --- Equivalence guard: the two cohort readers must never drift ----------------------------------------


def test_both_representations_partition_labels_identically(
    data_dir, featurized_data_dir, labels_dir, tmp_path
):
    """The anti-drift test. materialize_labels resolves splits from tokenization/schemas/ on an MTD
    artifact and from the data/ shard layout on a featurized one; if those ever disagree, the failure
    would otherwise surface as a CoverageError two training runs later."""
    pytest.importorskip("meds_torchdata")
    if read_manifest(data_dir / "patients").get("representation", "mtd") != "mtd":
        pytest.skip("no MTD workspace in this repo (data_backend=custom_featurization)")

    _, mtd_summary = materialize_labels(labels_dir, data_dir / "patients", tmp_path / "mtd")
    _, feat_summary = materialize_labels(
        labels_dir, featurized_data_dir / "patients", tmp_path / "featurized"
    )
    mtd = {s: pl.read_parquet(tmp_path / "mtd" / f"{s}.parquet") for s in mtd_summary}
    feat = {s: pl.read_parquet(tmp_path / "featurized" / f"{s}.parquet") for s in feat_summary}
    assert set(mtd) == set(feat)
    for split in mtd:
        assert sorted(mtd[split]["subject_id"].to_list()) == sorted(feat[split]["subject_id"].to_list())
