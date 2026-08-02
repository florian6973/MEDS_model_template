"""Split-sharded input is a precondition, checked before any work happens.

meds-torch-data recovers split membership from the shard path and from nothing else, so input sharded
another way tensorizes into an artifact with no splits — surfacing as MTD's "No schema files found", long
after the cause and in MTD's vocabulary rather than this template's.

Resharding on the user's behalf is what could not be made to work: ``reshard_to_split`` needs
``metadata/subject_splits.parquet`` in its own input, and a MEDS-transforms pipeline drops that file, so
``pipeline`` plus resharding could only ever fail — and always *after* the pipeline had run. Requiring the
layout is the one rule that holds either way.

Needs no model and no MTD: the check is a directory listing.
"""
from datetime import datetime

import polars as pl
import pytest
from meds_model_base.commands.preprocess_data import _require_split_sharded


def _write_shard(fp):
    fp.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "subject_id": [1, 2],
            "time": [datetime(2020, 1, 1)] * 2,
            "code": ["A", "B"],
            "numeric_value": [1.0, 2.0],
        }
    ).write_parquet(fp)


def test_split_sharded_input_is_accepted(tmp_path):
    for split in ("train", "tuning", "held_out"):
        _write_shard(tmp_path / "data" / split / "0.parquet")
    _require_split_sharded(tmp_path, "external_meds_dir")


def test_a_missing_split_is_still_fine(tmp_path):
    """Not every dataset carries all three; only the *layout* is required."""
    _write_shard(tmp_path / "data" / "train" / "0.parquet")
    _require_split_sharded(tmp_path, "external_meds_dir")


def test_flat_sharding_is_refused_with_the_remedy(tmp_path):
    _write_shard(tmp_path / "data" / "0.parquet")

    with pytest.raises(ValueError) as excinfo:
        _require_split_sharded(tmp_path, "external_meds_dir")

    message = str(excinfo.value)
    assert "not sharded by split" in message
    # Refusing without saying what to run would just move the dead end earlier.
    assert "reshard_to_split" in message
    assert "MEDS_transform-pipeline" in message


def test_partially_split_sharded_is_refused(tmp_path):
    """A stray top-level shard alongside split directories loses exactly those subjects, silently."""
    _write_shard(tmp_path / "data" / "train" / "0.parquet")
    _write_shard(tmp_path / "data" / "leftover.parquet")

    with pytest.raises(ValueError, match=r"leftover\.parquet"):
        _require_split_sharded(tmp_path, "external_meds_dir")


def test_dot_directories_are_ignored(tmp_path):
    """MEDS ETL output carries data/.logs/; it is not a shard."""
    _write_shard(tmp_path / "data" / "train" / "0.parquet")
    _write_shard(tmp_path / "data" / ".logs" / "junk.parquet")
    _require_split_sharded(tmp_path, "external_meds_dir")


def test_an_empty_dataset_is_refused(tmp_path):
    (tmp_path / "data").mkdir()
    with pytest.raises(FileNotFoundError, match="No MEDS data shards"):
        _require_split_sharded(tmp_path, "external_meds_dir")
