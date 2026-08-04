"""Labels are partitioned by the split each subject actually landed in the patients artifact under.

Needs no model, so unlike most of this suite it runs from the moment a repository is generated: the
``data_dir`` fixture is a real ``preprocess_data`` run over the meds-testing-helpers dataset, in
whichever representation this repo's ``data_backend`` selected.

The bug being pinned is quiet by construction. A ``preprocess_data`` pipeline that filters subjects
tokenizes them away while their labels survive; ``pretrain`` and ``supervised_train`` read training data
through the schema directories, see only the survivors, and succeed. ``predict`` is the first command to
compare the two and fails with ``CoverageError`` — accurate, but pointing at prediction rather than at the
preprocessing that caused it, two training runs later.
"""

from datetime import datetime

import polars as pl
import pytest
from meds_model_base.manifest import read_manifest
from meds_model_base.tasks import (
    TaskMaterializationError,
    cohort_subjects,
    materialize_labels,
    read_labels,
    split_labels,
)

#: Far outside any fixture's id space, so it can only ever be "absent from the cohort".
ABSENT_SUBJECT = 9_999_999


@pytest.fixture
def patients_dir(data_dir):
    return data_dir / "patients"


@pytest.fixture
def cohort(patients_dir):
    return cohort_subjects(patients_dir)


def _labels(subject_ids):
    return pl.DataFrame(
        {
            "subject_id": list(subject_ids),
            "prediction_time": [datetime(2020, 1, 1)] * len(subject_ids),
            "boolean_value": [True] * len(subject_ids),
        }
    )


def test_cohort_is_read_from_the_representations_own_layout(patients_dir, cohort):
    """The cohort comes from where the representation itself stores splits — the tokenization schema
    directories for an MTD artifact, the ``data/<split>/`` shard layout for a featurized one — and never
    from ``subject_splits.parquet``."""
    representation = read_manifest(patients_dir).get("representation", "mtd")
    layout = patients_dir / ("tokenization/schemas" if representation == "mtd" else "data")
    expected = {f.relative_to(layout).parts[0] for f in layout.rglob("*.parquet")}
    assert set(cohort) == expected
    assert all(cohort.values()), "every split should contribute at least one subject"


def test_the_artifact_carries_no_split_table(patients_dir):
    """Nothing reads it any more, so nothing writes it. The cohort is the only record of splits."""
    assert not (patients_dir / "metadata" / "subject_splits.parquet").exists()


def test_labels_are_partitioned_into_the_split_that_holds_the_subject(cohort):
    subjects = {split: sorted(s)[0] for split, s in sorted(cohort.items())}
    out = split_labels(_labels(subjects.values()), cohort)

    for split, subject in subjects.items():
        assert out[split]["subject_id"].to_list() == [subject]


def test_absent_subjects_are_dropped_with_a_warning(cohort, caplog):
    split, subjects = next(iter(sorted(cohort.items())))
    present = sorted(subjects)[:2]

    with caplog.at_level("WARNING"):
        out = split_labels(_labels([*present, ABSENT_SUBJECT]), cohort)

    assert sorted(out[split]["subject_id"].to_list()) == present
    assert "Dropping 1 label row(s) for 1 subject(s) absent from the patients cohort" in caplog.text


def test_a_complete_cohort_is_passed_through_silently(cohort, caplog):
    split, subjects = next(iter(sorted(cohort.items())))
    labels = _labels(sorted(subjects)[:3])

    with caplog.at_level("WARNING"):
        out = split_labels(labels, cohort)

    assert out[split].height == labels.height
    assert "Dropping" not in caplog.text, "nothing was filtered; the run must stay silent"


def test_raises_when_no_label_matches(cohort):
    """Labels for another dataset entirely: silently intersecting to nothing would be a plausible run."""
    with pytest.raises(TaskMaterializationError, match="No label row matched"):
        split_labels(_labels([ABSENT_SUBJECT]), cohort)


def test_materialize_drops_absent_subjects(patients_dir, cohort, tmp_path, caplog):
    """The regression, end to end through the command-facing entry point."""
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    for split, subjects in sorted(cohort.items()):
        _labels([*sorted(subjects)[:2], ABSENT_SUBJECT]).write_parquet(labels_dir / f"{split}.parquet")

    dest = tmp_path / "materialized"
    with caplog.at_level("WARNING"):
        materialize_labels(labels_dir, patients_dir, dest)

    for split in sorted(cohort):
        written = pl.read_parquet(dest / f"{split}.parquet")
        assert ABSENT_SUBJECT not in written["subject_id"].to_list()
    assert "Dropping" in caplog.text


def test_every_shard_is_read_whatever_it_is_named(tmp_path):
    """``meds-dev-task`` emits split-named *directories*; other producers emit other names.

    Verified against a real run: it executes ACES per input shard into
    ``{output_dir}/${data._prefix}.parquet``, so a split-sharded dataset yields ``train/0.parquet``. Those
    names are input *shard* names, coinciding with split names only because that dataset happened to be
    sharded by split — so they are concatenated, never interpreted.

    Subjects are disjoint per shard, as they are in a split-sharded dataset: shards are concatenated and
    then ``.unique()``d, so identical rows in two shards would collapse and mask a lost shard.
    """
    labels_dir = tmp_path / "meds_dev_labels"
    for i, split in enumerate(("train", "tuning", "held_out")):
        (labels_dir / split).mkdir(parents=True)
        _labels([2 * i + 1, 2 * i + 2]).write_parquet(labels_dir / split / "0.parquet")

    loaded = read_labels(labels_dir)
    assert loaded.height == 6, "every shard must be read, not just the first"
    assert sorted(loaded["subject_id"].to_list()) == [1, 2, 3, 4, 5, 6]


def test_dot_directories_are_not_label_shards(tmp_path):
    """``meds-dev-task`` always writes a ``.logs/`` beside its output. A run's logs are not label data."""
    labels_dir = tmp_path / "labels"
    (labels_dir / ".logs").mkdir(parents=True)
    _labels([1, 2]).write_parquet(labels_dir / "0.parquet")
    _labels([3, 4]).write_parquet(labels_dir / ".logs" / "junk.parquet")

    assert sorted(read_labels(labels_dir)["subject_id"].to_list()) == [1, 2]


def test_a_directory_with_no_labels_says_what_was_expected(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(TaskMaterializationError, match="no label parquet files"):
        read_labels(empty)


def test_a_non_label_parquet_names_the_file_and_its_columns(tmp_path):
    """Pointing at a MEDS data directory, or the patients artifact, is the likely mistake."""
    labels_dir = tmp_path / "not_labels"
    labels_dir.mkdir()
    pl.DataFrame({"subject_id": [1], "code": ["A"]}).write_parquet(labels_dir / "0.parquet")

    with pytest.raises(TaskMaterializationError) as excinfo:
        read_labels(labels_dir)
    assert "0.parquet" in str(excinfo.value)
    assert "prediction_time" in str(excinfo.value)


def test_the_cohort_reproduces_the_folder_layout_when_nothing_was_filtered(patients_dir, cohort, tmp_path):
    """Ignoring the folders costs nothing when they are right, which is the case worth pinning.

    ``meds-dev-task`` against a split-sharded dataset lays labels out under split-named directories. With
    no filtering pipeline, partitioning from the cohort must reproduce exactly that — otherwise "the
    folders are ignored" would be a behaviour change rather than a change of authority.
    """
    labels_dir = tmp_path / "by_folder"
    expected = {}
    for split, subjects in sorted(cohort.items()):
        chosen = sorted(subjects)[:2]
        (labels_dir / split).mkdir(parents=True)
        _labels(chosen).write_parquet(labels_dir / split / "0.parquet")
        expected[split] = chosen

    out = split_labels(read_labels(labels_dir), cohort)
    assert {s: sorted(df["subject_id"].to_list()) for s, df in out.items()} == expected


def test_a_label_filename_never_decides_a_split(patients_dir, cohort, tmp_path):
    """A flat ``train.parquet`` is a shard name like any other; the cohort decides where its rows go.

    Deliberate: meds-torch-data looks for a subject's tensor data under the split directory it was
    tokenized into, so a label filed under a split whose shards do not contain that subject would never
    match anything. Trusting the filename produces a silently unmatched label.
    """
    splits = sorted(cohort)
    subject = sorted(cohort[splits[0]])[0]

    labels_dir = tmp_path / "mislabelled"
    labels_dir.mkdir()
    # Filed under the *wrong* split's name, on purpose.
    _labels([subject]).write_parquet(labels_dir / f"{splits[-1]}.parquet")

    out = split_labels(read_labels(labels_dir), cohort)
    assert list(out) == [splits[0]], "the cohort must win over the filename"
