"""The synthetic cohorts carry static measurements, and a batch built from them can be collated.

Needs no model, so this runs from the moment a repository is generated.

The bug it pins is quiet in the worst way: it does not show up until the *slow* tier, and only for models
that read baseline data. `build_signal_dataset` used to emit nothing but timestamped events, so a cohort
built from it had no static measurements at all — and meds-torch-data cannot collate that under
`static_inclusion_mode: INCLUDE`:

    File ".../meds_torchdata/pytorch_dataset.py", line 1822, in collate
        static_data = JointNestedRaggedTensorDict(
    ValueError: Cannot infer dtype from empty values; provide an explicit `schema=`.

The model is never called, so the failure names the collator rather than the fixture that caused it. A
model whose datamodule sets `INCLUDE` — anything that reads age, sex, or any other baseline variable —
would pass every fast test and then fail the one tier that proves it learns.
"""

import polars as pl
import pytest
from meds_model_base.schemas import code_metadata_filepath
from meds_model_base.testing import (
    STATIC_CODES,
    STATIC_GROUPS,
    STATIC_NUMERIC,
    build_signal_dataset,
    build_workspace,
)

#: Small enough to tensorize in a couple of seconds; large enough for both static groups to appear on both
#: sides of the label, which is what the leakage check below needs.
COHORT = {"n_train": 24, "n_tuning": 8, "n_held_out": 8}


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    return build_signal_dataset(tmp_path_factory.mktemp("statics") / "raw", seed=0, **COHORT)


def _events(root) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet"))])


def test_every_subject_has_both_kinds_of_static_measurement(raw):
    """One value-less code and one numeric one, so both branches of a model's static handling are covered."""
    events = _events(raw)
    statics = events.filter(pl.col("time").is_null())
    assert len(statics) > 0, "the cohort has no static measurements at all"

    subjects = set(events["subject_id"].to_list())
    assert set(statics["subject_id"].to_list()) == subjects, "not every subject has static measurements"

    by_subject = statics.group_by("subject_id").agg(pl.col("code"))
    for codes in by_subject["code"].to_list():
        assert any(code in STATIC_GROUPS for code in codes), f"no categorical static among {codes}"
        assert STATIC_NUMERIC in codes, f"no numeric static among {codes}"

    numeric = statics.filter(pl.col("code") == STATIC_NUMERIC)
    assert numeric["numeric_value"].null_count() == 0, "the numeric static carries no value"
    assert statics.filter(pl.col("code").is_in(STATIC_GROUPS))["numeric_value"].null_count() == len(
        subjects
    ), "the categorical static should carry no value"


def test_static_codes_are_declared_in_the_metadata(raw):
    """A code absent from `metadata/codes.parquet` is a code the vocabulary will not have."""
    declared = set(pl.read_parquet(raw / code_metadata_filepath)["code"].to_list())
    assert set(STATIC_CODES) <= declared


def test_statics_carry_no_information_about_the_label(raw):
    """The baseline variables must not be a second signal.

    The designed-signal dataset exists to check that a model reads `SIGNAL_CODE`. A static that correlated
    with the outcome would let a model score well without ever reading it — and would give the negative
    control something real to learn, which is exactly the vacuous pass the control exists to rule out.
    """
    labels = pl.concat(
        [pl.read_parquet(p) for p in sorted((raw / "task_labels" / "signal_task").glob("*.parquet"))]
    )
    groups = _events(raw).filter(pl.col("code").is_in(STATIC_GROUPS)).select("subject_id", "code")
    joined = labels.join(groups, on="subject_id", how="inner")

    # Perfect separation would mean the group *is* the label. Both groups appearing on both sides is a
    # cheap, non-flaky refutation of that.
    seen = {(row["code"], row["boolean_value"]) for row in joined.iter_rows(named=True)}
    for code in STATIC_GROUPS:
        assert (code, True) in seen, f"{code} never occurs with a positive label"
        assert (code, False) in seen, f"{code} never occurs with a negative label"


def test_a_cohort_without_statics_is_refused_before_training(tmp_path_factory):
    """The half the fixture change does not fix: a *real* cohort with no static measurements.

    Giving the synthetic builders statics stops this suite tripping the collate crash, but it does nothing
    for a user whose dataset genuinely has no baseline variables — they still hit
    ``Cannot infer dtype from empty values`` inside the dataloader, naming neither the config key that
    asked for static data nor the cohort that lacks it. `build_datamodule` is the one place every command
    constructs a datamodule, so the mismatch is refused there instead, with both ways out named.
    """
    from meds_model_base.lightning import build_datamodule, require_statics_if_requested
    from omegaconf import OmegaConf

    root = tmp_path_factory.mktemp("no_statics")
    bare = build_signal_dataset(root / "raw", seed=0, with_statics=False, **COHORT)
    patients = build_workspace(bare, root / "data") / "patients"

    cfg = OmegaConf.create(
        {
            "datamodule": {
                "config": {"tensorized_cohort_dir": str(patients), "static_inclusion_mode": "INCLUDE"}
            }
        }
    )
    with pytest.raises(ValueError, match=r"no subject in .* has any static measurement"):
        require_statics_if_requested(cfg)
    # …and it is reached through the function every command actually calls, not only directly.
    with pytest.raises(ValueError, match=r"static_inclusion_mode=OMIT"):
        build_datamodule(cfg)

    # OMIT is the documented way out, and must not be blocked by the check.
    cfg.datamodule.config.static_inclusion_mode = "OMIT"
    require_statics_if_requested(cfg)


def test_a_batch_collates_with_statics_included(raw, tmp_path):
    """The regression itself: `INCLUDE` must survive collation, and the tensors must be populated.

    This is the assertion the crash would fail. It runs the real preprocessing, so it also proves the
    static measurements survive tensorization rather than merely existing in the raw parquet.
    """
    import torch
    from meds_torchdata import MEDSPytorchDataset, MEDSTorchDataConfig
    from meds_torchdata.types import StaticInclusionMode, SubsequenceSamplingStrategy

    patients = build_workspace(raw, tmp_path / "data") / "patients"
    dataset = MEDSPytorchDataset(
        MEDSTorchDataConfig(
            tensorized_cohort_dir=str(patients),
            max_seq_len=32,
            seq_sampling_strategy=SubsequenceSamplingStrategy.TO_END,
            static_inclusion_mode=StaticInclusionMode.INCLUDE,
        ),
        split="train",
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=dataset.collate, num_workers=0)
    batch = next(iter(loader))

    assert batch.static_code is not None, "the batch carries no static tensors"
    assert batch.static_code.numel() > 0, "the static tensors are empty"
    assert (batch.static_code != 0).any(), "every static code is padding"
    # The numeric baseline variable must arrive as a value, not merely as a code.
    assert batch.static_numeric_value_mask.any(), "no static measurement carries a numeric value"
