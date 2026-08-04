"""Presence featurization: stamp one 0/1 column per ACES predicate onto MEDS data.

This is the ``featurization=predicates`` branch of ``preprocess_data`` (see
``docs/design-featurization.md`` in MEDS_model_template). A predicates YAML file — the same per-dataset
binding syntax MEDS-DEV ships for task extraction — is parsed down to its **presence subset**, and each
predicate becomes a dense ``predicate//<name>`` Int8 column: 1 where the event matches, 0 elsewhere.
Everything else about the data is untouched: no rows dropped, no columns modified, same shard layout.
The model decides downstream whether to read ``numeric_value`` on active rows and how to aggregate.

Supported predicate forms (deliberately *pure code matching* plus ``or()``):

- exact:   ``code: LAB//50912//mg/dL``
- regex:   ``code: { regex: "^ICU_ADMISSION//.*" }``   (``str.contains`` — anchors live in the pattern,
  matching ACES semantics)
- any-of:  ``code: { any: [HR, PULSE] }``
- derived: ``expr: or(creatinine_1, creatinine_2)``    (column-wise max of its inputs)

Everything else — value bounds, ``and()``, ``???`` placeholders, unknown keys — is *unsupported*:
skipped with a warning by default (cascading through ``or()`` expressions that reference a skipped
predicate), or a hard error under ``strict=True``. Skips are never silent: they are logged, returned to
the caller, and recorded in the artifact manifest. Value bounds in particular are a deliberate
omission, not a gap: a threshold is model-side semantics under presence featurization (the model reads
``numeric_value`` on active rows), and a bounded predicate whose bounds were quietly dropped would be a
lie under a trustworthy name.

The template does not depend on ``es-aces`` for this: the supported subset is a few polars expressions,
and the dependency would drag in its own ``meds`` pin.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import yaml

from .schemas import code_metadata_filepath, dataset_metadata_filepath

logger = logging.getLogger(__name__)

#: Every generated feature column is ``predicate//<name>``; the prefix is what keeps the columns out of
#: the way of MEDS's own (and any pipeline-added) columns, and what tests key on.
PREDICATE_COLUMN_PREFIX = "predicate//"

#: The ordered feature-space definition written at the artifact root. Feature order is part of what a
#: trained checkpoint means (feature *i* at training time must be the same predicate at predict time),
#: so consumers read this file rather than scanning parquet column order.
FEATURES_FILENAME = "features.json"

_OR_EXPR = re.compile(r"^or\(\s*([A-Za-z0-9_]+(?:\s*,\s*[A-Za-z0-9_]+)+)\s*\)$")


class PredicatesError(ValueError):
    """A predicates file (or the featurization it implies) is unusable, with the reason spelled out."""


@dataclass
class ParsedPredicates:
    """The outcome of parsing a predicates file down to the supported presence subset.

    ``exprs`` maps predicate name → boolean polars expression **over the base MEDS columns** for plain
    predicates, and → list of input names for derived ``or()`` predicates (resolved at featurize time,
    after the plain columns exist). ``order`` is the canonical feature order (file order, plains and
    deriveds interleaved as declared); ``skipped`` maps skipped predicate name → reason.
    """

    plain: dict[str, pl.Expr] = field(default_factory=dict)
    derived: dict[str, list[str]] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        return list(self.order)

    def column(self, name: str) -> str:
        return f"{PREDICATE_COLUMN_PREFIX}{name}"

    @property
    def columns(self) -> list[str]:
        return [self.column(n) for n in self.order]


def _parse_plain(name: str, spec: dict) -> pl.Expr | str:
    """One plain predicate → a boolean expression, or a reason string if unsupported."""
    extra = sorted(set(spec) - {"code"})
    if extra:
        return f"unsupported key(s) {', '.join(extra)} (presence featurization is code matching only)"
    code = spec["code"]
    match code:
        case str() if code:
            return pl.col("code") == code
        case {"regex": str() as pattern} if pattern and len(code) == 1:
            return pl.col("code").str.contains(pattern)
        case {"any": list() as options} if (
            options and len(code) == 1 and all(isinstance(o, str) for o in options)
        ):
            return pl.col("code").is_in(options)
        case _:
            return f"unsupported code form {code!r} (expected a string, {{regex: ...}} or {{any: [...]}})"


def parse_predicates(raw: dict, *, strict: bool = False, source: str = "predicates") -> ParsedPredicates:
    """Parse a loaded predicates mapping down to the supported presence subset.

    Args:
        raw: the ``predicates:`` mapping of a predicates YAML file (name → spec).
        strict: if True, any unsupported entry is a :class:`PredicatesError` listing every offender;
            if False (default), unsupported entries are skipped with a warning, cascading through
            ``or()`` expressions that reference them.
        source: where the mapping came from, for error messages.

    Raises:
        PredicatesError: on a non-mapping input, in strict mode on any unsupported entry, and always
            when nothing featurizable remains — an empty feature space is never a valid outcome.

    Examples:
        >>> parsed = parse_predicates({
        ...     "hr": {"code": "HR"},
        ...     "adm": {"code": {"regex": "^ADMISSION//.*"}},
        ...     "vitals": {"code": {"any": ["HR", "TEMP"]}},
        ...     "hr_or_adm": {"expr": "or(hr, adm)"},
        ... })
        >>> parsed.names
        ['hr', 'adm', 'vitals', 'hr_or_adm']
        >>> parsed.skipped
        {}

        Unsupported forms (here: value bounds, and()) skip with a cascade — ``high_or_low`` references a
        skipped predicate, so it is skipped too:

        >>> parsed = parse_predicates({
        ...     "hr": {"code": "HR"},
        ...     "high_hr": {"code": "HR", "value_min": 110},
        ...     "low_hr": {"code": "HR", "value_max": 40},
        ...     "high_or_low": {"expr": "or(high_hr, low_hr)"},
        ...     "both": {"expr": "and(hr, high_hr)"},
        ... })
        >>> parsed.names
        ['hr']
        >>> sorted(parsed.skipped)
        ['both', 'high_hr', 'high_or_low', 'low_hr']

        The same file under ``strict=True`` is an error listing every offender:

        >>> parse_predicates({"hr": {"code": "HR"}, "high_hr": {"code": "HR", "value_min": 110}},
        ...     strict=True)
        Traceback (most recent call last):
            ...
        meds_model_base.featurize.PredicatesError: predicates has unsupported predicate(s) ...high_hr...

        Nothing featurizable is always an error:

        >>> parse_predicates({"high_hr": {"code": "HR", "value_min": 110}})
        Traceback (most recent call last):
            ...
        meds_model_base.featurize.PredicatesError: No featurizable predicate remains ...
    """
    if not isinstance(raw, dict) or not raw:
        raise PredicatesError(f"{source} must be a non-empty mapping of predicate name -> spec.")

    parsed = ParsedPredicates()
    pending_derived: dict[str, list[str]] = {}

    for name, spec in raw.items():
        if not isinstance(spec, dict):
            parsed.skipped[name] = f"spec is {type(spec).__name__}, expected a mapping"
        elif "code" in spec:
            result = _parse_plain(name, spec)
            if isinstance(result, str):
                parsed.skipped[name] = result
            else:
                parsed.plain[name] = result
        elif "expr" in spec:
            extra = sorted(set(spec) - {"expr"})
            m = _OR_EXPR.match(str(spec["expr"]).strip()) if not extra else None
            if extra:
                parsed.skipped[name] = f"unsupported key(s) {', '.join(extra)} on a derived predicate"
            elif m is None:
                parsed.skipped[name] = f"unsupported expr {spec['expr']!r} (only or(a, b, ...) is supported)"
            else:
                pending_derived[name] = [p.strip() for p in m.group(1).split(",")]
        else:
            parsed.skipped[name] = "neither a plain (code:) nor a derived (expr:) predicate"

    # Resolve derived predicates against what survived, iterating so or() over or() works; anything
    # referencing a skipped or unknown name cascades into the skip list with the reference named.
    progressed = True
    while pending_derived and progressed:
        progressed = False
        for name, inputs in list(pending_derived.items()):
            if all(i in parsed.plain or i in parsed.derived for i in inputs):
                parsed.derived[name] = inputs
                del pending_derived[name]
                progressed = True
    for name, inputs in pending_derived.items():
        missing = [i for i in inputs if i not in parsed.plain and i not in parsed.derived]
        parsed.skipped[name] = "references skipped or unknown predicate(s) " + ", ".join(missing)

    parsed.order = [n for n in raw if n in parsed.plain or n in parsed.derived]

    if parsed.skipped:
        listing = "; ".join(f"{n}: {why}" for n, why in parsed.skipped.items())
        if strict:
            raise PredicatesError(
                f"{source} has unsupported predicate(s) under featurization_strict: {listing}"
            )
        logger.warning(
            "Skipping %d unsupported predicate(s) from %s (featurization_strict=false): %s",
            len(parsed.skipped),
            source,
            listing,
        )

    if not parsed.order:
        raise PredicatesError(
            f"No featurizable predicate remains in {source}: every entry was skipped "
            f"({'; '.join(f'{n}: {why}' for n, why in parsed.skipped.items())}). "
            "An empty feature space is never a valid artifact."
        )
    return parsed


def read_predicates_file(path: Path | str, *, strict: bool = False) -> ParsedPredicates:
    """Read and parse a predicates YAML file (a mapping with a ``predicates:`` key, as ACES files are)."""
    path = Path(path)
    if not path.is_file():
        raise PredicatesError(f"external_predicates_file {path} does not exist.")
    try:
        loaded = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise PredicatesError(f"Could not parse {path}: {e}") from e
    if not isinstance(loaded, dict) or "predicates" not in loaded:
        raise PredicatesError(
            f"{path} has no top-level 'predicates:' mapping; it does not look like a predicates file."
        )
    return parse_predicates(loaded["predicates"], strict=strict, source=str(path))


def predicates_digest(path: Path | str) -> str:
    """sha256 of the predicates file bytes — the manifest's record of *which* predicates built an artifact.

    The manifest otherwise records only a path to a file that lives outside the artifact and can change
    after the build; the digest is what makes two runs distinguishable and a drifted file detectable.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def literal_code_predicates(parsed: ParsedPredicates, raw: dict) -> dict[str, list[str]]:
    """The featurized predicates whose codes can be *emitted* (exact / any-of), name → codes.

    A regex cannot be reverse-instantiated into a code, so this is the subset the synthetic
    signal-dataset builder can plant events for (see ``testing.synthetic``).

    Examples:
        >>> raw = {"hr": {"code": "HR"}, "vitals": {"code": {"any": ["HR", "TEMP"]}},
        ...        "adm": {"code": {"regex": "^ADM.*"}}}
        >>> literal_code_predicates(parse_predicates(raw), raw)
        {'hr': ['HR'], 'vitals': ['HR', 'TEMP']}
    """
    out: dict[str, list[str]] = {}
    for name in parsed.order:
        if name not in parsed.plain:
            continue
        code = raw[name]["code"]
        if isinstance(code, str):
            out[name] = [code]
        elif isinstance(code, dict) and "any" in code:
            out[name] = list(code["any"])
    return out


def featurize_frame(df: pl.DataFrame, parsed: ParsedPredicates) -> pl.DataFrame:
    """Append the predicate columns to one MEDS frame. Pure augmentation — existing columns untouched.

    Examples:
        >>> df = pl.DataFrame({"subject_id": [1, 1, 2], "code": ["HR", "DISCHARGE", "TEMP"],
        ...                    "numeric_value": [99.0, None, 37.0]})
        >>> parsed = parse_predicates({"hr": {"code": "HR"}, "temp": {"code": "TEMP"},
        ...                            "vital": {"expr": "or(hr, temp)"}})
        >>> out = featurize_frame(df, parsed)
        >>> out.columns
        ['subject_id', 'code', 'numeric_value', 'predicate//hr', 'predicate//temp', 'predicate//vital']
        >>> out["predicate//hr"].to_list(), out["predicate//vital"].to_list()
        ([1, 0, 0], [1, 0, 1])
        >>> out["predicate//hr"].dtype
        Int8
    """
    collisions = sorted(set(parsed.columns) & set(df.columns))
    if collisions:
        raise PredicatesError(
            f"Input already carries predicate column(s) {', '.join(collisions)} — refusing to overwrite "
            "(was this data already featurized?)."
        )
    df = df.with_columns(
        [expr.cast(pl.Int8).fill_null(0).alias(parsed.column(n)) for n, expr in parsed.plain.items()]
    )
    for name in parsed.order:  # file order; or() over or() resolves because inputs precede by iteration
        if name in parsed.derived:
            df = df.with_columns(
                pl.max_horizontal([pl.col(parsed.column(i)) for i in parsed.derived[name]])
                .cast(pl.Int8)
                .alias(parsed.column(name))
            )
    return df


def featurize_dataset(input_dir: Path, staging: Path, parsed: ParsedPredicates) -> dict[str, int]:
    """Featurize every ``data/<split>/…`` shard of ``input_dir`` into the same layout under ``staging``.

    Copies ``metadata/codes.parquet`` and ``metadata/dataset.json`` through. Deliberately does **not**
    copy ``metadata/subject_splits.parquet``: split membership travels as the shard layout and nothing
    else — a copied splits table describes the *input* to preprocessing, and a filtering pipeline makes
    those two different things (the same invariant the MTD branch keeps).

    Returns per-predicate match counts across all shards (events where the column is 1), which the
    caller logs and folds into the manifest.
    """
    import shutil

    data_in = Path(input_dir) / "data"
    counts: dict[str, int] = dict.fromkeys(parsed.order, 0)
    n_shards = 0
    for fp in sorted(data_in.rglob("*.parquet")):
        rel = fp.relative_to(data_in)
        if any(part.startswith(".") for part in rel.parts):
            continue
        out = featurize_frame(pl.read_parquet(fp), parsed)
        dest = staging / "data" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        out.write_parquet(dest)
        n_shards += 1
        for name in parsed.order:
            counts[name] += int(out[parsed.column(name)].sum())

    if not n_shards:
        raise PredicatesError(f"No data shards found under {data_in}; nothing to featurize.")

    for rel in (code_metadata_filepath, dataset_metadata_filepath):
        src = Path(input_dir) / rel
        if src.is_file():
            dest = staging / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)

    (staging / FEATURES_FILENAME).write_text(
        json.dumps(
            {
                "version": 1,
                "features": [{"name": n, "column": parsed.column(n)} for n in parsed.order],
            },
            indent=2,
        )
        + "\n"
    )

    unmatched = sorted(n for n, c in counts.items() if c == 0)
    if unmatched:
        # A warning, not an error: on real data it signals a binding mismatch worth reading, but it is
        # also the expected outcome when real-vocabulary predicates run over synthetic test data.
        logger.warning(
            "%d of %d predicate(s) matched no event at all: %s. If this is real data, check the "
            "bindings; over a synthetic or foreign-vocabulary dataset this is expected.",
            len(unmatched),
            len(counts),
            ", ".join(unmatched),
        )
    return counts


def load_features(patients_dir: Path | str) -> list[dict[str, str]]:
    """The ordered feature definitions of a featurized patients artifact (``features.json``).

    This — not parquet column order, not the predicates YAML — is what a datamodule reads to learn the
    feature space: ``vocab_size = len(load_features(dir))``, columns selected in this order.
    """
    fp = Path(patients_dir) / FEATURES_FILENAME
    if not fp.is_file():
        raise PredicatesError(
            f"No {FEATURES_FILENAME} in {patients_dir}; it is not a featurized patients artifact."
        )
    return json.loads(fp.read_text())["features"]


__all__ = [
    "FEATURES_FILENAME",
    "PREDICATE_COLUMN_PREFIX",
    "ParsedPredicates",
    "PredicatesError",
    "featurize_dataset",
    "featurize_frame",
    "literal_code_predicates",
    "load_features",
    "parse_predicates",
    "predicates_digest",
    "read_predicates_file",
]
