"""Learnability assertions for the synthetic-data property test.

The property test's contract: on a dataset with a *designed* signal the model reaches high AUROC, while on
a *negative control* (shuffled labels) it does not — proving the model actually learned the signal rather
than the test being vacuous.
"""

from __future__ import annotations

import polars as pl


def binary_auroc(labels: list[bool] | pl.Series, scores: list[float] | pl.Series) -> float:
    """Compute ROC-AUC without sklearn (rank statistic; ties averaged).

    Returns 0.5 for a degenerate single-class label set.

    Examples:
        >>> binary_auroc([False, False, True, True], [0.1, 0.2, 0.8, 0.9])
        1.0
        >>> binary_auroc([True, False, True, False], [0.5, 0.5, 0.5, 0.5])
        0.5
        >>> round(binary_auroc([False, True, True], [0.3, 0.1, 0.9]), 4)
        0.5
    """
    y = list(labels)
    s = list(scores)
    n_pos = sum(1 for v in y if v)
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = sorted(range(len(s)), key=lambda i: s[i])
    ranks = [0.0] * len(s)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and s[order[j + 1]] == s[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank for the tie group
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    sum_pos_ranks = sum(r for r, v in zip(ranks, y, strict=False) if v)
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def assert_learns_signal(
    predictions: pl.DataFrame,
    labels: pl.DataFrame,
    *,
    min_auroc: float = 0.75,
    label_col: str = "boolean_value",
    score_col: str = "predicted_boolean_probability",
) -> float:
    """Assert the model separated the designed signal: ``AUROC(predictions, labels) >= min_auroc``.

    ``predictions`` (model output; ``subject_id, prediction_time, score_col``) is joined to ``labels``
    (the *known* synthetic ground truth we generated — not real test-set labels) on the two keys.
    Returns the achieved AUROC.
    """
    joined = predictions.join(labels, on=["subject_id", "prediction_time"], how="inner")
    assert len(joined) > 0, "predictions and labels did not join on (subject_id, prediction_time)"
    auroc = binary_auroc(joined[label_col].to_list(), joined[score_col].to_list())
    assert auroc >= min_auroc, (
        f"model failed to learn the designed signal: AUROC={auroc:.3f} < {min_auroc} "
        f"(n={len(joined)}). Either the model is broken or the training budget is too small."
    )
    return auroc
