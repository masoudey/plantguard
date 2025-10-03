"""Evaluation metrics for PlantGuard models."""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def classification_report(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
    y_true = list(y_true)
    y_pred = list(y_pred)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def word_error_rate(reference: List[str], hypothesis: List[str]) -> float:
    """Compute word error rate for speech model evaluation."""
    distances = [
        _levenshtein(ref.split(), hyp.split()) / max(len(ref.split()), 1)
        for ref, hyp in zip(reference, hypothesis)
    ]
    return float(np.mean(distances)) if distances else 0.0


def _levenshtein(ref_tokens: List[str], hyp_tokens: List[str]) -> int:
    dp = [[0] * (len(hyp_tokens) + 1) for _ in range(len(ref_tokens) + 1)]
    for i in range(len(ref_tokens) + 1):
        dp[i][0] = i
    for j in range(len(hyp_tokens) + 1):
        dp[0][j] = j
    for i in range(1, len(ref_tokens) + 1):
        for j in range(1, len(hyp_tokens) + 1):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]
