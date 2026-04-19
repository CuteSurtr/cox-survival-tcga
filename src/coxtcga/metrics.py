"""Concordance index (Harrell's c-index) for risk scores.

Definition
----------
Given risk scores eta_i and observed (time, event) pairs, the c-index is

    c = sum_{(i, j) : i is comparable to j, eta_i > eta_j} 1
      / #{comparable (i, j) pairs}

A pair (i, j) is *comparable* when we can unambiguously order their event
times. With right-censoring the standard rule is:

  - if i has an event at t_i and t_i < t_j (j may be censored or event): (i, j) comparable, i should rank higher-risk.
  - if both have events and t_i == t_j: ties are counted as 0.5.
  - if i is censored and j is censored with t_i != t_j: not comparable.

c = 0.5 is chance; c = 1 is perfect risk ranking.

We implement the O(n^2) version which is fine for n up to a few thousand
(the TCGA survival subsamples we work with fit easily).
"""

from __future__ import annotations

import numpy as np


def concordance_index(
    durations: np.ndarray,
    events: np.ndarray,
    risk_scores: np.ndarray,
) -> float:
    t = np.asarray(durations, dtype=np.float64)
    d = np.asarray(events, dtype=np.float64)
    eta = np.asarray(risk_scores, dtype=np.float64)
    n = len(t)
    num = 0.0
    den = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Ensure (i, j) has i as the "earlier or event" subject.
            if d[i] == 1 and t[i] < t[j]:
                den += 1
                if eta[i] > eta[j]:
                    num += 1
                elif eta[i] == eta[j]:
                    num += 0.5
            elif d[i] == 1 and d[j] == 1 and t[i] == t[j]:
                # Tie in event times: treated as 0.5 for either ordering.
                # Count this pair once from the (i, j) perspective only.
                if i < j:
                    den += 1
                    num += 0.5
    return num / den if den > 0 else float("nan")
