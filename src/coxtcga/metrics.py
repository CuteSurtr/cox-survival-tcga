from __future__ import annotations
import numpy as np

def concordance_index(durations: np.ndarray, events: np.ndarray, risk_scores: np.ndarray) -> float:
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
            if d[i] == 1 and t[i] < t[j]:
                den += 1
                if eta[i] > eta[j]:
                    num += 1
                elif eta[i] == eta[j]:
                    num += 0.5
            elif d[i] == 1 and d[j] == 1 and (t[i] == t[j]):
                if i < j:
                    den += 1
                    num += 0.5
    return num / den if den > 0 else float('nan')
