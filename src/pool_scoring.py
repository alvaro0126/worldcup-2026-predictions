"""Pick the scoreline that maximises expected points under the pool's rules.

The Bodytech pool does not reward "most likely exact score"; it gives partial
credit for the right winner and, crucially, more for the right goal difference
(100) than the winner alone (70).  So the points-maximising prediction is rarely
a cautious 1-1: for any favourite a decisive scoreline earns far more on average.

For a match we have the model's full score-probability matrix P(x, y).  The
expected points of a candidate prediction (px, py) is

    E[pts] = sum_{x,y} P(x, y) * points(px, py ; x, y)

and we return the (px, py) that maximises it.
"""
from __future__ import annotations
import numpy as np

from . import config


def match_points(px: int, py: int, x: int, y: int,
                 pts: dict = config.POOL_POINTS) -> int:
    """Points scored by predicting (px, py) when the result is (x, y)."""
    if px == x and py == y:
        return pts["exact"]
    pred_diff, act_diff = px - py, x - y
    if px == py:                      # predicted a draw
        return pts["draw"] if act_diff == 0 else pts["miss"]
    if (pred_diff > 0) == (act_diff > 0) and act_diff != 0:   # same winner
        return pts["winner_diff"] if pred_diff == act_diff else pts["winner_only"]
    return pts["miss"]


def expected_points(px: int, py: int, matrix: np.ndarray,
                    pts: dict = config.POOL_POINTS) -> float:
    n = matrix.shape[0]
    total = 0.0
    for x in range(n):
        for y in range(n):
            p = matrix[x, y]
            if p > 0:
                total += p * match_points(px, py, x, y, pts)
    return float(total)


def optimal_prediction(matrix: np.ndarray, pts: dict = config.POOL_POINTS,
                       max_goals: int = config.POOL_PRED_MAX_GOALS,
                       tol: float = 1.0) -> tuple[int, int, float]:
    """Return (home, away, expected_points) of the points-maximising prediction.

    Among scorelines within `tol` points of the best expected value, prefer the
    one with the highest exact-hit probability (more natural scorelines, a touch
    more upside) at negligible expected-value cost.
    """
    cands = [(expected_points(px, py, matrix, pts), float(matrix[px, py]), px, py)
             for px in range(max_goals + 1) for py in range(max_goals + 1)]
    best_ev = max(c[0] for c in cands)
    near = [c for c in cands if c[0] >= best_ev - tol]
    near.sort(key=lambda c: c[1], reverse=True)      # highest exact probability
    ev, _exact, px, py = near[0]
    return px, py, ev
