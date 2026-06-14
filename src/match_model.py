"""Dixon-Coles bivariate-Poisson match engine.

Two responsibilities:
  * turn a pair of (effective) Elo ratings into expected goals (lambda_home,
    lambda_away) via the calibrated log-linear map, and
  * turn a pair of lambdas into a full score-probability matrix with the
    Dixon-Coles (1997) low-score correction, plus the summaries we report
    (W/D/L, modal scoreline, expected goals).

The engine is deliberately pure: no I/O, no globals beyond config constants.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.stats import poisson

from . import config


@dataclass(frozen=True)
class Params:
    alpha: float   # base log goal-rate (sets scoring level at an even game)
    b: float       # Elo sensitivity: log goal-supremacy per 100 Elo / 2
    rho: float     # Dixon-Coles low-score dependence (negative => more draws)


def lambdas_from_elo(elo_h: float, elo_a: float,
                     host_h: bool, host_a: bool, p: Params) -> tuple[float, float]:
    """Expected goals for home/away from their Elo ratings and host status."""
    eff_h = elo_h + (config.HOST_ELO_BONUS if host_h else 0.0)
    eff_a = elo_a + (config.HOST_ELO_BONUS if host_a else 0.0)
    delta = (eff_h - eff_a) / 100.0
    lam_h = np.exp(p.alpha + p.b * delta)
    lam_a = np.exp(p.alpha - p.b * delta)
    return float(lam_h), float(lam_a)


def score_matrix(lam_h: float, lam_a: float, rho: float,
                 max_goals: int = config.MAX_GOALS) -> np.ndarray:
    """Joint P(home=x, away=y) as a (max_goals+1) x (max_goals+1) matrix."""
    k = np.arange(max_goals + 1)
    ph = poisson.pmf(k, lam_h)
    pa = poisson.pmf(k, lam_a)
    m = np.outer(ph, pa)                      # independent-Poisson baseline
    # Dixon-Coles correction on the four low-score cells
    m[0, 0] *= 1.0 - lam_h * lam_a * rho
    m[0, 1] *= 1.0 + lam_h * rho
    m[1, 0] *= 1.0 + lam_a * rho
    m[1, 1] *= 1.0 - rho
    m = np.clip(m, 0.0, None)                 # guard against extreme rho
    total = m.sum()
    if total <= 0:
        raise ValueError("degenerate score matrix")
    return m / total


def outcome_probs(m: np.ndarray) -> tuple[float, float, float]:
    """(P home win, P draw, P away win)."""
    home = float(np.tril(m, -1).sum())        # x > y
    draw = float(np.trace(m))                 # x == y
    away = float(np.triu(m, 1).sum())         # x < y
    return home, draw, away


def modal_score(m: np.ndarray) -> tuple[int, int]:
    """Most likely exact scoreline (argmax of the joint matrix)."""
    x, y = np.unravel_index(int(np.argmax(m)), m.shape)
    return int(x), int(y)


def score_given_outcome(m: np.ndarray, outcome: str) -> tuple[int, int]:
    """Most likely exact scoreline restricted to a given outcome H/D/A."""
    mask = np.zeros_like(m, dtype=bool)
    idx = np.indices(m.shape)
    if outcome == "H":
        mask = idx[0] > idx[1]
    elif outcome == "A":
        mask = idx[0] < idx[1]
    else:
        mask = idx[0] == idx[1]
    masked = np.where(mask, m, -1.0)
    x, y = np.unravel_index(int(np.argmax(masked)), m.shape)
    return int(x), int(y)


def expected_goals(m: np.ndarray) -> tuple[float, float]:
    k = np.arange(m.shape[0])
    eh = float((m.sum(axis=1) * k).sum())
    ea = float((m.sum(axis=0) * k).sum())
    return eh, ea


@dataclass(frozen=True)
class Prediction:
    lam_home: float
    lam_away: float
    p_home: float
    p_draw: float
    p_away: float
    modal: tuple[int, int]
    modal_given_outcome: tuple[int, int]
    exp_home: float
    exp_away: float

    @property
    def best_outcome(self) -> str:
        return "H" if self.p_home >= max(self.p_draw, self.p_away) else (
            "A" if self.p_away >= self.p_draw else "D")


def predict(lam_h: float, lam_a: float, rho: float) -> Prediction:
    m = score_matrix(lam_h, lam_a, rho)
    ph, pd_, pa = outcome_probs(m)
    modal = modal_score(m)
    eh, ea = expected_goals(m)
    best = "H" if ph >= max(pd_, pa) else ("A" if pa >= pd_ else "D")
    return Prediction(
        lam_home=lam_h, lam_away=lam_a,
        p_home=ph, p_draw=pd_, p_away=pa,
        modal=modal, modal_given_outcome=score_given_outcome(m, best),
        exp_home=eh, exp_away=ea,
    )


def predict_from_elo(elo_h: float, elo_a: float, host_h: bool, host_a: bool,
                     p: Params) -> Prediction:
    lam_h, lam_a = lambdas_from_elo(elo_h, elo_a, host_h, host_a, p)
    return predict(lam_h, lam_a, p.rho)
