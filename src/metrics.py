"""Out-of-sample validation on the matches already played.

Each completed match is scored against the **pre-tournament** Elo prior (the
model *before* it saw that result), so this is an honest hold-out check rather
than an in-sample fit.  We report outcome hit-rate and the Ranked Probability
Score (RPS; lower is better, good tournament forecasts sit around 0.17-0.18).
"""
from __future__ import annotations
from dataclasses import dataclass

from .data import Team, Result
from .match_model import Params, predict_from_elo


def _rps(p_home: float, p_draw: float, p_away: float, outcome: str) -> float:
    """Ranked Probability Score for an ordered 3-outcome forecast (H, D, A)."""
    a_home = 1.0 if outcome == "H" else 0.0
    a_draw = 1.0 if outcome == "D" else 0.0
    c1p, c2p = p_home, p_home + p_draw
    c1a, c2a = a_home, a_home + a_draw
    return 0.5 * ((c1p - c1a) ** 2 + (c2p - c2a) ** 2)


@dataclass(frozen=True)
class Validation:
    n: int
    outcome_hit_rate: float
    exact_hit_rate: float
    mean_rps: float
    rows: list


def evaluate_prior(teams: dict[str, Team], results: list[Result],
                   params: Params) -> Validation:
    rows, hits, exact, rps_sum = [], 0, 0, 0.0
    finals = [r for r in results if r.status == "final"]
    for r in finals:
        pr = predict_from_elo(teams[r.home].elo, teams[r.away].elo,
                              r.host_home, r.host_away, params)
        actual = "H" if r.home_goals > r.away_goals else (
            "A" if r.home_goals < r.away_goals else "D")
        pred = pr.best_outcome
        hit = int(pred == actual)
        ex = int(pr.modal == (r.home_goals, r.away_goals))
        rps = _rps(pr.p_home, pr.p_draw, pr.p_away, actual)
        hits += hit
        exact += ex
        rps_sum += rps
        rows.append({
            "match": f"{r.home} {r.home_goals}-{r.away_goals} {r.away}",
            "pred_outcome": pred, "actual_outcome": actual,
            "model_score": f"{pr.modal[0]}-{pr.modal[1]}",
            "p_home": pr.p_home, "p_draw": pr.p_draw, "p_away": pr.p_away,
            "rps": rps, "outcome_hit": bool(hit),
        })
    n = len(finals)
    return Validation(
        n=n,
        outcome_hit_rate=hits / n if n else 0.0,
        exact_hit_rate=exact / n if n else 0.0,
        mean_rps=rps_sum / n if n else 0.0,
        rows=rows,
    )
