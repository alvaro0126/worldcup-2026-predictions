"""Turn the fitted model + simulation into one recommended exact score per match.

Group matches use their real fixtures (played results are shown as-is; upcoming
games get the model's modal scoreline).  Knockout matches use the most-likely
matchup for that bracket slot across all simulations, then the modal scoreline
for that pairing.  Every row also carries a backup "score given the most-likely
outcome" for pools that score the result separately from the exact score.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from . import config
from .data import Match, Result
from .bayes import BayesModel
from .simulate import SimAgg


@dataclass(frozen=True)
class Rec:
    match_id: str
    rnd: str
    group: Optional[str]
    date: str
    home: str
    away: str
    venue: str
    status: str                 # final / in_progress / scheduled / projected
    rec_home: int               # score to enter in the pool
    rec_away: int
    p_home: float
    p_draw: float
    p_away: float
    exp_home: float
    exp_away: float
    best_outcome: str           # H / D / A
    alt_home: int               # backup: modal score given the best outcome
    alt_away: int
    actual_home: Optional[int]
    actual_away: Optional[int]
    matchup_prob: Optional[float]
    note: str


def _outcome_char(gh: int, ga: int) -> str:
    return "H" if gh > ga else ("A" if gh < ga else "D")


def group_recs(bm: BayesModel, fixtures: list[Match],
               results: list[Result]) -> list[Rec]:
    final_map = {(r.home, r.away): (r.home_goals, r.away_goals)
                 for r in results if r.status == "final"}
    live_map = {(r.home, r.away): (r.home_goals, r.away_goals)
                for r in results if r.status == "in_progress"}

    recs: list[Rec] = []
    group_fx = sorted((m for m in fixtures if m.rnd == "Group"),
                      key=lambda m: (m.date, m.match_id))
    for m in group_fx:
        pr = bm.predict(m.home, m.away, m.host_home, m.host_away)
        key = (m.home, m.away)
        actual_h = actual_a = None
        if key in final_map:
            gh, ga = final_map[key]
            status, rec_h, rec_a, actual_h, actual_a = "final", gh, ga, gh, ga
            hit = _outcome_char(*pr.modal) == _outcome_char(gh, ga)
            note = f"actual {gh}-{ga}; model said {pr.modal[0]}-{pr.modal[1]} " \
                   f"({'outcome hit' if hit else 'outcome miss'})"
        elif key in live_map:
            gh, ga = live_map[key]
            status, rec_h, rec_a, actual_h, actual_a = "in_progress", *pr.modal, gh, ga
            note = f"LIVE {gh}-{ga} — held out of the Bayesian update until full-time"
        else:
            status, rec_h, rec_a = "scheduled", pr.modal[0], pr.modal[1]
            note = "host advantage applied" if (m.host_home or m.host_away) else ""
        recs.append(Rec(
            match_id=m.match_id, rnd="Group", group=m.group, date=m.date,
            home=m.home, away=m.away, venue=m.venue, status=status,
            rec_home=rec_h, rec_away=rec_a,
            p_home=pr.p_home, p_draw=pr.p_draw, p_away=pr.p_away,
            exp_home=pr.exp_home, exp_away=pr.exp_away,
            best_outcome=pr.best_outcome,
            alt_home=pr.modal_given_outcome[0], alt_away=pr.modal_given_outcome[1],
            actual_home=actual_h, actual_away=actual_a,
            matchup_prob=None, note=note,
        ))
    return recs


def knockout_recs(bm: BayesModel, agg: SimAgg) -> list[Rec]:
    """One projected match per knockout slot (73..104), most-likely pairing."""
    recs: list[Rec] = []
    order = list(range(73, 103)) + [103, 104]
    for mn in order:
        pairs = agg.ko_pair.get(mn)
        if not pairs:
            continue
        (home, away), cnt = pairs.most_common(1)[0]
        prob = cnt / agg.n
        pr = bm.predict(home, away)        # knockouts are neutral venue
        rnd = config.round_of(mn)
        advancer = home if pr.p_home >= pr.p_away else away
        recs.append(Rec(
            match_id=f"M{mn}", rnd=rnd, group=None,
            date=config.ROUND_DATES.get(rnd, ""),
            home=home, away=away, venue="(projected)", status="projected",
            rec_home=pr.modal[0], rec_away=pr.modal[1],
            p_home=pr.p_home, p_draw=pr.p_draw, p_away=pr.p_away,
            exp_home=pr.exp_home, exp_away=pr.exp_away,
            best_outcome=pr.best_outcome,
            alt_home=pr.modal_given_outcome[0], alt_away=pr.modal_given_outcome[1],
            actual_home=None, actual_away=None, matchup_prob=prob,
            note=f"most-likely pairing ({prob:.0%}); {advancer} favoured to advance",
        ))
    return recs


def all_recs(bm: BayesModel, fixtures: list[Match], results: list[Result],
             agg: SimAgg) -> list[Rec]:
    return group_recs(bm, fixtures, results) + knockout_recs(bm, agg)
