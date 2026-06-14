"""Load and normalise the input data.

Produces immutable snapshots (plain dicts / tuples) consumed by the rest of the
pipeline.  The one piece of real work here is putting every team on a single,
canonical Elo scale (eloratings.net), bridging the teams for which we only have
the Metis model's Elo via a linear regression on the ~18 overlapping teams.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

from . import config


HOST_COUNTRY = {"Mexico": "MEX", "Canada": "CAN", "United States": "USA"}


@dataclass(frozen=True)
class Team:
    name: str
    group: str
    confederation: str
    host: bool
    elo: float                 # canonical (eloratings.net) scale
    elo_source: str            # "eloratings.net" or "metis->bridged"
    elo_metis: Optional[float]
    fifa_rank: Optional[int]
    nivel: Optional[float]     # El País composite "nivel" (cross-check only)


@dataclass(frozen=True)
class Match:
    match_id: str
    rnd: str
    group: Optional[str]
    date: str
    home: str
    away: str
    venue: str
    venue_country: str
    host_home: bool
    host_away: bool


@dataclass(frozen=True)
class Result:
    date: str
    home: str
    away: str
    home_goals: int
    away_goals: int
    status: str
    host_home: bool
    host_away: bool


def _host_flags(home: str, away: str, venue_country: str) -> tuple[bool, bool]:
    hh = home in HOST_COUNTRY and venue_country == HOST_COUNTRY[home]
    ha = away in HOST_COUNTRY and venue_country == HOST_COUNTRY[away]
    return hh, ha


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares y ~ intercept + slope * x ; returns (intercept, slope)."""
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept), float(slope)


def _blend_elo(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return (canonical_elo, source) Series on the eloratings.net scale.

    The 18 teams that have an eloratings.net number use it directly (the gold
    standard both source documents rely on).  Every other team is the *average*
    of whatever independent signals it has, each first mapped onto the
    eloratings scale by a regression fit on the teams that have both:
        * Metis-model Elo,
        * FIFA world ranking (via log-rank), and
        * El País "nivel".
    Blending several sources is far more robust than trusting any single one
    (e.g. Metis alone badly underrates the USA; FIFA rank corrects it).
    """
    base = df.dropna(subset=["elo_eloratings"])
    m = base.dropna(subset=["elo_metis"])
    a_m, b_m = _fit_line(m["elo_metis"].to_numpy(float),
                         m["elo_eloratings"].to_numpy(float))
    r = base.dropna(subset=["fifa_rank"])
    a_r, b_r = _fit_line(np.log(r["fifa_rank"].to_numpy(float)),
                         r["elo_eloratings"].to_numpy(float))
    nv = base.dropna(subset=["elp_nivel"])
    a_n, b_n = _fit_line(nv["elp_nivel"].to_numpy(float),
                         nv["elo_eloratings"].to_numpy(float))

    def blend(row) -> tuple[float, str]:
        if not pd.isna(row["elo_eloratings"]):
            return float(row["elo_eloratings"]), "eloratings.net"
        ests, srcs = [], []
        if not pd.isna(row["elo_metis"]):
            ests.append(a_m + b_m * row["elo_metis"]); srcs.append("metis")
        if not pd.isna(row["fifa_rank"]):
            ests.append(a_r + b_r * np.log(row["fifa_rank"])); srcs.append("fifa")
        if not pd.isna(row["elp_nivel"]):
            ests.append(a_n + b_n * row["elp_nivel"]); srcs.append("nivel")
        return float(np.mean(ests)), "blend(" + "+".join(srcs) + ")"

    blended = df.apply(blend, axis=1)
    return (blended.map(lambda t: t[0]), blended.map(lambda t: t[1]))


def load_teams() -> dict[str, Team]:
    df = pd.read_csv(config.RATINGS_CSV)
    df["team"] = df["team"].map(config.canon)
    elo, source = _blend_elo(df)
    df = df.assign(elo=elo, elo_source=source)
    teams: dict[str, Team] = {}
    for _, r in df.iterrows():
        teams[r["team"]] = Team(
            name=r["team"],
            group=r["group"],
            confederation=r["confederation"],
            host=bool(r["host"]),
            elo=float(r["elo"]),
            elo_source=r["elo_source"],
            elo_metis=None if pd.isna(r["elo_metis"]) else float(r["elo_metis"]),
            fifa_rank=None if pd.isna(r["fifa_rank"]) else int(r["fifa_rank"]),
            nivel=None if pd.isna(r["elp_nivel"]) else float(r["elp_nivel"]),
        )
    if len(teams) != 48:
        raise ValueError(f"expected 48 teams, got {len(teams)}")
    return teams


def load_fixtures() -> list[Match]:
    df = pd.read_csv(config.FIXTURES_CSV)
    for col in ("home", "away"):
        df[col] = df[col].map(config.canon)
    out: list[Match] = []
    for _, r in df.iterrows():
        hh, ha = _host_flags(r["home"], r["away"], r["venue_country"])
        out.append(Match(
            match_id=r["match_id"], rnd=r["round"],
            group=None if pd.isna(r["group"]) else r["group"],
            date=r["date"], home=r["home"], away=r["away"],
            venue=r["venue"], venue_country=r["venue_country"],
            host_home=hh, host_away=ha,
        ))
    return out


def load_results(fixtures: list[Match]) -> list[Result]:
    df = pd.read_csv(config.RESULTS_CSV)
    for col in ("home", "away"):
        df[col] = df[col].map(config.canon)
    venue_by_pair = {(m.home, m.away): m.venue_country for m in fixtures}
    out: list[Result] = []
    for _, r in df.iterrows():
        vc = venue_by_pair.get((r["home"], r["away"]), "NEU")
        hh, ha = _host_flags(r["home"], r["away"], vc)
        out.append(Result(
            date=r["date"], home=r["home"], away=r["away"],
            home_goals=int(r["home_goals"]), away_goals=int(r["away_goals"]),
            status=str(r["status"]).strip().lower(),
            host_home=hh, host_away=ha,
        ))
    return out


def final_results(results: list[Result]) -> list[Result]:
    """Only completed matches feed the Bayesian update (live games are held)."""
    return [r for r in results if r.status == "final"]
