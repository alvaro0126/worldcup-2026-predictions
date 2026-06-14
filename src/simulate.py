"""Monte-Carlo tournament simulator.

Each simulation:
  * plays every group match (FINAL results are fixed; the rest are sampled from
    the posterior-sampled team strengths),
  * ranks each group by the FIFA tie-breakers (points, GD, GF, head-to-head),
  * ranks the twelve third-placed teams and takes the best eight,
  * allocates those eight to the bracket's third-place slots (respecting the
    group constraints) via an assignment solver,
  * plays the knockout bracket (draws resolved by a strength-weighted shootout).

Aggregated over many runs it yields advancement / title probabilities and the
most-likely team in every knockout slot, from which we build one coherent
"projected bracket" for the score recommendations.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linear_sum_assignment

from . import config
from .data import Team, Match, Result
from .bayes import BayesModel


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sample_score(lh: float, la: float, rng: np.random.Generator) -> tuple[int, int]:
    return int(rng.poisson(lh)), int(rng.poisson(la))


def _rank_key(rec: dict) -> tuple:
    return (rec["pts"], rec["gd"], rec["gf"])


def _standings(teams: list[str], played: list[tuple], rng: np.random.Generator) -> list[str]:
    """Return the four teams ordered 1st..4th.

    `played` is a list of (home, away, gh, ga) for all six group games.
    """
    rec = {t: {"pts": 0, "gd": 0, "gf": 0} for t in teams}
    for h, a, gh, ga in played:
        rec[h]["gf"] += gh; rec[a]["gf"] += ga
        rec[h]["gd"] += gh - ga; rec[a]["gd"] += ga - gh
        if gh > ga:
            rec[h]["pts"] += 3
        elif gh < ga:
            rec[a]["pts"] += 3
        else:
            rec[h]["pts"] += 1; rec[a]["pts"] += 1

    # primary sort, descending
    order = sorted(teams, key=lambda t: _rank_key(rec[t]), reverse=True)

    # break exact ties on (pts,gd,gf) by head-to-head, then random
    out: list[str] = []
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and _rank_key(rec[order[j + 1]]) == _rank_key(rec[order[i]]):
            j += 1
        tied = order[i:j + 1]
        if len(tied) > 1:
            h2h = {t: [0, 0, 0] for t in tied}   # pts, gd, gf among tied teams
            sset = set(tied)
            for h, a, gh, ga in played:
                if h in sset and a in sset:
                    h2h[h][1] += gh - ga; h2h[a][1] += ga - gh
                    h2h[h][2] += gh; h2h[a][2] += ga
                    if gh > ga:
                        h2h[h][0] += 3
                    elif gh < ga:
                        h2h[a][0] += 3
                    else:
                        h2h[h][0] += 1; h2h[a][0] += 1
            tied = sorted(tied, key=lambda t: (h2h[t][0], h2h[t][1], h2h[t][2],
                                               rng.random()), reverse=True)
        out.extend(tied)
        i = j + 1
    return out


@dataclass
class SimAgg:
    n: int = 0
    group_pos: dict = field(default_factory=lambda: defaultdict(lambda: np.zeros(4)))
    advanced: Counter = field(default_factory=Counter)
    reached: dict = field(default_factory=lambda: defaultdict(Counter))
    champion: Counter = field(default_factory=Counter)
    runner_up: Counter = field(default_factory=Counter)
    slot_home: dict = field(default_factory=lambda: defaultdict(Counter))
    slot_away: dict = field(default_factory=lambda: defaultdict(Counter))
    ko_pair: dict = field(default_factory=lambda: defaultdict(Counter))  # mn -> (home,away) counts


# --------------------------------------------------------------------------- #
# One simulated tournament
# --------------------------------------------------------------------------- #
def simulate_once(bm: BayesModel, groups: dict[str, list[str]],
                  group_fixtures: dict[str, list[Match]],
                  final_by_pair: dict[tuple, tuple],
                  rng: np.random.Generator, agg: SimAgg) -> None:
    atk, dfn = bm.sample_strengths(rng)

    pos1, pos2, thirds = {}, {}, {}
    third_record = {}
    for g, gteams in groups.items():
        played = []
        for m in group_fixtures[g]:
            if (m.home, m.away) in final_by_pair:
                gh, ga = final_by_pair[(m.home, m.away)]
            else:
                lh, la = bm.lambdas(m.home, m.away, m.host_home, m.host_away, atk, dfn)
                gh, ga = _sample_score(lh, la, rng)
            played.append((m.home, m.away, gh, ga))
        ordered = _standings(gteams, played, rng)
        for k, t in enumerate(ordered):
            agg.group_pos[t][k] += 1
        pos1[g], pos2[g], thirds[g] = ordered[0], ordered[1], ordered[2]
        # store third's record for cross-group ranking
        rec = {"pts": 0, "gd": 0, "gf": 0}
        t3 = ordered[2]
        for h, a, gh, ga in played:
            if h == t3:
                rec["gf"] += gh; rec["gd"] += gh - ga
                rec["pts"] += 3 if gh > ga else (1 if gh == ga else 0)
            elif a == t3:
                rec["gf"] += ga; rec["gd"] += ga - gh
                rec["pts"] += 3 if ga > gh else (1 if gh == ga else 0)
        third_record[g] = rec

    # best 8 of 12 thirds
    third_groups = sorted(groups.keys(),
                          key=lambda g: (third_record[g]["pts"], third_record[g]["gd"],
                                         third_record[g]["gf"], rng.random()),
                          reverse=True)
    qualifying_third_groups = third_groups[:8]

    # assign thirds to slots respecting allowed-group constraints
    slot_mns = list(config.THIRD_PLACE_SLOTS.keys())
    cost = np.zeros((8, 8))
    for si, mn in enumerate(slot_mns):
        allowed = set(config.THIRD_PLACE_SLOTS[mn])
        for ti, g in enumerate(qualifying_third_groups):
            cost[si, ti] = 0.0 if g in allowed else 1e6
    rows, cols = linear_sum_assignment(cost)
    third_slot_team = {}
    for si, ti in zip(rows, cols):
        third_slot_team[slot_mns[si]] = thirds[qualifying_third_groups[ti]]

    # build R32 matchups
    def resolve(slot):
        if isinstance(slot, tuple):       # ("3rd", [groups]) -> assigned team
            return None                   # filled below per-match
        pos, g = slot[0], slot[1]
        return pos1[g] if pos == "1" else pos2[g]

    matchups = {}
    for mn, (sh, sa) in config.R32_MATCHES.items():
        home = resolve(sh)
        away = third_slot_team[mn] if isinstance(sa, tuple) else resolve(sa)
        matchups[mn] = (home, away)
        agg.slot_home[mn][home] += 1
        agg.slot_away[mn][away] += 1
        agg.ko_pair[mn][(home, away)] += 1

    # play knockouts
    winners, losers = {}, {}
    reached_round = {}

    def play(home, away):
        lh, la = bm.lambdas(home, away, False, False, atk, dfn)
        gh, ga = _sample_score(lh, la, rng)
        if gh > ga:
            return home, away
        if ga > gh:
            return away, home
        # shootout: stronger lambda advances more often
        return (home, away) if rng.random() < lh / (lh + la) else (away, home)

    for mn in range(73, 89):
        h, a = matchups[mn]
        w, l = play(h, a)
        winners[mn], losers[mn] = w, l
        reached_round.setdefault(l, "R32")

    for mn in list(range(89, 101)) + [101, 102]:
        f1, f2 = config.KO_TREE[mn]
        h, a = winners[f1], winners[f2]
        agg.ko_pair[mn][(h, a)] += 1
        w, l = play(h, a)
        winners[mn], losers[mn] = w, l
        rnd = config.round_of(mn)
        reached_round[l] = rnd

    # final (match 104): winners of the two semi-finals
    fh, fa = winners[101], winners[102]
    agg.ko_pair[104][(fh, fa)] += 1
    agg.ko_pair[103][(losers[101], losers[102])] += 1   # third-place match
    champ, runner = play(fh, fa)
    winners[104] = champ
    reached_round[runner] = "F"

    # tally rounds
    agg.champion[champ] += 1
    agg.runner_up[runner] += 1
    agg.reached[champ]["W"] += 1
    for t, r in reached_round.items():
        agg.reached[t][r] += 1
    # advancement out of groups
    for g in groups:
        agg.advanced[pos1[g]] += 1
        agg.advanced[pos2[g]] += 1
    for g in qualifying_third_groups:
        agg.advanced[thirds[g]] += 1


def run(bm: BayesModel, teams: dict[str, Team], fixtures: list[Match],
        results: list[Result], n: int = config.N_SIMULATIONS,
        seed: int = config.RANDOM_SEED) -> SimAgg:
    groups: dict[str, list[str]] = defaultdict(list)
    for name, t in teams.items():
        groups[t.group].append(name)
    groups = {g: groups[g] for g in sorted(groups)}

    group_fixtures: dict[str, list[Match]] = defaultdict(list)
    for m in fixtures:
        if m.rnd == "Group":
            group_fixtures[m.group].append(m)

    final_by_pair = {(r.home, r.away): (r.home_goals, r.away_goals)
                     for r in results if r.status == "final"}

    rng = np.random.default_rng(seed)
    agg = SimAgg(n=n)
    for _ in range(n):
        simulate_once(bm, groups, group_fixtures, final_by_pair, rng, agg)
    return agg
