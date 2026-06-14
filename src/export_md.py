"""Write the compact Markdown sheet to feed to Claude-in-Chrome for filling an
online prediction pool.  Every line is one match -> one recommended exact score.
"""
from __future__ import annotations
from itertools import groupby

from . import config
from .recommend import Rec
from .metrics import Validation


def _line(r: Rec) -> str:
    modal = f"{r.rec_home}-{r.rec_away}"
    backup = f"{r.alt_home}-{r.alt_away}"
    pick_team = {"H": r.home, "D": "draw", "A": r.away}[r.best_outcome]
    head = f"- **{r.date}** — {r.home} vs {r.away} → "

    if r.status == "final":
        return head + f"**{modal}**  `[PLAYED — final result]`"
    if r.status == "in_progress":
        live = f"{r.actual_home}-{r.actual_away}"
        return head + f"**{modal}**  `[LIVE {live} — not final; model pick shown]`"

    # scheduled group game or projected knockout
    modal_is_draw = r.rec_home == r.rec_away
    if r.best_outcome == "D" or not modal_is_draw:
        suffix = "" if r.best_outcome == "D" else f" ({pick_team})"
        score = f"**{modal}**{suffix}"
    else:
        # closest single score is a draw but a side is favoured -> offer a decisive backup
        score = f"**{modal}** _(if decisive: {backup} {pick_team})_"
    if r.status == "projected":
        score += f"  `[projected matchup {r.matchup_prob:.0%}]`"
    return head + score


def build(path, group_recs: list[Rec], ko_recs: list[Rec],
          val: Validation, generated: str, n_sims: int) -> None:
    L: list[str] = []
    L.append("# World Cup 2026 — Recommended Scores")
    L.append("")
    L.append(f"*Generated {generated} · Dixon-Coles Poisson model + Bayesian daily "
             f"update · {n_sims:,} simulations · pre-tournament hold-out RPS "
             f"{val.mean_rps:.3f}, outcome hit-rate {val.outcome_hit_rate:.0%}.*")
    L.append("")
    L.append("## Instructions for Claude (in Chrome)")
    L.append("")
    L.append("Fill the prediction pool by entering the **bold score** after each "
             "arrow for the matching fixture. Notes:")
    L.append("- Match teams by name (the pool may localise names, e.g. *Türkiye*=Turkey, "
             "*Korea Republic*=South Korea, *USA*=United States, *Czechia*=Czech Republic).")
    L.append("- `[PLAYED — final]` matches are already decided — the score shown **is the "
             "actual result**; enter it only if the pool still wants it.")
    L.append("- `[LIVE — not final]` is still in progress; wait for full-time before trusting it.")
    L.append("- `[PROJECTED matchup ..%]` knockout games use the **most-likely teams** for that "
             "slot; **confirm the real matchup on the site** before entering, since the bracket "
             "isn't set yet. The % is how often that exact pairing occurred in simulation.")
    L.append("- The exact score is the single most-likely scoreline; '(pick: …)' is the "
             "most-likely match result if the pool also scores that.")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## Group stage")
    for g, rows in groupby(sorted(group_recs, key=lambda r: (r.group, r.date, r.match_id)),
                           key=lambda r: r.group):
        L.append("")
        L.append(f"### Group {g}")
        for r in rows:
            L.append(_line(r))
    L.append("")
    L.append("## Knockout stage — projected matchups (verify before entering)")
    for rnd in ["R32", "R16", "QF", "SF", "3P", "F"]:
        rows = [r for r in ko_recs if r.rnd == rnd]
        if not rows:
            continue
        L.append("")
        L.append(f"### {config.ROUND_NAMES[rnd]}")
        for r in rows:
            L.append(_line(r))
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")
