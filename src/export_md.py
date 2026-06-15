"""Write the compact Markdown sheet to feed to Claude-in-Chrome for filling an
online prediction pool.  Every line is one match -> one recommended exact score.
"""
from __future__ import annotations
from itertools import groupby

from . import config
from .recommend import Rec
from .metrics import Validation


def _line(r: Rec) -> str:
    score = f"{r.rec_home}-{r.rec_away}"
    head = f"- **{r.date}** — {r.home} vs {r.away} → "
    if r.status == "final":
        return head + f"**{score}**  `[PLAYED — final result]`"
    if r.status == "in_progress":
        live = f"{r.actual_home}-{r.actual_away}"
        return head + f"**{score}**  `[LIVE {live} — not final]`"
    # scheduled or projected: the points-maximising pick (winner implied by the score)
    side = r.home if r.rec_home > r.rec_away else (r.away if r.rec_home < r.rec_away else "draw")
    out = head + f"**{score}** ({side})"
    if r.status == "projected":
        out += f"  `[projected matchup {r.matchup_prob:.0%}]`"
    return out


def build(path, group_recs: list[Rec], ko_recs: list[Rec],
          val: Validation, generated: str, n_sims: int) -> None:
    L: list[str] = []
    L.append("# World Cup 2026 — Recommended Scores")
    L.append("")
    L.append(f"*Generated {generated} · Dixon-Coles Poisson + Bayesian update · "
             f"{n_sims:,} sims · each score maximises expected Bodytech points "
             f"(140 exact / 100 winner+diff / 70 winner or draw).*")
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
    L.append("- Each score is the one that **maximises expected Bodytech points** over the "
             "model's full probability distribution, so favourites get a decisive scoreline "
             "instead of 1-1. The team in parentheses is the predicted winner.")
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
