#!/usr/bin/env python3
"""World Cup 2026 score-recommendation pipeline.

Daily use:
    1. update data/results.csv  (set status=final as games finish)
    2. python run.py            (re-fits the Bayesian model and regenerates outputs)

Outputs land in outputs/:
    * world_cup_2026_recommendations.xlsx
    * pool_picks.md
"""
from __future__ import annotations
import argparse
import shutil
from datetime import date

from src import (config, data, calibrate, bayes, simulate, metrics,
                 recommend, export_excel, export_md, export_html, fetch_results)


def main() -> None:
    ap = argparse.ArgumentParser(description="World Cup 2026 score recommendations")
    ap.add_argument("--sims", type=int, default=config.N_SIMULATIONS,
                    help="number of Monte-Carlo tournament simulations")
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--recalibrate", action="store_true",
                    help="re-fit (alpha,b,rho) to the El Pais anchors before running")
    ap.add_argument("--no-fetch", action="store_true",
                    help="skip pulling the latest results from Wikipedia")
    args = ap.parse_args()

    if not args.no_fetch:
        print("Fetching latest results...")
        fetch_results.update_results_csv()
        print()

    teams = data.load_teams()
    params = calibrate.calibrate() if args.recalibrate else calibrate.load_params()
    fixtures = data.load_fixtures()
    results = data.load_results(fixtures)

    bm = bayes.fit(teams, results, params)
    val = metrics.evaluate_prior(teams, results, params)
    agg = simulate.run(bm, teams, fixtures, results, n=args.sims, seed=args.seed)

    config.OUTPUTS.mkdir(exist_ok=True)
    config.DOCS.mkdir(exist_ok=True)
    xlsx = config.OUTPUTS / "world_cup_2026_recommendations.xlsx"
    md = config.OUTPUTS / "pool_picks.md"
    html_page = config.DOCS / "index.html"
    generated = date.today().strftime("%-d %B %Y")
    export_excel.build(xlsx, bm, teams, fixtures, results, agg, params, val)
    export_md.build(md, recommend.group_recs(bm, fixtures, results),
                    recommend.knockout_recs(bm, agg), val,
                    date.today().isoformat(), agg.n)
    export_html.build(html_page, bm, teams, fixtures, results, agg, params, val, generated)

    desktop_note = ""
    if config.DESKTOP.exists():
        shutil.copyfile(md, config.DESKTOP_MD)
        desktop_note = f"\n  {config.DESKTOP_MD}  (Desktop copy)"

    n_final = len([r for r in results if r.status == "final"])
    n_live = len([r for r in results if r.status == "in_progress"])
    print(calibrate.report(params))
    print()
    print(f"Bayesian fit on {n_final} final results ({n_live} live game held out).")
    print(f"Validation (pre-tournament prior): n={val.n}  "
          f"outcome hit {val.outcome_hit_rate:.0%}  exact {val.exact_hit_rate:.0%}  "
          f"RPS {val.mean_rps:.3f}")
    print()
    print(f"Title odds (top 8 of {agg.n:,} sims):")
    for t, c in agg.champion.most_common(8):
        print(f"  {t:16s} {c/agg.n:5.1%}")
    print()
    print(f"Wrote:\n  {xlsx}\n  {md}\n  {html_page}{desktop_note}")


if __name__ == "__main__":
    main()
