# World Cup 2026 — Score Recommendations

A sound, reproducible statistical model that recommends **one exact score for every
one of the 104 matches** of the 2026 FIFA World Cup, and **updates itself daily** as
results come in.

Built to mirror the methodology of two reference pieces (both supplied as the brief):
EL PAÍS's tournament model ("¿Quién ganará el Mundial?") and The Economist's
"How to win the World Cup" — both of which use **World Football Elo** as the core
strength metric and a **Dixon-Coles Poisson** match engine.

---

## What it produces

`outputs/world_cup_2026_recommendations.xlsx` — 8 sheets:

| Sheet | Contents |
|---|---|
| Read me | Method summary, validation, caveats, how to update |
| Recommendations (104) | **Headline:** one recommended exact score per match |
| Group fixtures | 72 group games (real fixtures; played games show actuals) |
| Projected knockouts | 32 knockout games, most-likely matchup per bracket slot |
| Group finish odds | P(1st/2nd/3rd/4th) and P(advance) per team |
| Title odds | P(reach QF / SF / final / win) per team |
| Team ratings | Elo prior → Bayesian-updated Elo, sources, title odds |
| Validation | Pre-tournament prior vs games already played |

`outputs/pool_picks.md` — a compact match → score sheet to feed **Claude-in-Chrome**
for entering picks into an online prediction pool.

---

## The model (why it is sound)

**1. Team strength — multi-source, on one scale.**
World Football **Elo** (eloratings.net) is the canonical metric both source articles
rely on. The 18 strongest teams use their eloratings.net value directly; every other
team is the average of whatever independent signals it has, each first mapped onto the
eloratings scale by regression: **Metis-model Elo + FIFA world ranking + El País
"nivel"**. Blending beats any single source (e.g. Metis alone badly underrates the USA;
the FIFA ranking corrects it). Each team's `Elo source` is recorded for transparency.

**2. Match engine — Dixon-Coles bivariate Poisson.**
Goals follow a Poisson law; expected goals come from the Elo difference (plus host
advantage for the three host nations in their own country). The **Dixon-Coles (1997)**
correction fixes the low-score / draw probabilities. The three free parameters are
**calibrated to El País's published anchors** so the engine reproduces a forecaster
that has beaten Goldman Sachs, UBS and the betting markets:

| Check | Target (El País) | This model |
|---|---|---|
| Even game | ~2.55 goals, ~27% draws | 2.55 goals, 36/28/36 |
| Spain v Germany (neutral) | 52 / 27 / 21 | 53 / 25 / 22 |
| Argentina v Jordan | modal 2-0 / 3-0 | modal 2-0 |

**3. Bayesian daily update.**
Each team has a latent attack and defence with a **Normal prior centred on its Elo**
(so with zero games the model equals the calibrated Elo model exactly). Every **final**
result pulls a team's attack/defence via the match likelihood, **shrunk toward the Elo
prior** — so a 7-1 win over weak Curaçao moves Germany only +73 Elo (the opponent's
weakness and the prior absorb the rest), not a naïve over-reaction. Inference is MAP
(penalised maximum likelihood); posterior spread is propagated into the simulations.
**Live / in-progress games are held out until full-time.**

**4. Tournament simulation.**
50,000 Monte-Carlo tournaments: group games (fixed results + sampled rest) → FIFA
tie-breakers → best-8 third-placed teams allocated to the official bracket slots →
knockouts (draws resolved by a strength-weighted shootout). Yields advancement / title
odds and the most-likely team in every knockout slot.

**5. Recommended score.**
The single best exact-score prediction is the **mode of the Dixon-Coles score matrix**
(it shifts up automatically for big favourites). A **backup score** — the most-likely
score *given* the most-likely result — is provided for pools that score the result
separately.

---

## Usage

```bash
pip install -r requirements.txt
python run.py                 # fetch latest results + full 50k-sim run
python run.py --no-fetch      # skip the network fetch, run on current results.csv
python run.py --sims 10000    # faster
python run.py --recalibrate   # re-fit the El País anchors first
```

Every run also writes a copy of the picks to your **Desktop**
(`~/Desktop/world_cup_2026_picks.md`).

### Keeping it updated as matches are played (the whole point)

`run.py` **pulls the latest completed scores automatically** before each run
(`src/fetch_results.py` scrapes the Wikipedia 2026 World Cup pages and merges
final scores into `data/results.csv`). So you usually just run:

```bash
python run.py
```

**Run it continuously** with the loop script (refreshes on an interval):

```bash
./update.sh            # every 30 min
./update.sh 600        # every 10 min
```

or a cron entry (every 30 min):

```cron
*/30 * * * * cd /Users/andres/projects/worldcup-2026-predictions && /Users/andres/anaconda3/bin/python run.py >> outputs/update.log 2>&1
```

The fetcher is conservative: it only writes matches with a numeric scoreline,
never downgrades or deletes rows you entered by hand, and if the network or parse
fails it leaves `data/results.csv` untouched and the pipeline still runs. You (or
Claude-in-Chrome) can always hand-edit `data/results.csv` — it stays the source of
truth. Mark a still-in-play game `status=in_progress` to hold it out until full-time.

```bash
python -m src.fetch_results --dry-run   # preview what the fetcher would change
```

---

## Data

| File | What | Source |
|---|---|---|
| `data/ratings.csv` | 48 teams: group, Elo (eloratings + Metis), FIFA rank, nivel | eloratings.net, metisfootball.com, FIFA seeding, El País |
| `data/fixtures.csv` | 72 group fixtures with dates, venues, host flags | Wikipedia per-group pages |
| `data/results.csv` | Results so far (status final / in_progress) | Wikipedia / FIFA standings |
| `data/calibration.json` | Fitted (alpha, b, rho) | produced by `calibrate.py` |

## Caveats

- A World Cup is **designed** to be unpredictable — even the favourite wins only ~1 in
  6. These are probabilistic best guesses, not locks.
- Knockout matchups are **projections** (the bracket isn't drawn yet). Each is the
  single most-likely pairing for that slot; the matchup % shows how uncertain it is, and
  the same marquee pair can appear in two adjacent slots when a team is equally likely to
  win or finish second in its group. Re-run after the group stage to lock real matchups.
- Knockouts are treated as **neutral venue** (host advantage is modelled only in the
  group stage, where venues are known).
- **Squad value** (Transfermarkt), which El País weights directly, is proxied here via
  Elo. That lifts star-squad sides (France, Brazil, Portugal) a little in their model
  relative to this Elo-driven one — the main reason our title odds differ slightly.

## Layout

```
data/        ratings.csv, fixtures.csv, results.csv, calibration.json
src/         config, data, match_model, calibrate, bayes, simulate,
             recommend, metrics, export_excel, export_md
run.py       daily pipeline entry point
outputs/     the .xlsx and .md deliverables
```
