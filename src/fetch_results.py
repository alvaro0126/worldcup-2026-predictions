"""Self-updating results fetcher.

Pulls completed match scores from the Wikipedia 2026 World Cup pages and merges
them into data/results.csv (status=final), so the pipeline can refresh itself as
games are played.  Idempotent and conservative:
  * only writes matches that have a numeric scoreline,
  * never deletes or downgrades rows you entered by hand,
  * if the network or parse fails, it leaves results.csv untouched and the rest
    of the pipeline still runs on whatever is already there.

The hand-editable CSV remains the source of truth — this is a convenience layer.
"""
from __future__ import annotations
import csv
import re
import urllib.request
from dataclasses import dataclass

from . import config

GROUPS = "ABCDEFGHIJKL"
KNOCKOUT_PAGE = "2026_FIFA_World_Cup_knockout_stage"
GROUP_PAGE = "2026_FIFA_World_Cup_Group_{}"
UA = {"User-Agent": "Mozilla/5.0 (wc2026-predictions model updater)"}

_MATCH_RE = re.compile(
    r'<th class="fhome"[^>]*>.*?<a [^>]*>([^<]+)</a>.*?</th>\s*'
    r'<th class="fscore">\s*([0-9]+)[–-]([0-9]+)\s*</th>\s*'
    r'<th class="faway"[^>]*>.*?<a [^>]*>([^<]+)</a>',
    re.DOTALL)
_DATE_RE = re.compile(
    r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|'
    r'September|October|November|December)\s+2026')
_MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"], start=1)}


@dataclass(frozen=True)
class Fetched:
    date: str
    home: str
    away: str
    home_goals: int
    away_goals: int


def _iso_date(day: str, month_name: str) -> str:
    return f"2026-{_MONTHS[month_name]:02d}-{int(day):02d}"


def _get(page: str) -> str:
    url = f"https://en.wikipedia.org/wiki/{page}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return r.read().decode("utf-8", "ignore")


def _parse(html: str) -> list[Fetched]:
    """One Fetched per completed match in a page (chunked by football box)."""
    out: list[Fetched] = []
    for chunk in html.split('class="footballbox"')[1:]:
        m = _MATCH_RE.search(chunk)
        if not m:
            continue
        home = config.canon(m.group(1).strip())
        away = config.canon(m.group(4).strip())
        gh, ga = int(m.group(2)), int(m.group(3))
        d = _DATE_RE.search(chunk)
        date = _iso_date(d.group(1), d.group(2)) if d else ""
        out.append(Fetched(date, home, away, gh, ga))
    return out


def fetch_all() -> list[Fetched]:
    """Best-effort scrape of every group page plus the knockout page."""
    results: list[Fetched] = []
    pages = [GROUP_PAGE.format(g) for g in GROUPS] + [KNOCKOUT_PAGE]
    for page in pages:
        try:
            results.extend(_parse(_get(page)))
        except Exception as exc:               # noqa: BLE001 - stay resilient
            print(f"  warn: could not fetch/parse {page}: {exc}")
    return results


def _read_csv() -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    if not config.RESULTS_CSV.exists():
        return rows
    with config.RESULTS_CSV.open() as f:
        for r in csv.DictReader(f):
            r["home"] = config.canon(r["home"])
            r["away"] = config.canon(r["away"])
            rows[(r["home"], r["away"])] = r
    return rows


def _fixture_dates() -> dict[tuple[str, str], str]:
    dates: dict[tuple[str, str], str] = {}
    if config.FIXTURES_CSV.exists():
        with config.FIXTURES_CSV.open() as f:
            for r in csv.DictReader(f):
                dates[(config.canon(r["home"]), config.canon(r["away"]))] = r["date"]
    return dates


def update_results_csv(dry_run: bool = False) -> dict:
    """Merge fetched final scores into results.csv. Returns a change summary."""
    fetched = fetch_all()
    if not fetched:
        print("  no results fetched (offline?); leaving results.csv untouched")
        return {"added": 0, "updated": 0, "unchanged": 0, "fetched": 0}

    existing = _read_csv()
    fx_dates = _fixture_dates()
    added = updated = unchanged = 0
    changes: list[str] = []

    for fx in fetched:
        key = (fx.home, fx.away)
        date = fx.date or fx_dates.get(key) or (existing.get(key, {}) or {}).get("date", "")
        new = {"date": date, "home": fx.home, "away": fx.away,
               "home_goals": str(fx.home_goals), "away_goals": str(fx.away_goals),
               "status": "final"}
        old = existing.get(key)
        if old is None:
            added += 1
            changes.append(f"+ {fx.home} {fx.home_goals}-{fx.away_goals} {fx.away}")
        elif (old.get("home_goals"), old.get("away_goals"), old.get("status")) != \
             (new["home_goals"], new["away_goals"], new["status"]):
            updated += 1
            changes.append(f"~ {fx.home} {fx.home_goals}-{fx.away_goals} {fx.away} "
                           f"(was {old.get('home_goals')}-{old.get('away_goals')}/{old.get('status')})")
        else:
            unchanged += 1
        existing[key] = new

    if not dry_run:
        ordered = sorted(existing.values(), key=lambda r: (r["date"], r["home"]))
        with config.RESULTS_CSV.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["date", "home", "away",
                                              "home_goals", "away_goals", "status"])
            w.writeheader()
            w.writerows(ordered)

    for c in changes:
        print("  " + c)
    summary = {"added": added, "updated": updated, "unchanged": unchanged,
               "fetched": len(fetched)}
    print(f"  fetched {summary['fetched']} matches: "
          f"{added} added, {updated} updated, {unchanged} unchanged"
          + (" (dry-run, not written)" if dry_run else ""))
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Fetch latest WC2026 results into results.csv")
    ap.add_argument("--dry-run", action="store_true", help="report without writing")
    update_results_csv(dry_run=ap.parse_args().dry_run)
