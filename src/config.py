"""Static configuration: paths, model hyperparameters, calibration targets,
team-name aliases, and the official 2026 knockout-bracket structure.

All tunable numbers live here so nothing is hard-coded deep in the logic.
"""
from __future__ import annotations
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
RATINGS_CSV = DATA / "ratings.csv"
FIXTURES_CSV = DATA / "fixtures.csv"
RESULTS_CSV = DATA / "results.csv"
CALIBRATION_JSON = DATA / "calibration.json"

# A copy of the Markdown picks is dropped here on every run (machine-specific).
DESKTOP = Path.home() / "Desktop"
DESKTOP_MD = DESKTOP / "world_cup_2026_picks.md"

# --------------------------------------------------------------------------- #
# Model hyper-parameters
# --------------------------------------------------------------------------- #
MAX_GOALS = 10            # truncate the score matrix at 10-10 (prob mass beyond is ~0)
HOST_ELO_BONUS = 80.0     # Elo points a host nation gains when playing in its own country
PRIOR_SIGMA = 0.20        # SD of the Normal prior on each team's log attack / defence
                          # (how far in-tournament form may pull a team off its Elo prior)
PER_GAME_INFO = 1.6       # approx Fisher information per played match (for posterior SD)
N_SIMULATIONS = 50_000    # Monte-Carlo tournament simulations
RANDOM_SEED = 20260614    # reproducibility (tournament starts 2026-06-14 snapshot)
ELO_REF = 1850.0          # reference Elo (cancels out; keeps attack/defence centred)

# --------------------------------------------------------------------------- #
# Calibration targets — anchored to EL PAÍS's published model
#   * Spain vs Germany on neutral ground -> 52% / 27% / 21% (W/D/L)
#   * an even game -> draw probability ~27%, total goals ~2.55
#   These come straight from the El País methodology section.
# --------------------------------------------------------------------------- #
CAL_EVEN_TOTAL_GOALS = 2.55
CAL_EVEN_DRAW_PROB = 0.27
CAL_ANCHOR_ELO_DIFF = 2157 - 1932   # Spain(2157) - Germany(1932) = 225
CAL_ANCHOR_WIN = 0.52
CAL_ANCHOR_DRAW = 0.27
CAL_ANCHOR_LOSS = 0.21

# --------------------------------------------------------------------------- #
# Team-name aliases -> canonical names used across every file
# --------------------------------------------------------------------------- #
ALIASES = {
    "Czech Republic": "Czechia",
    "Korea Republic": "South Korea",
    "South Korea": "South Korea",
    "Turkiye": "Turkey",
    "Türkiye": "Turkey",
    "USA": "United States",
    "US": "United States",
    "Cote d'Ivoire": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "Congo DR": "DR Congo",
    "DR Congo": "DR Congo",
    "Cape Verde Islands": "Cape Verde",
    "Cabo Verde": "Cape Verde",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Curacao": "Curaçao",
}

def canon(name: str) -> str:
    """Return the canonical team name."""
    n = name.strip()
    return ALIASES.get(n, n)

# --------------------------------------------------------------------------- #
# Knockout-bracket structure (official, from Wikipedia 2026 knockout stage)
#   Slot codes: "1X" = winner of group X, "2X" = runner-up of group X,
#   ("3rd", [groups]) = a best-third-placed team coming from one of those groups.
# --------------------------------------------------------------------------- #
R32_MATCHES = {
    73: ("2A", "2B"),
    74: ("1E", ("3rd", ["A", "B", "C", "D", "F"])),
    75: ("1F", "2C"),
    76: ("1C", "2F"),
    77: ("1I", ("3rd", ["C", "D", "F", "G", "H"])),
    78: ("2E", "2I"),
    79: ("1A", ("3rd", ["C", "E", "F", "H", "I"])),
    80: ("1L", ("3rd", ["E", "H", "I", "J", "K"])),
    81: ("1D", ("3rd", ["B", "E", "F", "I", "J"])),
    82: ("1G", ("3rd", ["A", "E", "H", "I", "J"])),
    83: ("2K", "2L"),
    84: ("1H", "2J"),
    85: ("1B", ("3rd", ["E", "F", "G", "I", "J"])),
    86: ("1J", "2H"),
    87: ("1K", ("3rd", ["D", "E", "I", "J", "L"])),
    88: ("2D", "2G"),
}

# Winner-of-match -> winner-of-match tree for R16..Final
KO_TREE = {
    89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
    93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87),
    97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96),
    101: (97, 98), 102: (99, 100),
    104: (101, 102),   # final: winners of the semifinals
}
THIRD_PLACE_MATCH = (103, (101, 102))   # losers of the semifinals

ROUND_NAMES = {
    "R32": "Round of 32", "R16": "Round of 16", "QF": "Quarter-final",
    "SF": "Semi-final", "3P": "Third-place", "F": "Final",
}

# Approximate knockout-round date windows (official 2026 schedule)
ROUND_DATES = {
    "R32": "2026-06-28/07-03", "R16": "2026-07-04/07-07",
    "QF": "2026-07-09/07-11", "SF": "2026-07-14/07-15",
    "3P": "2026-07-18", "F": "2026-07-19",
}

def round_of(match_no: int) -> str:
    if 73 <= match_no <= 88:
        return "R32"
    if 89 <= match_no <= 96:
        return "R16"
    if 97 <= match_no <= 100:
        return "QF"
    if 101 <= match_no <= 102:
        return "SF"
    if match_no == 103:
        return "3P"
    if match_no == 104:
        return "F"
    return "Group"

# The eight R32 matches whose second team is a best-third-placed side
THIRD_PLACE_SLOTS = {mn: home_away[1][1]
                     for mn, home_away in R32_MATCHES.items()
                     if isinstance(home_away[1], tuple)}
