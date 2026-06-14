"""Bayesian attack/defence layer that updates the Elo prior with match results.

Model (Dixon-Coles bivariate Poisson with team effects):

    log lambda_home = mu + atk[home] - def[away] (+ host term)
    log lambda_away = mu + atk[away] - def[home] (+ host term)

Priors: atk[i] and def[i] are each Normal( m_i , sigma^2 ) with
    m_i = b * (Elo_i - Elo_ref) / 100
so that with **zero** games played the posterior mean equals the calibrated
Elo model exactly.  Each FINAL result then pulls a team's attack/defence away
from its Elo prior, shrunk toward it by the prior (a 7-1 win over a weak side
moves Germany only modestly; the opponent's weakness absorbs much of it).

Inference is MAP (penalised maximum likelihood) via L-BFGS-B — fast, deterministic,
re-runnable every day.  Posterior spread per team is approximated in closed form
(prior precision + Fisher information per game) and propagated into the simulator.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

from . import config
from .data import Team, Result
from .match_model import Params, Prediction, predict


def _log_tau(gh: int, ga: int, lh: float, la: float, rho: float) -> float:
    if gh == 0 and ga == 0:
        tau = 1.0 - lh * la * rho
    elif gh == 0 and ga == 1:
        tau = 1.0 + lh * rho
    elif gh == 1 and ga == 0:
        tau = 1.0 + la * rho
    elif gh == 1 and ga == 1:
        tau = 1.0 - rho
    else:
        tau = 1.0
    return float(np.log(max(tau, 1e-9)))


def _logpmf_dc(gh: int, ga: int, lh: float, la: float, rho: float) -> float:
    return float(poisson.logpmf(gh, lh) + poisson.logpmf(ga, la)) + \
        _log_tau(gh, ga, lh, la, rho)


@dataclass(frozen=True)
class BayesModel:
    names: list[str]
    idx: dict[str, int]
    atk: np.ndarray           # MAP attack (posterior mean)
    dfn: np.ndarray           # MAP defence (posterior mean)
    post_sd: np.ndarray       # per-team posterior SD (shared by atk & def)
    prior_mean: np.ndarray    # m_i  (Elo prior in attack/defence space)
    ngames: np.ndarray        # final games played, per team
    params: Params
    hg: float                 # host advantage in log-goal units

    # ---- effective, post-update Elo (for reporting) ---------------------- #
    def effective_elo(self, name: str) -> float:
        i = self.idx[name]
        strength = 0.5 * (self.atk[i] + self.dfn[i])
        return config.ELO_REF + 100.0 * strength / self.params.b

    def prior_elo(self, name: str) -> float:
        i = self.idx[name]
        return config.ELO_REF + 100.0 * self.prior_mean[i] / self.params.b

    # ---- lambdas / predictions ------------------------------------------- #
    def _lams(self, home: str, away: str, host_home: bool, host_away: bool,
              atk: np.ndarray, dfn: np.ndarray) -> tuple[float, float]:
        i, j = self.idx[home], self.idx[away]
        logh = self.params.alpha + atk[i] - dfn[j]
        loga = self.params.alpha + atk[j] - dfn[i]
        if host_home:
            logh += self.hg
            loga -= self.hg
        if host_away:
            loga += self.hg
            logh -= self.hg
        return float(np.exp(logh)), float(np.exp(loga))

    def predict(self, home: str, away: str,
                host_home: bool = False, host_away: bool = False) -> Prediction:
        """Posterior-mean prediction for ANY matchup (the reusable predictor)."""
        lh, la = self._lams(home, away, host_home, host_away, self.atk, self.dfn)
        return predict(lh, la, self.params.rho)

    def sample_strengths(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Draw a full set of team attack/defence values from the posterior."""
        atk = rng.normal(self.atk, self.post_sd)
        dfn = rng.normal(self.dfn, self.post_sd)
        return atk, dfn

    def lambdas(self, home: str, away: str, host_home: bool, host_away: bool,
                atk: np.ndarray, dfn: np.ndarray) -> tuple[float, float]:
        return self._lams(home, away, host_home, host_away, atk, dfn)


def fit(teams: dict[str, Team], results: list[Result], params: Params) -> BayesModel:
    names = sorted(teams)
    idx = {n: i for i, n in enumerate(names)}
    n = len(names)
    elo = np.array([teams[nm].elo for nm in names])
    m = params.b * (elo - config.ELO_REF) / 100.0
    sigma = config.PRIOR_SIGMA
    hg = params.b * config.HOST_ELO_BONUS / 100.0

    finals = [r for r in results if r.status == "final"]
    # pre-index the matches for the objective
    obs = [(idx[r.home], idx[r.away], r.home_goals, r.away_goals,
            r.host_home, r.host_away) for r in finals]

    def nll(x: np.ndarray) -> float:
        atk, dfn = x[:n], x[n:]
        val = 0.0
        for i, j, gh, ga, hh, ha in obs:
            logh = params.alpha + atk[i] - dfn[j]
            loga = params.alpha + atk[j] - dfn[i]
            if hh:
                logh += hg
                loga -= hg
            if ha:
                loga += hg
                logh -= hg
            val -= _logpmf_dc(gh, ga, float(np.exp(logh)), float(np.exp(loga)),
                              params.rho)
        val += ((atk - m) ** 2).sum() / (2 * sigma ** 2)
        val += ((dfn - m) ** 2).sum() / (2 * sigma ** 2)
        return val

    x0 = np.concatenate([m, m])
    res = minimize(nll, x0, method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-10})
    atk, dfn = res.x[:n], res.x[n:]

    ngames = np.array([sum(1 for r in finals if nm in (r.home, r.away))
                       for nm in names], dtype=float)
    post_sd = 1.0 / np.sqrt(1.0 / sigma ** 2 + ngames * config.PER_GAME_INFO)

    return BayesModel(names=names, idx=idx, atk=atk, dfn=dfn, post_sd=post_sd,
                      prior_mean=m, ngames=ngames, params=params, hg=hg)
