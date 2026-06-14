"""Calibrate (alpha, b, rho) to EL PAÍS's published anchors.

We choose the three match-model parameters so the engine reproduces:
  * an even game  -> ~2.55 total goals and ~27% draws,
  * Spain vs Germany on neutral ground -> 52% / 27% / 21% (W/D/L).

These are the concrete numbers El País published for their GAM-Poisson model,
so anchoring to them keeps us consistent with a forecaster that has beaten the
big banks and the betting markets at the last two World Cups.
"""
from __future__ import annotations
import json
import numpy as np
from scipy.optimize import least_squares

from . import config
from .match_model import Params, predict_from_elo


def _residuals(theta: np.ndarray) -> list[float]:
    alpha, b, rho = theta
    p = Params(alpha=alpha, b=b, rho=rho)

    even = predict_from_elo(config.ELO_REF, config.ELO_REF, False, False, p)
    even_total = even.exp_home + even.exp_away

    d = config.CAL_ANCHOR_ELO_DIFF
    anchor = predict_from_elo(config.ELO_REF + d / 2, config.ELO_REF - d / 2,
                              False, False, p)

    return [
        even_total - config.CAL_EVEN_TOTAL_GOALS,
        even.p_draw - config.CAL_EVEN_DRAW_PROB,
        anchor.p_home - config.CAL_ANCHOR_WIN,
        anchor.p_away - config.CAL_ANCHOR_LOSS,
    ]


def calibrate(save: bool = True) -> Params:
    res = least_squares(
        _residuals,
        x0=np.array([0.24, 0.13, -0.08]),
        bounds=([0.0, 0.05, -0.20], [0.6, 0.30, 0.15]),
    )
    alpha, b, rho = res.x
    params = Params(alpha=float(alpha), b=float(b), rho=float(rho))
    if save:
        config.CALIBRATION_JSON.write_text(
            json.dumps({"alpha": params.alpha, "b": params.b, "rho": params.rho},
                       indent=2))
    return params


def load_params() -> Params:
    if config.CALIBRATION_JSON.exists():
        d = json.loads(config.CALIBRATION_JSON.read_text())
        return Params(alpha=d["alpha"], b=d["b"], rho=d["rho"])
    return calibrate()


def report(params: Params) -> str:
    """Human-readable check of the fit against the anchors."""
    lines = ["Calibration:",
             f"  alpha={params.alpha:.4f}  b={params.b:.4f}  rho={params.rho:.4f}"]
    even = predict_from_elo(config.ELO_REF, config.ELO_REF, False, False, params)
    lines.append(f"  even game     -> total {even.exp_home + even.exp_away:.2f} goals, "
                 f"W/D/L {even.p_home:.0%}/{even.p_draw:.0%}/{even.p_away:.0%}")
    d = config.CAL_ANCHOR_ELO_DIFF
    anc = predict_from_elo(config.ELO_REF + d / 2, config.ELO_REF - d / 2,
                           False, False, params)
    lines.append(f"  Spain v Germ. -> W/D/L {anc.p_home:.0%}/{anc.p_draw:.0%}/{anc.p_away:.0%} "
                 f"(target 52/27/21)  modal {anc.modal[0]}-{anc.modal[1]}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(calibrate()))
