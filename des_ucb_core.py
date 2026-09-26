"""DES-UCB. Drift-triggered Evidence-Switching UCB, research prototype.

Everything the notebook needs is importable from here.

* estimators      RidgeUCB, PriorEstimator, FeedbackEstimator
* controller      EvidenceSwitch, PageHinkley, ADWIN, BOBWindowSelector
* agents          DESUCBAgent and baselines B1..B9 (shared step()/observe() interface)
* environments    SyntheticEnv (dynamic regret) and RatedItemEnv / ReplayEnv (real data)
* runner/metrics  run(), synthetic_metrics(), real_metrics(), paired_bootstrap_ci()
* claims          ClaimRegistry
* config          load_config()

Metric namespaces are kept apart. ``dynamic`` regret exists only for the synthetic environment,
where theta_t is known. ``offline`` regret is measured against the best rated candidate.
``replay`` reward is the only unbiased real-data quantity and needs a uniform-random logging policy.
"""
from __future__ import annotations

import copy
import json
import math
import os
import warnings
import zipfile
import zlib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

REGRET_KINDS = ("dynamic", "offline", "replay")


# ----------------------------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------------------------
def _deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(*names: str, overrides: Optional[dict] = None) -> dict:
    """Load configs/default.yaml then layer the named yaml files (and a dict) on top."""
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    for n in names:
        p = Path(n)
        if not p.exists():
            p = ROOT / "configs" / (n if n.endswith(".yaml") else n + ".yaml")
        cfg = _deep_update(cfg, yaml.safe_load(p.read_text()) or {})
    if overrides:
        cfg = _deep_update(cfg, overrides)
    return cfg


# ----------------------------------------------------------------------------------------------
# estimators
# ----------------------------------------------------------------------------------------------
class RidgeUCB:
    """Ridge regression with a LinUCB score, maintained with Sherman-Morrison updates."""

    def __init__(self, d: int, lam: float = 1.0, beta: float = 0.5, S: float = 5.0):
        self.d, self.lam, self.beta, self.S = int(d), float(lam), float(beta), float(S)
        self.reset()

    def reset(self) -> None:
        self.V = self.lam * np.eye(self.d)
        self.V_inv = np.eye(self.d) / self.lam
        self.b = np.zeros(self.d)
        self.n = 0.0
        self._theta: Optional[np.ndarray] = None

    def _rank1(self, x: np.ndarray, w: float) -> None:
        # V <- V + w x x^T with Sherman-Morrison on V_inv, w may be negative for removals
        Vx = self.V_inv @ x
        denom = 1.0 + w * float(x @ Vx)
        if abs(denom) < 1e-12:  # pragma: no cover - degenerate removal, fall back to solve
            self.V += w * np.outer(x, x)
            self.V_inv = np.linalg.inv(self.V)
            self._theta = None
            return
        self.V += w * np.outer(x, x)
        self.V_inv -= (w / denom) * np.outer(Vx, Vx)
        self._theta = None

    def add(self, x: np.ndarray, r: float, weight: float = 1.0) -> None:
        x = np.asarray(x, dtype=float)
        self._rank1(x, weight)
        self.b += weight * r * x
        self.n += weight

    def remove(self, x: np.ndarray, r: float, weight: float = 1.0) -> None:
        x = np.asarray(x, dtype=float)
        self._rank1(x, -weight)
        self.b -= weight * r * x
        self.n -= weight

    def theta(self) -> np.ndarray:
        if self._theta is None:
            th = self.V_inv @ self.b
            nrm = float(np.linalg.norm(th))
            if nrm > self.S:
                th = th * (self.S / nrm)
            self._theta = th
        return self._theta

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.atleast_2d(X) @ self.theta()

    def width(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", X, self.V_inv, X), 0.0))

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X) + self.beta * self.width(X)

    def exact_theta(self) -> np.ndarray:
        """Direct solve (for tests)."""
        return np.linalg.solve(self.V, self.b)


class PriorEstimator(RidgeUCB):
    """P, the prior-driven source. Ridge fit on the pre-interaction rows D_prior = (X, r)."""

    def __init__(self, d, lam=1.0, beta=0.5, S=5.0, alpha: float = 0.0):
        super().__init__(d, lam, beta, S)
        self.alpha = float(alpha)  # live-row mixing weight (0 = prior never changes)

    def fit_from(self, D_prior: Optional[Tuple[np.ndarray, np.ndarray]]) -> "PriorEstimator":
        self.reset()
        if D_prior is not None:
            X, r = D_prior
            for xi, ri in zip(np.asarray(X, float), np.asarray(r, float)):
                self.add(xi, ri)
        return self

    def refit(self, D) -> "PriorEstimator":
        return self.fit_from(D)


class FeedbackEstimator(RidgeUCB):
    """F, the feedback-driven source. Sliding-window ridge on this user's live rows.

    The ridge state covers the rows with timestamp s > t - window (``rows``). Rows that leave the
    window stay in ``history`` (the ``max_history`` most recent), so a later and larger window can be
    rebuilt exactly. ``purge_older_than`` is the drift suppression step and drops them for good.
    """

    def __init__(self, d, lam=1.0, beta=0.5, S=5.0, window: int = 400, max_history: Optional[int] = None):
        super().__init__(d, lam, beta, S)
        self.window = int(window)
        self.max_history = int(max_history) if max_history is not None else self.window
        self.history: deque = deque()  # (t, x, r), oldest first, a suffix of it is active
        self.rows: deque = deque()     # active rows (s > t - window)

    def _sync(self, t: int) -> None:
        """Make ``rows`` equal to the history rows with s > t - window."""
        t_cut = int(t) - self.window + 1  # keep s >= t_cut
        while self.rows and self.rows[0][0] < t_cut:
            _, x, r = self.rows.popleft()
            self.remove(x, r)
        first_active = self.rows[0][0] if self.rows else int(t) + 1
        # the window grew, so history rows with t_cut <= s < first_active are re-added
        for s, x, r in reversed(self.history):
            if s >= first_active:
                continue
            if s < t_cut:
                break
            self.rows.appendleft((s, x, r))
            self.add(x, r)
        while len(self.history) > max(self.max_history, len(self.rows)):
            self.history.popleft()

    def add_live(self, t: int, x: np.ndarray, r: float) -> None:
        row = (int(t), np.asarray(x, float), float(r))
        self.history.append(row)
        self.rows.append(row)
        self.add(x, r)
        self._sync(t)

    def set_window(self, w: int, t: int) -> None:
        self.window = int(max(1, w))
        self._sync(t)

    def purge_older_than(self, t0: int) -> None:
        """Drop every row with s < t0 for good. This is the drift suppression step."""
        t0 = int(t0)
        # TODO(dhaya): a bisect on the timestamps would be faster than popping one by one,
        # but the deque is short (a few hundred rows) so it has never shown up in profiling
        while self.history and self.history[0][0] < t0:
            self.history.popleft()
        while self.rows and self.rows[0][0] < t0:
            _, x, r = self.rows.popleft()
            self.remove(x, r)

    def __len__(self) -> int:
        return len(self.rows)


class CombinedEstimator(RidgeUCB):
    """C, the pooled source. Ridge fit on the prior rows and every live row since the last purge.
    There is no sliding window. Under a stationary preference this equals warm-started LinUCB.
    After a drift it still carries the stale prior rows, so it misleads whenever the prior does.
    ``rebuild(rows)`` restores the prior state and re-adds the kept rows."""

    def __init__(self, P: RidgeUCB):
        super().__init__(P.d, P.lam, P.beta, P.S)
        self.P = P
        self.rebuild([])

    def rebuild(self, rows: Sequence[Tuple[int, np.ndarray, float]]) -> None:
        self.V = self.P.V.copy()
        self.V_inv = self.P.V_inv.copy()
        self.b = self.P.b.copy()
        self.n = float(self.P.n)
        self._theta = None
        for _, x, r in rows:
            self.add(x, r)


class DiscountedRidge:
    """B5, discounted LinUCB. V and b decay by gamma each round. Same predict and score interface."""

    def __init__(self, d, lam=1.0, beta=0.5, S=5.0, gamma=0.995):
        self.d, self.lam, self.beta, self.S, self.gamma = d, lam, beta, S, gamma
        self.V = lam * np.eye(d)
        self.b = np.zeros(d)
        self._theta = None
        self._V_inv = None

    def add(self, x, r, weight=1.0):
        self.V = self.gamma * self.V + (1 - self.gamma) * self.lam * np.eye(self.d) + weight * np.outer(x, x)
        self.b = self.gamma * self.b + weight * r * x
        self._theta = None
        self._V_inv = None

    def _inv(self):
        if self._V_inv is None:
            self._V_inv = np.linalg.inv(self.V)
        return self._V_inv

    def theta(self):
        if self._theta is None:
            th = self._inv() @ self.b
            n = np.linalg.norm(th)
            self._theta = th * (self.S / n) if n > self.S else th
        return self._theta

    def predict(self, X):
        return np.atleast_2d(X) @ self.theta()

    def width(self, X):
        X = np.atleast_2d(X)
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", X, self._inv(), X), 0.0))

    def score(self, X):
        return self.predict(X) + self.beta * self.width(X)


# ----------------------------------------------------------------------------------------------
# switch, detectors, window selector
# ----------------------------------------------------------------------------------------------
class EvidenceSwitch:
    """Meta-controller. Picks the evidence source to condition on each round.

    kind="hedge" is the default. Full-information exponential weights. Every source's squared
    prediction error on the observed (x_t, r_t) is known after the round. All sources are charged
    their clipped loss and no importance weighting is needed.
    kind="exp3" is adversarial EXP3. Only the active source is credited, with an importance-weighted,
    clipped and rescaled gain. This is ablation A10.
    kind="ucb1" is stochastic UCB1 over the sources. Ablation A7 only.
    """

    SOURCES = ("C", "F")

    def __init__(self, kind: str = "hedge", c: float = 1.0, gamma: float = 0.0, G: float = 1.0,
                 rng: Optional[np.random.Generator] = None, adaptive_gamma: bool = False,
                 sources: Optional[Sequence[str]] = None, eta: float = 1.0):
        assert kind in ("hedge", "exp3", "ucb1"), kind
        self.SOURCES = tuple(sources) if sources is not None else ("C", "F")
        assert "F" in self.SOURCES and len(self.SOURCES) >= 2, self.SOURCES
        assert len(set(self.SOURCES)) == len(self.SOURCES), self.SOURCES
        self.kind, self.c, self.gamma, self.G = kind, float(c), float(gamma), float(G)
        self.eta = float(eta)
        self.gamma0 = self.gamma
        self.adaptive_gamma = bool(adaptive_gamma)  # gamma_t = min(gamma0, sqrt(K ln K / (e-1) n_t))
        self.n_credited = 0
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.K = len(self.SOURCES)
        self.logw = np.zeros(self.K)         # EXP3 log-weights
        self.counts = np.zeros(self.K)       # UCB1
        self.means = np.zeros(self.K)
        self.t = 0
        self.burn_in_remaining = 0
        self.last_probs = np.full(self.K, 1.0 / self.K)
        self.forced = False

    # -- probabilities ---------------------------------------------------------------------
    def probs(self) -> np.ndarray:
        if self.kind == "ucb1":
            raise ValueError("probs() not defined for ucb1")
        if self.adaptive_gamma and self.kind == "exp3":
            n = max(self.n_credited, 1)
            self.gamma = min(self.gamma0, math.sqrt(self.K * math.log(self.K) / ((math.e - 1) * n)))
        w = np.exp(self.logw - self.logw.max())
        p = (1 - self.gamma) * w / w.sum() + self.gamma / self.K
        return p / p.sum()

    def select(self, t: int) -> str:
        if self.burn_in_remaining > 0:
            self.forced = True
            self.last_probs = np.eye(self.K)[self.SOURCES.index("F")]
            return "F"
        self.forced = False
        if self.kind in ("exp3", "hedge"):
            p = self.probs()
            self.last_probs = p
            i = int(self.rng.choice(self.K, p=p))
            # i = int(np.argmax(p))  # deterministic variant, tried in Jan, made A10 worse
        else:
            if np.any(self.counts == 0):
                i = int(np.argmin(self.counts))
            else:
                ucb = self.means + self.c * np.sqrt(np.log(max(self.t, 1)) / self.counts)
                i = int(np.argmax(ucb))
            self.last_probs = np.eye(self.K)[i]
        return self.SOURCES[i]

    def rescale(self, gain: float) -> float:
        g = float(np.clip(gain, -self.G, self.G))
        return (g + self.G) / (2 * self.G)  # -> [0,1]

    def update(self, m: str, gain: float, losses: Optional[Dict[str, float]] = None,
               credited: Optional[bool] = None) -> None:
        i = self.SOURCES.index(m)
        g01 = self.rescale(gain)
        if not (credited if credited is not None else not self.forced):
            return  # forced burn-in rounds are not choices of the switch, so no credit
        if self.kind == "hedge":
            assert losses is not None, "hedge needs every source's loss"
            ell = np.array([min(max(float(losses[s]), 0.0), self.G) / self.G for s in self.SOURCES])
            self.logw -= self.eta * ell
            self.logw -= self.logw.max()
            self.n_credited += 1
        elif self.kind == "exp3":
            p = max(float(self.last_probs[i]), 1e-6)
            self.logw[i] += self.gamma * (g01 / p) / self.K
            self.logw -= self.logw.max()  # keep numerically bounded
            self.n_credited += 1
        else:
            self.counts[i] += 1
            self.means[i] += (g01 - self.means[i]) / self.counts[i]

    def force_feedback(self, B: int) -> None:
        self.burn_in_remaining = int(B)

    def reset_source(self, m: str) -> None:
        """Forget the credit of source m. It starts again as no better than the other."""
        i = self.SOURCES.index(m)
        if self.kind in ("exp3", "hedge"):
            self.logw[i] = float(self.logw.min())
        else:
            self.counts[i] = 0.0
            self.means[i] = 0.0

    def tick(self) -> None:
        self.t += 1
        if self.burn_in_remaining > 0:
            self.burn_in_remaining -= 1


class PageHinkley:
    """Page-Hinkley test for an upward mean shift of a nonnegative statistic such as a squared error.

    Fires when m_t - min_s m_s > lam_threshold, with m_t = alpha_forget*m_{t-1} + (x_t - mean_t - delta).
    The running mean is reset after each firing.
    """

    def __init__(self, delta=0.005, lam_threshold=50.0, alpha_forget=0.999, min_instances=30):
        self.delta, self.lam, self.alpha, self.min_instances = delta, lam_threshold, alpha_forget, int(min_instances)
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m = 0.0
        self.m_min = 0.0

    def update(self, x: float) -> bool:
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.m = self.alpha * self.m + (x - self.mean - self.delta)
        self.m_min = min(self.m_min, self.m)
        # if self.n % 50 == 0: print(f"PH t={self.n} m={self.m:.3f} m_min={self.m_min:.3f}")
        if self.n >= self.min_instances and (self.m - self.m_min) > self.lam:
            self.reset()
            return True
        return False


class ADWIN:
    """ADWIN (Bifet and Gavalda 2007). Plain O(n) split search with a capped window."""

    def __init__(self, delta=0.002, min_len=10, max_len=1000):
        self.delta, self.min_len, self.max_len = delta, int(min_len), int(max_len)
        self.reset()

    def reset(self):
        self.win: deque = deque()

    def update(self, x: float) -> bool:
        self.win.append(float(x))
        if len(self.win) > self.max_len:
            self.win.popleft()
        n = len(self.win)
        if n < 2 * self.min_len:
            return False
        # TODO: the exponential bucket compression of the original paper is not implemented,
        # this is the plain O(n) split search over the full window. Fine for max_len <= 2000.
        arr = np.fromiter(self.win, float, n)
        cs = np.cumsum(arr)
        total = cs[-1]
        fired = False
        for k in range(self.min_len, n - self.min_len):
            n0, n1 = k, n - k
            mu0, mu1 = cs[k - 1] / n0, (total - cs[k - 1]) / n1
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            dp = self.delta / n
            var = arr.var() + 1e-12
            eps = math.sqrt(2.0 / m * var * math.log(2.0 / dp)) + 2.0 / (3.0 * m) * math.log(2.0 / dp)
            if abs(mu0 - mu1) > eps:
                for _ in range(k):
                    self.win.popleft()
                fired = True
                break
        return fired


class GLRMeanChange:
    """Gaussian GLR test for a single mean change with unknown common variance.

    This is the detector family behind the detection-augmented procedures of Huang et al. 2025
    and Gerogiannis et al. 2025. Fires when
        max_s  n/2 * log(sigma0^2 / sigma1^2(s))  >  threshold + 1.5 * log(n)
    where sigma0^2 is the variance of the whole stream and sigma1^2(s) the pooled within-segment
    variance of the split at s. The stream is capped at ``max_len`` and cleared on a firing."""

    def __init__(self, threshold=5.0, min_len=10, max_len=2000):
        self.threshold, self.min_len, self.max_len = float(threshold), int(min_len), int(max_len)
        self.reset()

    def reset(self):
        self.win: deque = deque()

    def update(self, x: float) -> bool:
        self.win.append(float(x))
        if len(self.win) > self.max_len:
            self.win.popleft()
        n = len(self.win)
        if n < 2 * self.min_len:
            return False
        arr = np.fromiter(self.win, float, n)
        cs, cs2 = np.cumsum(arr), np.cumsum(arr * arr)
        s = np.arange(self.min_len, n - self.min_len + 1)
        n0, n1 = s.astype(float), (n - s).astype(float)
        sum0, sum1 = cs[s - 1], cs[-1] - cs[s - 1]
        sq0, sq1 = cs2[s - 1], cs2[-1] - cs2[s - 1]
        rss1 = (sq0 - sum0 ** 2 / n0) + (sq1 - sum1 ** 2 / n1)
        rss0 = cs2[-1] - cs[-1] ** 2 / n
        stat = 0.5 * n * np.log((rss0 + 1e-12) / (rss1 + 1e-12))
        if float(stat.max()) > self.threshold + 1.5 * math.log(n):
            self.reset()
            return True
        return False


def make_detector(cfg_det: dict):
    if cfg_det["type"] == "page_hinkley":
        return PageHinkley(cfg_det["delta"], cfg_det["lam_threshold"], cfg_det["alpha_forget"],
                           cfg_det["min_instances"])
    if cfg_det["type"] == "adwin":
        return ADWIN(cfg_det.get("delta", 0.002))
    raise ValueError(cfg_det["type"])


class BOBWindowSelector:
    """Bandit-over-Bandit. EXP3 over candidate sliding windows, one decision per block of H rounds."""

    def __init__(self, candidate_windows: Sequence[int], H: int, gamma: float = 0.1,
                 rng: Optional[np.random.Generator] = None):
        self.windows = [int(w) for w in candidate_windows]
        self.H, self.gamma = int(H), float(gamma)
        self.K = len(self.windows)
        self.logw = np.zeros(self.K)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.last_p = np.full(self.K, 1.0 / self.K)
        self.last_i = 0

    def probs(self) -> np.ndarray:
        w = np.exp(self.logw - self.logw.max())
        return (1 - self.gamma) * w / w.sum() + self.gamma / self.K

    def pick(self) -> int:
        p = self.probs()
        self.last_p = p
        self.last_i = int(self.rng.choice(self.K, p=p))
        return self.windows[self.last_i]

    def update(self, w: int, block_reward: float) -> None:
        i = self.windows.index(int(w))
        g = float(np.clip(block_reward, 0.0, 1.0))
        self.logw[i] += self.gamma * (g / max(self.last_p[i], 1e-6)) / self.K
        self.logw -= self.logw.max()


# ----------------------------------------------------------------------------------------------
# agents
# ----------------------------------------------------------------------------------------------
@dataclass
class Flags:
    no_switch: bool = False        # A1, no controller, fixed prior-then-F schedule
    no_suppression: bool = False   # A2, the detector fires but nothing is purged, reset or forced
    no_detector: bool = False      # A3, the detector never fires
    no_bob: bool = False           # A4, fixed window w_0
    decay_instead: bool = False    # A5, on drift the prior is down-weighted instead of suppressed
    uncertainty_gain: bool = False # A6, gain is the UCB-width reduction instead of the realised error gain
    switch_kind: Optional[str] = None  # A7 uses "ucb1"
    gated: bool = False            # consult the switch only when the sources disagree on the arm
    adaptive_gamma: bool = False   # V2, the EXP3 exploration rate decays with the credited rounds
    detector_signal: str = "active"  # which residual drives the detector. active, F or mean
    no_pooled: bool = False        # A8, drop the pooled source C and switch over P and F only
    no_gate: bool = False          # A9, always consult the switch, even when every source agrees
    purge_keep: Optional[Union[int, str]] = None  # rows kept after a drift. None takes the cfg value

    def label(self) -> str:
        on = [k for k, v in self.__dict__.items() if v is True]
        if self.switch_kind:
            on.append(f"switch={self.switch_kind}")
        if self.detector_signal != "active":
            on.append(f"detector={self.detector_signal}")
        if self.purge_keep is not None:
            on.append(f"purge_keep={self.purge_keep}")
        return "DES-UCB" if not on else "DES-UCB[" + ",".join(on) + "]"


def _cfg_get(cfg: dict, key: str, default=None):
    return cfg.get(key, default)


class BaseAgent:
    """step(t, X_cand) returns idx. observe(t, x, r) updates. source, window and drift_fired are logged."""

    name = "agent"
    needs_env = False

    def __init__(self):
        self.source = "-"
        self.window = -1
        self.drift_fired = False

    def step(self, t: int, X_cand: np.ndarray) -> int:  # pragma: no cover
        raise NotImplementedError

    def observe(self, t: int, x: np.ndarray, r: float) -> None:  # pragma: no cover
        raise NotImplementedError


class DESUCBAgent(BaseAgent):
    name = "DES-UCB"

    def __init__(self, d: int, D_prior, cfg: dict, flags: Optional[Flags] = None,
                 rng: Optional[np.random.Generator] = None):
        super().__init__()
        self.cfg = cfg
        self.flags = flags or Flags()
        self.rng = rng if rng is not None else np.random.default_rng(cfg.get("seed", 0))
        lam, beta, S = cfg["lam"], cfg["beta"], cfg["S"]
        self.P = PriorEstimator(d, lam, beta, S, alpha=cfg.get("alpha_prior_mix", 0.0)).fit_from(D_prior)
        self.F = FeedbackEstimator(d, lam, beta, S, window=cfg["w_0"],
                                   max_history=max([cfg["w_0"], *cfg["candidate_windows"]]))
        self.C = CombinedEstimator(self.P)
        sw = cfg["switch"]
        kind = self.flags.switch_kind or sw["kind"]
        sources = [str(s) for s in sw.get("sources", ["C", "F"])]
        if self.flags.no_pooled:
            # A8. the prior-carrying source is the prior alone, the two-source design P against F
            sources = list(dict.fromkeys("P" if s == "C" else s for s in sources))
        self.prior_src = "C" if "C" in sources else "P"
        gamma = sw["gamma"] if kind == "hedge" else sw.get("exp3_gamma", sw["gamma"])
        self.switch = EvidenceSwitch(kind, sw["c"], gamma, sw["G"], rng=self.rng,
                                     adaptive_gamma=self.flags.adaptive_gamma, sources=sources,
                                     eta=sw.get("eta", 1.0))
        self.gate = bool(self.flags.gated or (sw.get("gate", False) and not self.flags.no_gate))
        pk = self.flags.purge_keep if self.flags.purge_keep is not None else cfg.get("purge_keep", "half")
        self.purge_keep = pk  # "half" means w/2 but at least w_min, otherwise a number of rows
        self.detector = make_detector(cfg["detector"])
        assert self.flags.detector_signal in ("active", "F", "mean"), self.flags.detector_signal
        self.bob = BOBWindowSelector(cfg["candidate_windows"], cfg["H"], cfg["bob_gamma"], rng=self.rng)
        self.H = int(cfg["H"])
        self.w_min = int(cfg["w_min"])
        self.B = int(cfg["burn_in_B"])
        self.window = int(cfg["w_0"])
        self.bob_w: Optional[int] = None
        self.source = self.prior_src
        self.m = self.prior_src
        self.block_rewards: List[float] = []
        self.fired_at: List[int] = []
        self.prior_weight = 1.0  # used only by decay_instead
        self.trace: List[str] = []
        self.forced = True
        self.agreed = False
        self.env: Optional["Env"] = None  # set by the runner for the full-information diagnostic
        self.extra: Dict[str, float] = {}
        self._last_X: Optional[np.ndarray] = None

    # -- helpers --------------------------------------------------------------------------
    def _est(self, m: str):
        return {"P": self.P, "F": self.F, "C": self.C}[m]

    def _score(self, m: str, X: np.ndarray) -> np.ndarray:
        if m == self.prior_src and self.flags.decay_instead and self.prior_weight < 1.0:
            # decayed prior. shrink the prior-carrying score towards the F score instead of purging
            return self.prior_weight * self._est(m).score(X) + (1 - self.prior_weight) * self.F.score(X)
        return self._est(m).score(X)

    def _predict(self, m: str, X: np.ndarray) -> np.ndarray:
        if m == self.prior_src and self.flags.decay_instead and self.prior_weight < 1.0:
            return self.prior_weight * self._est(m).predict(X) + (1 - self.prior_weight) * self.F.predict(X)
        return self._est(m).predict(X)

    def _f_eligible(self) -> bool:
        """F enters the switch and the agreement gate only once it holds ``w_min`` live rows.
        A forced burn-in overrides this. ``on_drift`` sizes the burn-in so that F is never handed
        back to the prior-carrying source while still below the eligibility count."""
        return self.switch.burn_in_remaining > 0 or len(self.F) >= self.w_min

    def _select_source(self, t: int) -> str:
        if not self._f_eligible():
            # such rounds are forced to the prior-carrying source and not credited to the switch
            self.switch.forced = True
            return self.prior_src
        if self.flags.no_switch:
            # A1, no controller. prior until F holds w_min rows, then F
            # suppression can still force F
            if self.switch.burn_in_remaining > 0:
                self.switch.forced = True
                return "F"
            self.switch.forced = False
            return self.prior_src if len(self.F) < self.w_min else "F"
        return self.switch.select(t)

    # -- interface ------------------------------------------------------------------------
    def step(self, t: int, X_cand: np.ndarray) -> int:
        self.drift_fired = False
        self.agreed = False
        self._last_X = X_cand
        if self.gate and self.switch.burn_in_remaining == 0 and len(self.F) >= self.w_min:
            arms = {m: int(np.argmax(self._score(m, X_cand))) for m in self.switch.SOURCES}
            if len(set(arms.values())) == 1:
                # every source recommends the same arm, so there is nothing to decide
                # no exploration cost, and the round is not credited to the switch
                self.agreed = True
                self.switch.forced = True
                self.m = "F"
                self.source = self.m
                self.forced = True
                return arms["F"]
        self.m = self._select_source(t)
        self.source = self.m
        self.forced = bool(self.switch.forced)
        s = self._score(self.m, X_cand)
        return int(np.argmax(s))

    def observe(self, t: int, x: np.ndarray, r: float) -> None:
        x = np.asarray(x, float)
        m = self.m
        others = [s for s in self.switch.SOURCES if s != m]
        # 1. realised gain of the active source before the update
        #    squared-error reduction from conditioning on m rather than on the best other source
        pred_m = float(self._predict(m, x)[0])
        if self.flags.uncertainty_gain:
            wm = float(self._est(m).width(x)[0])
            wo = min(float(self._est(o).width(x)[0]) for o in others)
            gain = (wo - wm) / (wo + wm + 1e-12)
        else:
            e_m = (pred_m - r) ** 2
            e_o = min((float(self._predict(o, x)[0]) - r) ** 2 for o in others)
            gain = (e_o - e_m) / (e_o + e_m + 1e-12)  # scale-free relative error reduction in [-1, 1]
            # gain = np.clip(e_o - e_m, -self.G, self.G) / self.G  # old absolute version, noise-level dependent
        sq = {s: (float(self._predict(s, x)[0]) - r) ** 2 for s in self.switch.SOURCES}
        # scale-free share of the round's squared error, summing to one over the sources
        # so the hedge weights move at the same rate whatever the reward noise level
        tot = sum(sq.values()) + 1e-12
        losses = {s: v / tot for s, v in sq.items()}
        e_P = sq.get("P", (float(self._predict("P", x)[0]) - r) ** 2)
        e_F = sq["F"]
        self.extra = {"e_P": e_P, "e_F": e_F, "gain": float(gain), "agreed": float(self.agreed)}
        if "C" in sq:
            self.extra["e_C"] = sq["C"]
        if self.env is not None and self._last_X is not None:
            # full-information diagnostic, synthetic only
            # expected reward of the arm each source would have played, whatever source was active
            aP = int(np.argmax(self._score(self.prior_src, self._last_X)))
            aF = int(np.argmax(self._score("F", self._last_X)))
            self.extra["pol_gap_F_minus_P"] = self.env.expected_reward(t, aF) - self.env.expected_reward(t, aP)
        # 2. update evidence
        self.F.add_live(t, x, r)
        self.C.add(x, r)
        if m == "P" and self.P.alpha > 0:
            self.P.add(x, r, self.P.alpha)
        # 3. credit the switch
        #    full-information losses exist on every round, so the hedge switch is also charged on
        #    agreed rounds and on cold-start rounds where F is not yet eligible but still predicts
        #    burn-in rounds are never credited
        credited = None
        if self.switch.kind == "hedge" and self.switch.burn_in_remaining == 0 and len(self.F) >= self.F.d:
            credited = True
        self.switch.update(m, gain, losses=losses, credited=credited)
        self.switch.tick()
        # 4. drift detection on a squared residual
        #    the active source by default, F only as a common signal, or the mean of both
        if self.flags.detector_signal == "F":
            e = e_F
        elif self.flags.detector_signal == "mean":
            e = 0.5 * (e_P + e_F)
        else:
            e = (r - pred_m) ** 2
        if not self.flags.no_detector and self.detector.update(e):
            self.on_drift(t)
        # 5. bandit-over-bandit window selection over blocks [kH, (k+1)H)
        #    at the last round of a block credit the window active for the whole block, then pick the next
        self.block_rewards.append(r)
        if not self.flags.no_bob and (t + 1) % self.H == 0:
            if self.bob_w is not None:
                self.bob.update(self.bob_w, self._reward01(float(np.mean(self.block_rewards))))
            self.block_rewards = []
            self.bob_w = self.bob.pick()
            self.window = self.bob_w
            self.F.set_window(self.bob_w, t)
        self.trace.append(m)

    def _reward01(self, r: float) -> float:
        lo, hi = self.cfg.get("reward_range", (-2.0, 2.0))
        return float(np.clip((r - lo) / (hi - lo), 0.0, 1.0))

    def on_drift(self, t: int) -> None:
        # FIXME: two firings in the same block are handled, but the second purge is a no-op
        # when purge_keep=0. Worth a test at some point.
        self.fired_at.append(int(t))
        self.drift_fired = True
        if self.flags.no_suppression:
            return
        if self.flags.decay_instead:
            self.prior_weight *= 0.5  # passive decay of the prior's influence
            return
        if self.purge_keep == "half":
            w = max(self.w_min, self.window // 2)
            # the running BOB block is only spoiled if the window changes before the block is complete
            # on a block's last round every reward, this one included, was already drawn under bob_w
            if w != self.window and (t + 1) % self.H != 0:
                self.bob_w = None
                self.block_rewards = []
            self.window = w
            self.F.set_window(w, t)
            self.F.purge_older_than(t - w + 1)  # same cutoff as the active window, keep s > t - w only
        else:
            # keep only the purge_keep most recent rows, 0 restarts F completely
            # the window itself is left to BOB and the purge cannot be undone
            self.F.purge_older_than(t - int(self.purge_keep) + 1)
        self.C.rebuild(self.F.history)
        for s in self.switch.SOURCES:
            if s != "F":
                self.switch.reset_source(s)
        self.switch.force_feedback(max(self.B, self.w_min - len(self.F)))


class LinUCBAgent(BaseAgent):
    """B1 cold LinUCB and B2 warm-start LinUCB. Neither forgets."""

    def __init__(self, d, D_prior, cfg, warm: bool = False):
        super().__init__()
        self.name = "B2-LinUCB-warm" if warm else "B1-LinUCB"
        self.est = PriorEstimator(d, cfg["lam"], cfg["beta"], cfg["S"]).fit_from(D_prior if warm else None)
        self.source = "P+F" if warm else "F"

    def step(self, t, X_cand):
        return int(np.argmax(self.est.score(X_cand)))

    def observe(self, t, x, r):
        self.est.add(x, r)


class SWUCBAgent(BaseAgent):
    """B3 SW-UCB with a fixed window. Warm-started from D_prior like every other baseline that
    sees the prior. ``sw_warm`` in ``baselines`` turns this off."""

    name = "B3-SW-UCB"

    def __init__(self, d, D_prior, cfg, window: Optional[int] = None):
        super().__init__()
        self.est = FeedbackEstimator(d, cfg["lam"], cfg["beta"], cfg["S"],
                                     window=window or cfg["baselines"]["sw_window"])
        self.window = self.est.window
        self.source = "F"

    def step(self, t, X_cand):
        return int(np.argmax(self.est.score(X_cand)))

    def observe(self, t, x, r):
        self.est.add_live(t, x, r)


class BOBAgent(BaseAgent):
    """B4 Bandit-over-Bandit SW-UCB. EXP3 over candidate windows with block length H."""

    name = "B4-BOB"

    def __init__(self, d, D_prior, cfg, rng=None):
        super().__init__()
        self.cfg = cfg
        rng = rng if rng is not None else np.random.default_rng(cfg.get("seed", 0))
        self.est = FeedbackEstimator(d, cfg["lam"], cfg["beta"], cfg["S"], window=cfg["w_0"],
                                     max_history=max([cfg["w_0"], *cfg["candidate_windows"]]))
        bob_gamma = cfg["baselines"].get("bob_gamma") or cfg["bob_gamma"]
        self.bob = BOBWindowSelector(cfg["candidate_windows"], cfg["H"], bob_gamma, rng=rng)
        self.H = int(cfg["H"])
        self.window = int(cfg["w_0"])
        self.bob_w: Optional[int] = None
        self.block: List[float] = []
        self.source = "F"

    def step(self, t, X_cand):
        return int(np.argmax(self.est.score(X_cand)))

    def observe(self, t, x, r):
        self.est.add_live(t, x, r)
        self.block.append(r)
        if (t + 1) % self.H == 0:
            if self.bob_w is not None:
                lo, hi = self.cfg.get("reward_range", (-2.0, 2.0))
                self.bob.update(self.bob_w, float(np.clip((np.mean(self.block) - lo) / (hi - lo), 0, 1)))
            self.block = []
            self.bob_w = self.bob.pick()
            self.window = self.bob_w
            self.est.set_window(self.window, t)


class DiscountedLinUCBAgent(BaseAgent):
    name = "B5-D-LinUCB"

    def __init__(self, d, D_prior, cfg):
        super().__init__()
        self.est = DiscountedRidge(d, cfg["lam"], cfg["beta"], cfg["S"], cfg["baselines"]["discount_gamma"])
        self.source = "F"

    def step(self, t, X_cand):
        return int(np.argmax(self.est.score(X_cand)))

    def observe(self, t, x, r):
        self.est.add(np.asarray(x, float), r)


class PHRestartLinUCBAgent(BaseAgent):
    """B6 LinUCB restarted whenever Page-Hinkley fires on the squared residual."""

    name = "B6-PH-restart"

    def __init__(self, d, D_prior, cfg):
        super().__init__()
        self.est = RidgeUCB(d, cfg["lam"], cfg["beta"], cfg["S"])
        det_cfg = dict(cfg["detector"])
        if cfg["baselines"].get("ph_restart_lam_threshold") is not None:
            det_cfg["lam_threshold"] = cfg["baselines"]["ph_restart_lam_threshold"]
        self.det = make_detector(det_cfg)
        self.fired_at: List[int] = []
        self.source = "F"

    def step(self, t, X_cand):
        self.drift_fired = False
        return int(np.argmax(self.est.score(X_cand)))

    def observe(self, t, x, r):
        e = (r - float(self.est.predict(x)[0])) ** 2
        self.est.add(x, r)
        if self.det.update(e):
            self.est.reset()
            self.fired_at.append(int(t))
            self.drift_fired = True


class DALLinUCBAgent(BaseAgent):
    """B10 detection-augmented LinUCB, after DAL (Gerogiannis et al. 2025) and the modular DAB
    procedures (Huang et al. 2025).

    A stationary base learner, forced exploration at rate alpha, and a quickest-change detector on
    the forced rounds only. The base learner restarts in full on detection. Candidates here are
    fresh every round, so the forced round plays a uniformly random candidate and the detector
    watches the base model's squared residual on that round. The signed residual of a random arm has
    mean zero before and after a parameter change. The squared one shifts. The monitored stream
    therefore does not depend on the base policy's arm choice. As in DAL the forced rounds pay the
    price of a random arm. The base learner starts warm from the prior like B2 and restarts cold
    like B6."""

    name = "B10-DAL-LinUCB"

    def __init__(self, d, D_prior, cfg, rng=None):
        super().__init__()
        b = cfg["baselines"]
        self.est = PriorEstimator(d, cfg["lam"], cfg["beta"], cfg["S"]).fit_from(D_prior)
        self.alpha = float(b.get("dal_alpha", 0.1))
        self.det = GLRMeanChange(b.get("dal_glr_threshold", 5.0), b.get("dal_min_len", 10))
        self.rng = rng if rng is not None else np.random.default_rng(cfg.get("seed", 0))
        self.fired_at: List[int] = []
        self.source = "P+F"
        self.explore = False

    def step(self, t, X_cand):
        self.drift_fired = False
        self.explore = bool(self.rng.random() < self.alpha)
        self.forced = self.explore
        if self.explore:
            return int(self.rng.integers(len(X_cand)))
        return int(np.argmax(self.est.score(X_cand)))

    def observe(self, t, x, r):
        resid = r - float(self.est.predict(x)[0])
        self.est.add(x, r)
        if self.explore and self.det.update(resid * resid):
            self.est.reset()
            self.source = "F"
            self.fired_at.append(int(t))
            self.drift_fired = True


class PeriodicRestartLinUCBAgent(BaseAgent):
    """B11 restarted LinUCB. The estimator is re-initialised every ``restart_period`` rounds.

    This is the restart schedule of Zhao et al. 2020. The restarted weighted-OFUL variants of
    Wang et al. 2024 build on it, and with homoscedastic noise their variance weights are constant.
    Starts warm from the prior and restarts cold."""

    name = "B11-Restart-LinUCB"

    def __init__(self, d, D_prior, cfg):
        super().__init__()
        self.est = PriorEstimator(d, cfg["lam"], cfg["beta"], cfg["S"]).fit_from(D_prior)
        self.period = int(cfg["baselines"].get("restart_period", 500))
        self.fired_at: List[int] = []
        self.source = "P+F"

    def step(self, t, X_cand):
        self.drift_fired = False
        return int(np.argmax(self.est.score(X_cand)))

    def observe(self, t, x, r):
        self.est.add(x, r)
        if (t + 1) % self.period == 0:
            self.est.reset()
            self.source = "F"
            self.fired_at.append(int(t))
            self.drift_fired = True


class PriorOnlyAgent(BaseAgent):
    name = "B7-Prior-only"

    def __init__(self, d, D_prior, cfg):
        super().__init__()
        self.est = PriorEstimator(d, cfg["lam"], cfg["beta"], cfg["S"]).fit_from(D_prior)
        self.source = "P"

    def step(self, t, X_cand):
        return int(np.argmax(self.est.score(X_cand)))

    def observe(self, t, x, r):
        pass


class RandomAgent(BaseAgent):
    name = "B8-Random"

    def __init__(self, d, D_prior, cfg, rng=None):
        super().__init__()
        self.rng = rng if rng is not None else np.random.default_rng(cfg.get("seed", 0))

    def step(self, t, X_cand):
        return int(self.rng.integers(len(X_cand)))

    def observe(self, t, x, r):
        pass


class OracleAgent(BaseAgent):
    """B9. Picks the candidate with the highest hidden expected reward on synthetic data, or the
    best available reward on real data. Needs the environment."""

    name = "B9-Oracle"
    needs_env = False

    def __init__(self, d, D_prior, cfg):
        super().__init__()
        self.env = None

    def step(self, t, X_cand):
        return int(self.env.oracle_idx(t))

    def observe(self, t, x, r):
        pass


def make_agent(name: str, d: int, D_prior, cfg: dict, seed: int = 0) -> BaseAgent:
    """Agent factory by name. DES-UCB ablations are 'A1' to 'A10' or 'DES-UCB[flag,...]'."""
    rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
    if name == "DES-UCB":
        return DESUCBAgent(d, D_prior, cfg, Flags(), rng=rng)
    abl = {"A1": Flags(no_switch=True), "A2": Flags(no_suppression=True), "A3": Flags(no_detector=True),
           "A4": Flags(no_bob=True), "A5": Flags(decay_instead=True), "A6": Flags(uncertainty_gain=True),
           "A7": Flags(switch_kind="ucb1"), "A8": Flags(no_pooled=True), "A9": Flags(no_gate=True),
           "A10": Flags(switch_kind="exp3")}
    if name in abl:
        a = DESUCBAgent(d, D_prior, cfg, abl[name], rng=rng)
        a.name = name
        return a
    if "[" in name and name.endswith("]") and name[:name.index("[")] in ("DES-UCB", *abl):
        base = name[:name.index("[")]
        inner = name[name.index("[") + 1:-1]
        fl = copy.copy(abl[base]) if base in abl else Flags()
        for tok in inner.split(","):
            if tok.startswith("switch="):
                fl.switch_kind = tok.split("=")[1]
            elif tok.startswith("detector="):
                fl.detector_signal = tok.split("=")[1]
            elif tok.startswith("purge_keep="):
                v = tok.split("=")[1]
                fl.purge_keep = v if v == "half" else int(v)
            else:
                setattr(fl, tok, True)
        a = DESUCBAgent(d, D_prior, cfg, fl, rng=rng)
        a.name = name
        return a
    table: Dict[str, Callable[[], BaseAgent]] = {
        "B1": lambda: LinUCBAgent(d, D_prior, cfg, warm=False),
        "B2": lambda: LinUCBAgent(d, D_prior, cfg, warm=True),
        "B3": lambda: SWUCBAgent(d, D_prior, cfg),
        "B4": lambda: BOBAgent(d, D_prior, cfg, rng=rng),
        "B5": lambda: DiscountedLinUCBAgent(d, D_prior, cfg),
        "B6": lambda: PHRestartLinUCBAgent(d, D_prior, cfg),
        "B7": lambda: PriorOnlyAgent(d, D_prior, cfg),
        "B8": lambda: RandomAgent(d, D_prior, cfg, rng=rng),
        "B9": lambda: OracleAgent(d, D_prior, cfg),
        "B10": lambda: DALLinUCBAgent(d, D_prior, cfg, rng=rng),
        "B11": lambda: PeriodicRestartLinUCBAgent(d, D_prior, cfg),
    }
    if name in table:
        return table[name]()
    raise KeyError(name)


ALL_BASELINES = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"]
ALL_ABLATIONS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"]
VARIANTS = {
    "V1": "DES-UCB[purge_keep=half]",
    "V2": "DES-UCB[switch=exp3,adaptive_gamma]",
    "V3": "DES-UCB[detector=F]",
    "V4": "DES-UCB[detector=mean]",
}
AGENT_LABELS = {
    "DES-UCB": "DES-UCB", "A1": "A1 no_switch", "A2": "A2 no_suppression", "A3": "A3 no_detector",
    "A4": "A4 no_bob", "A5": "A5 decay_instead", "A6": "A6 uncertainty_gain", "A7": "A7 switch=ucb1",
    "A8": "A8 no_pooled", "A9": "A9 no_gate", "A10": "A10 switch=exp3",
    "B1": "B1 LinUCB", "B2": "B2 LinUCB warm", "B3": "B3 SW-UCB", "B4": "B4 BOB", "B5": "B5 D-LinUCB",
    "B6": "B6 PH-restart", "B7": "B7 Prior-only", "B8": "B8 Random", "B9": "B9 Oracle",
    "B10": "B10 DAL-LinUCB", "B11": "B11 Restart-LinUCB",
    "DES-UCB[purge_keep=half]": "V1 half-window purge",
    "DES-UCB[switch=exp3,adaptive_gamma]": "V2 EXP3 + adaptive gamma",
    "DES-UCB[gated]": "gated switch", "DES-UCB[gated,adaptive_gamma]": "gated + adaptive gamma",
    "DES-UCB[detector=F]": "V3 detector on F residual", "DES-UCB[detector=mean]": "V4 detector on mean residual",
    "A2[detector=F]": "A2 no_suppression, detector on F", "A2[gated]": "A2 no_suppression, gated",
    "A5[gated]": "A5 decay, gated",
}


# ----------------------------------------------------------------------------------------------
# environments
# ----------------------------------------------------------------------------------------------
class Env:
    """Interaction protocol shared by synthetic and real environments.

    candidates(t) gives X_cand. reward(t, idx) gives r. best(t) gives the oracle or best-available
    reward. Also oracle_idx(t), true_change(t), regret_kind in REGRET_KINDS, D_prior, d and T.
    """

    regret_kind = "dynamic"
    d: int
    T: int
    D_prior: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def reset(self) -> None:
        """Clear any agent-dependent state so every agent faces the identical environment."""

    def candidates(self, t: int) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def reward(self, t: int, idx: int) -> float:  # pragma: no cover
        raise NotImplementedError

    def best(self, t: int) -> float:  # pragma: no cover
        raise NotImplementedError

    def expected_reward(self, t: int, idx: int) -> float:
        """Noise-free reward of candidate idx. Regret is best(t) - expected_reward(t, idx).
        Real-data rewards are deterministic given the rating, so the default is reward()."""
        return self.reward(t, idx)

    def oracle_idx(self, t: int) -> int:  # pragma: no cover
        raise NotImplementedError

    def true_change(self, t: int) -> bool:
        return False


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


class SyntheticEnv(Env):
    """Linear contextual bandit with a hidden, time-varying user parameter theta_t.

    Items x_a ~ N(0, I_d), unit norm. K cohort prototypes mu_k. The user's theta_0 is a noisy copy
    of its cohort prototype. Reward r = x^T theta_t + N(0, sigma^2), or Bernoulli(sigmoid). The
    oracle is max_a x_a^T theta_t over the candidate set. This gives dynamic regret.
    """

    regret_kind = "dynamic"

    def __init__(self, cfg: dict, seed: int, user: int):
        self.cfg = cfg
        self.d, self.T = int(cfg["d"]), int(cfg["T"])
        self.N, self.K_cand = int(cfg["N_items"]), int(cfg["K_cand"])
        dr, pr = cfg["drift"], cfg["prior"]
        self.rng = np.random.default_rng([int(seed), int(user), 1234])
        item_rng = np.random.default_rng([int(seed), 999])  # items shared across users of a seed
        self.items = _unit(item_rng.standard_normal((self.N, self.d)))
        self.K = int(pr["K_cohorts"])
        self.protos = _unit(item_rng.standard_normal((self.K, self.d)))
        self.cohort = int(self.rng.integers(self.K))
        theta0 = _unit(self.protos[self.cohort] + 0.3 * self.rng.standard_normal(self.d))
        self.theta0 = theta0
        u = _unit(self.rng.standard_normal(self.d))
        self.reverse = dr.get("direction", "random") == "reverse"
        if self.reverse:
            # misleading prior. the user moves against its pre-drift parameter
            # the prior is not just noisy, it points the wrong way (theta_1 = -theta_0 at magnitude 2)
            u = -theta0
        self.magnitude = float(dr["magnitude"])
        self.theta1 = theta0 + self.magnitude * u
        self.sigma = float(dr["sigma_reward"])
        self.bernoulli = dr.get("reward_model", "gaussian") == "bernoulli"
        self.drift_type = dr["type"]
        self.change_points = [int(c) for c in (dr.get("change_points") or [])]
        self.thetas = self._trajectory(dr)
        # prior evidence rows
        self.p_mismatch = float(pr["p_mismatch"])
        self.mismatched = bool(self.rng.random() < self.p_mismatch)
        if pr["prior_from_pre_drift"]:
            theta_prior = theta0
        else:
            theta_prior = self.protos[self.cohort]  # population-level (cohort) prior
        if self.mismatched:
            other = [k for k in range(self.K) if k != self.cohort]
            theta_prior = self.protos[int(self.rng.choice(other))]
        n_rows = int(pr["n_prior_rows"])
        idx = self.rng.integers(self.N, size=n_rows)
        Xp = self.items[idx]
        rp = Xp @ theta_prior + float(pr["sigma_prior"]) * self.rng.standard_normal(n_rows)
        self.D_prior = (Xp, rp)
        self._cand_cache: Dict[int, np.ndarray] = {}
        self.cand_rng = np.random.default_rng([int(seed), int(user), 77])
        self.cand_idx = self.cand_rng.integers(self.N, size=(self.T, self.K_cand))
        self.noise = self.rng.standard_normal(self.T) * self.sigma
        self.unif = self.rng.random(self.T)
        self._mt = -1
        self._mu: Optional[np.ndarray] = None

    def _abrupt_targets(self) -> list:
        """One target per change point. Under ``direction = reverse`` every target is the negated
        pre-drift parameter, so the prior stays misleading for the whole post-drift run. Otherwise
        each later target adds a fresh random step of the same magnitude."""
        targets = [self.theta1]
        for _ in range(1, len(self.change_points)):
            if self.reverse:
                targets.append(self.theta1)
            else:
                targets.append(targets[-1] + self.magnitude * _unit(self.rng.standard_normal(self.d)))
        return targets

    def _trajectory(self, dr: dict) -> np.ndarray:
        T, th0, th1 = self.T, self.theta0, self.theta1
        th = np.tile(th0, (T + 1, 1))
        typ = dr["type"]
        if typ == "none":
            pass
        elif typ == "abrupt":
            targets = self._abrupt_targets()
            for cp, tgt in zip(self.change_points, targets):
                th[cp:] = tgt
        elif typ == "gradual":
            G = int(dr.get("gradual_G", 300))
            targets = self._abrupt_targets()
            for cp, tgt in zip(self.change_points, targets):
                start = th[cp - 1].copy() if cp > 0 else th0
                for s in range(cp, T + 1):
                    a = min(1.0, (s - cp + 1) / G)
                    th[s] = (1 - a) * start + a * tgt
        elif typ == "recurring":
            P = int(dr.get("recurring_P", 500))
            for s in range(T + 1):
                th[s] = th1 if (s // P) % 2 == 1 else th0
        elif typ == "incremental":
            eps = float(dr.get("incremental_eps", 0.01))
            steps = self.rng.standard_normal((T + 1, self.d))
            walk = np.cumsum(eps * _unit(steps), axis=0)
            th = th0 + walk
        elif typ == "sinusoidal":
            per = float(dr.get("sinusoidal_period", 500))
            u = _unit(th1 - th0)
            for s in range(T + 1):
                th[s] = th0 + self.magnitude * math.sin(2 * math.pi * s / per) * u
        else:
            raise ValueError(typ)
        return th

    def theta_t(self, t: int) -> np.ndarray:
        return self.thetas[min(t, self.T)]

    def candidates(self, t: int) -> np.ndarray:
        return self.items[self.cand_idx[t]]

    def _means(self, t: int) -> np.ndarray:
        if t != self._mt:
            mu = self.candidates(t) @ self.theta_t(t)
            self._mu = 1.0 / (1.0 + np.exp(-mu)) if self.bernoulli else mu
            self._mt = t
        return self._mu

    def reward(self, t: int, idx: int) -> float:
        mu = float(self._means(t)[idx])
        if self.bernoulli:
            return float(self.unif[t] < mu)
        return mu + float(self.noise[t])

    def expected_reward(self, t: int, idx: int) -> float:
        return float(self._means(t)[idx])

    def best(self, t: int) -> float:
        return float(self._means(t).max())

    def oracle_idx(self, t: int) -> int:
        return int(np.argmax(self._means(t)))

    def true_change(self, t: int) -> bool:
        if self.drift_type == "abrupt" or self.drift_type == "gradual":
            return t in self.change_points
        if self.drift_type == "recurring":
            P = int(self.cfg["drift"].get("recurring_P", 500))
            return t > 0 and t % P == 0
        return False


class RatedItemEnv(Env):
    """Real-data environment. Rewards are only revealed for items the user actually rated or clicked.

    The candidate set is K_cand rated and not yet shown items, or the actual impression list.
    'best' is the best available reward in that set. This gives offline_regret, never dynamic regret.
    """

    regret_kind = "offline"

    def __init__(self, item_feats: np.ndarray, rated_items: np.ndarray, rewards: np.ndarray,
                 T: int, K_cand: int, D_prior, seed: int, split_at: Optional[int] = None,
                 impressions: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
                 day_boundaries: Optional[List[int]] = None):
        self.feats = item_feats
        self.d = item_feats.shape[1]
        self.D_prior = D_prior
        self.rng = np.random.default_rng([seed, 4242])
        self.split_at = split_at
        self.day_boundaries = set(day_boundaries or [])
        self.rated_set = set(int(i) for i in rated_items)
        if impressions is not None:
            # MIND. the actual impression list per round, item ids and clicks
            self.impressions = impressions
            self.T = min(T, len(impressions))
            self.mode = "impressions"
        else:
            self.mode = "pool"
            self.rated = np.asarray(rated_items)
            self.rew = np.asarray(rewards, float)
            self.T = int(T)
            self.K_cand = int(K_cand)
            n = len(self.rated)
            if split_at is not None:
                # protocol B. early-half items first, late-half items after the split round
                early = np.arange(split_at)
                late = np.arange(split_at, n)
                self.rng.shuffle(early)
                self.rng.shuffle(late)
                self.order = np.concatenate([early, late])
                self.T_switch = int(self.T // 2)
            else:
                self.order = self.rng.permutation(n)
                self.T_switch = None
            self.reset()

    def reset(self) -> None:
        self.shown: set = set()
        self._cur: Dict[int, np.ndarray] = {}

    def _pool_for(self, t: int) -> np.ndarray:
        if t in self._cur:
            return self._cur[t]
        if self.T_switch is not None:
            half = self.split_at
            pool = self.order[:half] if t < self.T_switch else self.order[half:]
        else:
            pool = self.order
        avail = [i for i in pool if i not in self.shown]
        if len(avail) < self.K_cand:
            avail = avail + [i for i in self.order if i not in self.shown and i not in avail]
        avail = np.asarray(avail[: self.K_cand])
        self._cur[t] = avail
        return avail

    def candidates(self, t: int) -> np.ndarray:
        if self.mode == "impressions":
            ids, _ = self.impressions[t]
            return self.feats[ids]
        return self.feats[self.rated[self._pool_for(t)]]

    def _cand_rewards(self, t: int) -> np.ndarray:
        if self.mode == "impressions":
            return self.impressions[t][1].astype(float)
        return self.rew[self._pool_for(t)]

    def reward(self, t: int, idx: int) -> float:
        if self.mode == "impressions":
            ids, clicks = self.impressions[t]
            item = int(ids[idx])
        else:
            pos = int(self._pool_for(t)[idx])
            item = int(self.rated[pos])
            self.shown.add(pos)
        if item not in self.rated_set:
            raise RuntimeError("reward requested for an item the user never rated/saw")
        return float(self._cand_rewards(t)[idx])

    def expected_reward(self, t: int, idx: int) -> float:
        return float(self._cand_rewards(t)[idx])

    def best(self, t: int) -> float:
        return float(self._cand_rewards(t).max())

    def oracle_idx(self, t: int) -> int:
        return int(np.argmax(self._cand_rewards(t)))

    def true_change(self, t: int) -> bool:
        if self.mode == "pool":
            return self.T_switch is not None and t == self.T_switch
        return t in self.day_boundaries


class ReplayEnv(Env):
    """Replay estimator (Li et al. 2011) for logs with a uniformly random logging policy.

    A round only counts when the logged action equals the agent's action. The reward is the logged
    reward. The metric is replay_reward, the only real-data metric that may be called unbiased.
    """

    regret_kind = "replay"

    def __init__(self, contexts: List[np.ndarray], logged_action: np.ndarray, logged_reward: np.ndarray,
                 D_prior=None):
        self.ctx, self.a, self.r = contexts, np.asarray(logged_action, int), np.asarray(logged_reward, float)
        self.T = len(contexts)
        self.d = contexts[0].shape[1]
        self.D_prior = D_prior

    def candidates(self, t):
        return self.ctx[t]

    def logged(self, t) -> Tuple[int, float]:
        return int(self.a[t]), float(self.r[t])

    def reward(self, t, idx):
        return float(self.r[t])  # only called when idx == logged action

    def best(self, t):
        return float("nan")  # no counterfactual oracle under replay

    def oracle_idx(self, t):
        return int(self.a[t])


# ----------------------------------------------------------------------------------------------
# runner
# ----------------------------------------------------------------------------------------------
LOG_COLS = ["seed", "user", "t", "agent", "reward", "oracle_or_best_available", "regret_kind",
            "regret", "source", "window", "drift_fired", "true_change", "forced"]
DIAG_COLS = ["e_P", "e_F", "e_C", "gain", "agreed", "pol_gap_F_minus_P"]


def _simulate_one(env: Env, agent: BaseAgent, seed: int, user: int, agent_name: str) -> dict:
    T = env.T
    reward = np.empty(T); best = np.empty(T); regret = np.empty(T)
    source = np.empty(T, dtype=object); window = np.empty(T, dtype=np.int32)
    fired = np.zeros(T, dtype=bool); tchange = np.zeros(T, dtype=bool)
    counted = np.ones(T, dtype=bool)
    forced = np.zeros(T, dtype=bool)
    diag = isinstance(agent, DESUCBAgent)
    extra = {c: np.full(T, np.nan) for c in DIAG_COLS} if diag else {}
    if isinstance(agent, OracleAgent) or (diag and isinstance(env, SyntheticEnv)):
        agent.env = env
    replay = isinstance(env, ReplayEnv)
    for t in range(T):
        X = env.candidates(t)
        idx = agent.step(t, X)
        if replay:
            a_log, r_log = env.logged(t)
            if idx != a_log:
                counted[t] = False
                reward[t] = np.nan; best[t] = np.nan; regret[t] = np.nan
                source[t] = agent.source; window[t] = agent.window
                continue
            r = r_log
            b = float("nan")
        else:
            r = env.reward(t, idx)
            b = env.best(t)
        agent.observe(t, X[idx], r)
        reward[t] = r; best[t] = b
        regret[t] = (b - env.expected_reward(t, idx)) if not replay else np.nan
        source[t] = agent.source; window[t] = agent.window
        fired[t] = bool(agent.drift_fired); tchange[t] = bool(env.true_change(t))
        forced[t] = bool(getattr(agent, "forced", False))
        if diag:
            for c, v in agent.extra.items():
                extra[c][t] = v
    df = pd.DataFrame({
        "seed": np.int32(seed), "user": np.int32(user), "t": np.arange(T, dtype=np.int32),
        "agent": agent_name, "reward": reward, "oracle_or_best_available": best,
        "regret_kind": env.regret_kind, "regret": regret, "source": source, "window": window,
        "drift_fired": fired, "true_change": tchange, "forced": forced,
    })
    for c in DIAG_COLS:
        df[c] = extra[c].astype(np.float32) if diag else np.float32(np.nan)
    if replay:
        df["counted"] = counted
    return df


_CTX: dict = {}


def _job(args):
    seed, user = args
    env_factory, agent_names, cfg = _CTX["env_factory"], _CTX["agents"], _CTX["cfg"]
    env = env_factory(seed, user)
    if env is None:
        return []
    out = []
    for name in agent_names:
        env.reset()
        agent = make_agent(name, env.d, env.D_prior, cfg, seed=seed * 100003 + user)
        out.append(_simulate_one(env, agent, seed, user, name))
    return out


def run(env_factory: Callable[[int, int], Env], agents: Sequence[str], cfg: dict,
        n_seeds: Optional[int] = None, n_users: Optional[int] = None, exp_name: Optional[str] = None,
        n_jobs: Optional[int] = None, progress: bool = True,
        seeds: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """Loop over seeds, users, agents and rounds. Return the per-round log and save results/<exp>.parquet.

    ``seeds`` gives an explicit seed list, for instance tuning seeds disjoint from the evaluation
    seeds. Otherwise the seeds are ``range(n_seeds)``."""
    n_seeds = int(cfg["n_seeds"] if n_seeds is None else n_seeds)
    n_users = int(cfg["n_users"] if n_users is None else n_users)
    n_jobs = cfg.get("n_jobs", -1) if n_jobs is None else n_jobs
    if n_jobs is None or n_jobs < 1:
        n_jobs = os.cpu_count() or 1
    seed_list = list(range(n_seeds)) if seeds is None else [int(s) for s in seeds]
    jobs = [(s, u) for s in seed_list for u in range(n_users)]
    _CTX.update(env_factory=env_factory, agents=list(agents), cfg=cfg)
    frames: List[pd.DataFrame] = []
    import multiprocessing as mp
    if "fork" not in mp.get_all_start_methods():
        n_jobs = 1  # closures are shared through fork, so without fork the loop runs serially
    if n_jobs == 1 or len(jobs) == 1:
        it = map(_job, jobs)
    else:
        pool = mp.get_context("fork").Pool(int(n_jobs))
        it = pool.imap_unordered(_job, jobs, chunksize=max(1, len(jobs) // (4 * int(n_jobs))))
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(it, total=len(jobs), desc=exp_name or "run", leave=False)
    for res in it:
        frames.extend(res)
    if n_jobs != 1 and len(jobs) != 1:
        pool.close(); pool.join()
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=LOG_COLS)
    df["agent"] = df["agent"].astype("category")
    df["source"] = df["source"].astype("category")
    df["regret_kind"] = df["regret_kind"].astype("category")
    if exp_name:
        RESULTS_DIR.mkdir(exist_ok=True)
        df.to_parquet(RESULTS_DIR / f"{exp_name}.parquet", index=False)
    return df


def set_dotted(cfg: dict, key: str, value) -> dict:
    """Return a deep copy of cfg with the dotted key set, for instance ``detector.lam_threshold``."""
    out = copy.deepcopy(cfg)
    node = out
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value
    return out


def tune_method(env_factory_of_cfg: Callable[[dict], Callable[[int, int], Env]], method: str, grid: Dict[str, list],
                cfg: dict, seeds: Sequence[int], n_users: int, progress: bool = False,
                regimes: Optional[Sequence[str]] = None) -> Tuple[dict, pd.DataFrame]:
    """Grid-search one method on the tuning seeds by mean dynamic regret. Lower is better.

    Every method goes through this same routine with its own grid (``tuning.grids`` in
    default.yaml). Ties go to the first grid entry. With several ``regimes`` (``drift.type``
    values) the criterion is the mean over regimes of the regret relative to the best grid point
    of that regime. A cheap regime is then not drowned by an expensive one. Returns the selected
    setting and the full tuning table. That table has one row per grid point, ``regret_<regime>``
    for each regime and the selection criterion as ``criterion``."""
    import itertools
    keys = list(grid)
    regs = list(regimes) if regimes else [str(cfg["drift"]["type"])]
    rows = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        row = {"method": method, **dict(zip(keys, combo))}
        for rg in regs:
            cfg_c = set_dotted(cfg, "drift.type", rg)
            for k, v in zip(keys, combo):
                cfg_c = set_dotted(cfg_c, k, v)
            df = run(env_factory_of_cfg(cfg_c), [method], cfg_c, n_users=n_users, seeds=seeds, exp_name=None,
                     progress=progress)
            reg = dynamic_regret(df)
            row[f"regret_{rg}"] = float(reg.loc[method, "dynamic_regret_mean"])
            row[f"regret_std_{rg}"] = float(reg.loc[method, "dynamic_regret_std"])
            row["n_units"] = int(reg.loc[method, "n_units"])
        rows.append(row)
    tab = pd.DataFrame(rows)
    rel = np.column_stack([tab[f"regret_{rg}"] / max(float(tab[f"regret_{rg}"].min()), 1e-12) for rg in regs])
    tab["criterion"] = rel.mean(axis=1)
    tab["regret_mean"] = tab[[f"regret_{rg}" for rg in regs]].mean(axis=1)
    best = tab.iloc[int(tab["criterion"].to_numpy().argmin())]
    return {k: best[k] for k in keys}, tab


# ----------------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------------
def _assert_kind(df: pd.DataFrame, kind: str) -> None:
    kinds = set(df["regret_kind"].astype(str).unique())
    if kinds != {kind}:
        raise ValueError(f"metric namespace mismatch: expected only {kind!r}, log contains {sorted(kinds)}")


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0,
                        alpha: float = 0.05, clusters: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
    """Paired bootstrap CI for mean(a - b) over matched units. Returns (mean, lo, hi).

    With ``clusters`` (one label per unit, usually the seed) the bootstrap is two-stage. Clusters
    are resampled with replacement, then units within each drawn cluster. Units of one seed share
    the item set and the cohort prototypes, so this is the right resampling scheme. The flat
    bootstrap treats all units as exchangeable."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    rng = np.random.default_rng(seed)
    n = len(diff)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    # TODO: BCa interval instead of percentile? percentile is what was preregistered, keep it for now
    if clusters is None:
        idx = rng.integers(n, size=(n_boot, n))
        boots = diff[idx].mean(axis=1)
    else:
        clusters = np.asarray(clusters)
        groups = [np.flatnonzero(clusters == c) for c in np.unique(clusters)]
        G = len(groups)
        boots = np.empty(n_boot)
        for bnum in range(n_boot):
            picked = rng.integers(G, size=G)
            vals = [diff[g[rng.integers(len(g), size=len(g))]] for g in (groups[k] for k in picked)]
            boots[bnum] = np.concatenate(vals).mean()
    return float(diff.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def seed_level_ci(a: np.ndarray, b: np.ndarray, seeds: np.ndarray) -> Tuple[float, float, float, int]:
    """Mean paired difference per seed, then a t-interval over the few independent seeds.
    Returns (mean, lo, hi, n_seeds)."""
    from scipy import stats
    d = pd.Series(np.asarray(a, float) - np.asarray(b, float)).groupby(np.asarray(seeds)).mean()
    k = len(d)
    if k < 2:
        return float(d.mean()), float("nan"), float("nan"), k
    se = float(d.std(ddof=1) / math.sqrt(k))
    q = float(stats.t.ppf(0.975, k - 1))
    return float(d.mean()), float(d.mean() - q * se), float(d.mean() + q * se), k


def per_unit(df: pd.DataFrame, value: str = "regret", agg: str = "sum") -> pd.DataFrame:
    """One aggregated value per (seed, user, agent). Units are matched across agents."""
    g = df.groupby(["seed", "user", "agent"], observed=True)[value].agg(agg).reset_index()
    return g.pivot_table(index=["seed", "user"], columns="agent", values=value, observed=True)


def dynamic_regret(df: pd.DataFrame) -> pd.DataFrame:
    """SYNTHETIC namespace. Cumulative dynamic regret per unit, mean +- std per agent."""
    _assert_kind(df, "dynamic")
    pu = per_unit(df, "regret", "sum")
    return pd.DataFrame({"dynamic_regret_mean": pu.mean(), "dynamic_regret_std": pu.std(ddof=1), "n_units": pu.count()})


def regret_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Per-round cumulative regret curve. Mean and std over units, per agent, in long format."""
    c = df.sort_values("t").copy()
    c["cum"] = c.groupby(["seed", "user", "agent"], observed=True)["regret"].cumsum()
    out = c.groupby(["agent", "t"], observed=True)["cum"].agg(["mean", "std"]).reset_index()
    return out


def recovery_time(df: pd.DataFrame, ma: int = 20, tol: float = 0.10) -> pd.Series:
    """SYNTHETIC. Rounds after the first true change until the ma-round moving-average per-round
    regret is back within (1+tol) times the pre-change moving average. Mean over units, per agent."""
    _assert_kind(df, "dynamic")
    res = {}
    for agent, g in df.groupby("agent", observed=True):
        times = []
        for (_, _), u in g.groupby(["seed", "user"], observed=True):
            u = u.sort_values("t")
            cps = u.loc[u["true_change"], "t"].to_numpy()
            if len(cps) == 0:
                continue
            cp = int(cps[0])
            r = u["regret"].to_numpy()
            if cp < ma + 1:
                continue
            pre = r[max(0, cp - 200):cp].mean()
            mavg = pd.Series(r).rolling(ma).mean().to_numpy()
            target = pre * (1 + tol) + 1e-9
            post = np.where(mavg[cp:] <= target)[0]
            times.append(int(post[0]) if len(post) else int(len(r) - cp))
        res[agent] = float(np.mean(times)) if times else float("nan")
    return pd.Series(res, name="recovery_time")


PRIOR_CARRYING = ("P", "C", "P+F")


def stale_evidence_ratio(df: pd.DataFrame) -> pd.Series:
    """Fraction of post-first-change rounds on which the agent conditioned on a prior-carrying source.

    P, the pooled C and the warm single-estimator label P+F all hold the prior rows."""
    res = {}
    for agent, g in df.groupby("agent", observed=True):
        vals = []
        for _, u in g.groupby(["seed", "user"], observed=True):
            cps = u.loc[u["true_change"], "t"].to_numpy()
            if len(cps) == 0:
                continue
            post = u[u["t"] >= cps[0]]
            vals.append(float(post["source"].astype(str).isin(PRIOR_CARRYING).mean()))
        res[agent] = float(np.mean(vals)) if vals else float("nan")
    return pd.Series(res, name="stale_evidence_ratio")


def stale_ratio_per_unit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (s, u, a), g in df.groupby(["seed", "user", "agent"], observed=True):
        cps = g.loc[g["true_change"], "t"].to_numpy()
        if len(cps) == 0:
            continue
        post = g[g["t"] >= cps[0]]
        stale = float(post["source"].astype(str).isin(PRIOR_CARRYING).mean())
        rows.append({"seed": s, "user": u, "agent": a, "stale": stale})
    r = pd.DataFrame(rows)
    return r.pivot_table(index=["seed", "user"], columns="agent", values="stale", observed=True)


def detection_stats(df: pd.DataFrame, fa_window: int = 200) -> pd.DataFrame:
    """Detection delay and false alarms per agent. Delay is the number of rounds from a true change
    to the first firing before the next change. A false alarm is a firing more than fa_window rounds
    after the most recent change."""
    rows = []
    for agent, g in df.groupby("agent", observed=True):
        delays, fas, missed, n_changes = [], 0, 0, 0
        for _, u in g.groupby(["seed", "user"], observed=True):
            u = u.sort_values("t")
            cps = list(u.loc[u["true_change"], "t"].to_numpy())
            fires = u.loc[u["drift_fired"], "t"].to_numpy()
            n_changes += len(cps)
            bounds = cps + [int(u["t"].max()) + 1]
            for i, cp in enumerate(cps):
                f = fires[(fires >= cp) & (fires < bounds[i + 1])]
                if len(f):
                    delays.append(int(f[0] - cp))
                else:
                    missed += 1
            for f in fires:
                prev = [c for c in cps if c <= f]
                if not prev or f - prev[-1] > fa_window:
                    fas += 1
        rows.append({"agent": agent, "n_changes": n_changes, "detected": len(delays),
                     "missed": missed, "mean_delay": float(np.mean(delays)) if delays else float("nan"),
                     "median_delay": float(np.median(delays)) if delays else float("nan"),
                     "false_alarms_per_unit": fas / max(1, g.groupby(["seed", "user"], observed=True).ngroups),
                     "total_firings": int(g["drift_fired"].sum())})
    return pd.DataFrame(rows).set_index("agent")


def detection_by_source(df: pd.DataFrame, fa_window: int = 200) -> pd.DataFrame:
    """Firings split by the evidence source active in the firing round.

    A change event is detected by at most one firing, the first one within ``fa_window`` rounds
    after the change. That event and its delay go to the source active at that firing. Later
    firings inside the window are ``repeats``. Firings outside every window are ``false_alarms``.
    ``round_share`` is the share of all rounds spent on that source, so false-alarm rates can be
    compared per round."""
    rows = []
    for agent, g in df.groupby("agent", observed=True):
        src = g["source"].astype(str)
        occupancy = src.value_counts(normalize=True)
        stats = {s: {"firings": 0, "delays": [], "repeats": 0, "fas": 0} for s in occupancy.index}
        for _, u in g.groupby(["seed", "user"], observed=True):
            u = u.sort_values("t")
            cps = u.loc[u["true_change"], "t"].to_numpy()
            fires = u[u["drift_fired"]]
            ft = fires["t"].to_numpy()
            fs = fires["source"].astype(str).to_numpy()
            bounds = np.append(cps, u["t"].max() + 1)
            credited = np.zeros(len(ft), dtype=bool)
            for i, cp in enumerate(cps):
                idx = np.where((ft >= cp) & (ft <= cp + fa_window) & (ft < bounds[i + 1]))[0]
                if len(idx):
                    credited[idx[0]] = True
                    stats[fs[idx[0]]]["delays"].append(int(ft[idx[0]] - cp))
            for j, f in enumerate(ft):
                stats[fs[j]]["firings"] += 1
                if credited[j]:
                    continue
                prev = cps[cps <= f]
                if len(prev) and f - prev[-1] <= fa_window:
                    stats[fs[j]]["repeats"] += 1
                else:
                    stats[fs[j]]["fas"] += 1
        for s in sorted(occupancy.index):
            st = stats[s]
            n_rounds = int((src == s).sum())
            rows.append({"agent": agent, "source": s, "round_share": float(occupancy[s]),
                         "firings": st["firings"], "true_detections": len(st["delays"]),
                         "repeats": st["repeats"], "false_alarms": st["fas"],
                         "mean_delay": float(np.mean(st["delays"])) if st["delays"] else float("nan"),
                         "false_alarms_per_1k_rounds": 1000.0 * st["fas"] / max(1, n_rounds)})
    return pd.DataFrame(rows).set_index(["agent", "source"])


def regret_decomposition(df: pd.DataFrame, B: int) -> pd.DataFrame:
    """SYNTHETIC. Cumulative dynamic regret of every agent split by phase and source.

    Phases are ``pre`` (before the first true change), ``post<=B`` (the first B rounds after a true
    change) and ``post>B`` (the rest). Sources are P, F and ``forced``. A forced round is one the
    switch did not choose, that is warm-up, burn-in or gated agreement. Values are mean regret per
    unit, so the cells of an agent sum to its dynamic regret."""
    _assert_kind(df, "dynamic")
    d = df[["seed", "user", "agent", "t", "regret", "source", "forced", "true_change"]].copy()
    d = d.sort_values(["agent", "seed", "user", "t"])
    key = ["agent", "seed", "user"]
    cp_t = d["t"].where(d["true_change"])
    last_cp = cp_t.groupby([d[k] for k in key], observed=True).ffill()
    since = d["t"] - last_cp
    phase = np.where(since.isna(), "pre", np.where(since < B, f"post<={B}", f"post>{B}"))
    d["phase"] = phase
    d["src"] = np.where(d["forced"], "forced", d["source"].astype(str))
    n_units = d.groupby("agent", observed=True).apply(lambda g: g.groupby(["seed", "user"], observed=True).ngroups)
    tot = d.groupby(["agent", "phase", "src"], observed=True)["regret"].sum().reset_index()
    tot["regret_per_unit"] = tot.apply(lambda r: r["regret"] / n_units[r["agent"]], axis=1)
    out = tot.pivot_table(index="agent", columns=["phase", "src"], values="regret_per_unit",
                          fill_value=0.0, observed=True)
    out["total"] = out.sum(axis=1)
    return out


def switch_occupancy(df: pd.DataFrame) -> pd.DataFrame:
    """Share of rounds per agent and source, forced rounds included. DES-type agents also get the
    share of gated agreement rounds."""
    d = df.copy()
    d["src"] = np.where(d["forced"], "forced", d["source"].astype(str))
    occ = d.groupby("agent", observed=True)["src"].value_counts(normalize=True).unstack(fill_value=0.0)
    if "agreed" in d.columns:
        occ["agreed_share"] = d.groupby("agent", observed=True)["agreed"].mean()
    return occ


def selection_effect(df: pd.DataFrame) -> pd.DataFrame:
    """Full-information check of the switch's credit signal. Synthetic only.

    Per agent and active source it reports the mean realised gain (the Eq. 11 estimand, a
    chosen-arm residual comparison), the mean full-information policy gap E[r(a_F)] - E[r(a_P)]
    signed towards the active source, the sign agreement between the two, and their Spearman
    correlation. A large difference between the P-active and F-active rows is the selection effect."""
    from scipy import stats
    rows = []
    d = df.dropna(subset=["gain", "pol_gap_F_minus_P"])
    d = d[~d["forced"]]
    for (agent, src), g in d.groupby(["agent", "source"], observed=True):
        if len(g) < 10:
            continue
        sign = 1.0 if src == "F" else -1.0
        gap = sign * g["pol_gap_F_minus_P"].to_numpy(float)  # >0 : active source's arm is better
        gain = g["gain"].to_numpy(float)
        nz = gap != 0
        rho = stats.spearmanr(gain[nz], gap[nz]).statistic if nz.sum() > 10 else float("nan")
        rows.append({"agent": agent, "active_source": src, "n_rounds": int(len(g)),
                     "mean_gain": float(gain.mean()), "P(gain>0)": float((gain > 0).mean()),
                     "mean_policy_gap_active_minus_other": float(gap.mean()),
                     "P(active arm better)": float((gap > 0).mean()),
                     "sign_agreement": (float((np.sign(gain[nz]) == np.sign(gap[nz])).mean())
                                        if nz.any() else float("nan")),
                     "spearman_rho": float(rho)})
    return pd.DataFrame(rows).set_index(["agent", "active_source"])


def cold_start_reward(df: pd.DataFrame, ts: Sequence[int] = (10, 20, 50)) -> pd.DataFrame:
    """Mean reward over the first t rounds, for t in ts. Any namespace, since reward is always observed."""
    out = {}
    for t in ts:
        sub = df[df["t"] < t]
        out[f"mean_reward@{t}"] = sub.groupby("agent", observed=True)["reward"].mean()
    return pd.DataFrame(out)


def loglog_slope(df: pd.DataFrame, checkpoints: Sequence[int]) -> pd.DataFrame:
    """Fit log(cum regret at T) ~ a + slope*log(T) over the checkpoints, per agent, on the mean over units."""
    _assert_kind(df, "dynamic")
    rows = []
    for agent, g in df.groupby("agent", observed=True):
        cum = g.sort_values("t").groupby(["seed", "user"], observed=True)["regret"].cumsum()
        g = g.assign(cum=cum)
        R = [g.loc[g["t"] == (T - 1), "cum"].mean() for T in checkpoints]
        R = np.maximum(np.asarray(R, float), 1e-9)
        slope, intercept = np.polyfit(np.log(checkpoints), np.log(R), 1)
        rows.append({"agent": agent, "slope": float(slope), **{f"R@{T}": float(r) for T, r in zip(checkpoints, R)}})
    return pd.DataFrame(rows).set_index("agent")


def cum_precision(df: pd.DataFrame, ks: Sequence[int] = (5, 10, 20, 40), hit_col: str = "reward") -> pd.DataFrame:
    """REAL namespace. cumulative precision@k = mean over units of (hits in first k rounds)/k."""
    _assert_kind(df, "offline")
    out = {}
    for k in ks:
        sub = df[df["t"] < k]
        pu = sub.groupby(["seed", "user", "agent"], observed=True)[hit_col].sum() / k
        out[f"cum_precision@{k}"] = pu.groupby("agent", observed=True).mean()
    return pd.DataFrame(out)


def cum_precision_per_unit(df: pd.DataFrame, k: int) -> pd.DataFrame:
    _assert_kind(df, "offline")
    sub = df[df["t"] < k]
    pu = (sub.groupby(["seed", "user", "agent"], observed=True)["reward"].sum() / k).reset_index()
    return pu.pivot_table(index=["seed", "user"], columns="agent", values="reward", observed=True)


def cum_recall(df: pd.DataFrame, n_relevant: pd.Series) -> pd.Series:
    """REAL. Hits over T divided by the user's number of relevant items (rating of 4 or more, or clicked).

    ``n_relevant`` must be indexed by the ordinal ``user`` column of the log (0..n_users-1), as
    given by ``RatedDataset.n_relevant_by_ordinal()``. Raw dataset ids are rejected."""
    _assert_kind(df, "offline")
    hits = df.groupby(["seed", "user", "agent"], observed=True)["reward"].sum().reset_index()
    hits["n_rel"] = hits["user"].map(n_relevant)
    if hits["n_rel"].isna().any():
        missing = sorted(hits.loc[hits["n_rel"].isna(), "user"].unique())[:5]
        raise KeyError(f"cum_recall: no n_relevant for log users {missing}; index it by ordinal user")
    hits["recall"] = hits["reward"] / hits["n_rel"].clip(lower=1)
    return hits.groupby("agent", observed=True)["recall"].mean().rename("cum_recall@T")


def offline_regret(df: pd.DataFrame) -> pd.DataFrame:
    """REAL. Cumulative offline regret against the best rated candidate. Never dynamic regret."""
    _assert_kind(df, "offline")
    pu = per_unit(df, "regret", "sum")
    return pd.DataFrame({"offline_regret_mean": pu.mean(), "offline_regret_std": pu.std(ddof=1), "n_units": pu.count()})


def replay_reward(df: pd.DataFrame) -> pd.DataFrame:
    """REPLAY namespace. Mean logged reward over the matched rounds only."""
    _assert_kind(df, "replay")
    sub = df[df["counted"]]
    g = sub.groupby("agent", observed=True)["reward"]
    return pd.DataFrame({"replay_reward": g.mean(), "matched_rounds": g.count()})


def summarize_pairs(pu: pd.DataFrame, pairs: Iterable[Tuple[str, str]], seed: int = 0) -> pd.DataFrame:
    """Paired 95% CIs of mean(A - B) for each (A, B) over matched units.

    Three intervals are reported. The flat unit bootstrap (``ci_lo``, ``ci_hi``), the seed-clustered
    two-stage bootstrap (``cl_lo``, ``cl_hi``) and a t-interval over per-seed means (``seed_lo``,
    ``seed_hi``). ``ci_excludes_0`` uses the clustered bootstrap, which is the registered decision
    rule. ``flat_excludes_0`` is kept for comparison."""
    rows = []
    for a, b in pairs:
        sub = pu[[a, b]].dropna()
        seeds = sub.index.get_level_values("seed").to_numpy()
        m, lo, hi = paired_bootstrap_ci(sub[a].to_numpy(), sub[b].to_numpy(), seed=seed)
        _, clo, chi = paired_bootstrap_ci(sub[a].to_numpy(), sub[b].to_numpy(), seed=seed, clusters=seeds)
        _, slo, shi, k = seed_level_ci(sub[a].to_numpy(), sub[b].to_numpy(), seeds)
        rows.append({"A": a, "B": b, "mean_diff(A-B)": m, "ci_lo": lo, "ci_hi": hi,
                     "cl_lo": clo, "cl_hi": chi, "seed_lo": slo, "seed_hi": shi,
                     "n_units": len(sub), "n_seeds": int(k),
                     "ci_excludes_0": bool(chi < 0 or clo > 0), "flat_excludes_0": bool(hi < 0 or lo > 0)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# claim registry
# ----------------------------------------------------------------------------------------------
class ClaimRegistry:
    """Registered claims. assert_claim never raises. It records SUPPORTED or NOT SUPPORTED."""

    def __init__(self, path: Path = RESULTS_DIR / "claims.json"):
        self.path = Path(path)
        self.claims: Dict[str, dict] = {}

    def assert_claim(self, name: str, ok: Optional[bool], stats: dict, description: str = "") -> bool:
        verdict = "NOT RUN" if ok is None else ("SUPPORTED" if ok else "NOT SUPPORTED")
        clean = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in stats.items()}
        self.claims[name] = {"claim": name, "description": description, "verdict": verdict, "stats": clean}
        print(f"[{verdict}] {name}: {description}")
        for k, v in clean.items():
            print(f"    {k} = {v}")
        return bool(ok)

    def save(self) -> Path:
        self.path.parent.mkdir(exist_ok=True)
        self.path.write_text(json.dumps(self.claims, indent=2, default=str))
        return self.path

    def verdict(self, name: str) -> Optional[bool]:
        c = self.claims.get(name)
        if c is None or c["verdict"] == "NOT RUN":
            return None
        return c["verdict"] == "SUPPORTED"


def decision_table(reg: ClaimRegistry, det_delay: float, B: int, des_vs_b4_ratio: Optional[float],
                   all_not_across_magnitudes: Optional[bool]) -> List[str]:
    """Decision table. Returns every branch that applies, which may be several."""
    out = []
    sup, mis, red = reg.verdict("suppression"), reg.verdict("suppression_mis"), reg.verdict("not_redundant")
    if det_delay is None or (isinstance(det_delay, float) and math.isnan(det_delay)) or det_delay > B:
        out.append("DETECTION FAILURE: tune PH threshold, retest")
    if sup is False and red is False:
        out.append("SUPPRESSION REDUNDANT: contribution is evidence switching; drop 'suppresses' from title")
    if sup is False and des_vs_b4_ratio is not None and des_vs_b4_ratio < 0.5:
        out.append("VALUE IS IN PRIOR: retitle to prior-warm-started non-stationary bandit")
    if sup is True and mis is False:
        out.append("MECHANISM HANDLES STALENESS NOT MISSPECIFICATION: matches title; state scope")
    if sup is False and mis is True:
        out.append("MECHANISM IS MISSPECIFIED-PRIOR DETECTION: retitle")
    if all_not_across_magnitudes:
        out.append("NO EMPIRICAL SUPPORT: report phase diagram only")
    if not out:
        out.append("NO DECISION-TABLE BRANCH TRIGGERED (claims consistent with the title as stated)")
    return out


# ----------------------------------------------------------------------------------------------
# real-data loaders. each caches parquet under data/ and prints a stats table
# ----------------------------------------------------------------------------------------------
GROUPLENS = "https://files.grouplens.org/datasets/movielens/"
MIND_MIRROR = "https://huggingface.co/datasets/huyva/MIND-small/resolve/main/train/"
MIND_OFFICIAL = "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip"


def download(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    """Download with requests over verified TLS. A certificate failure is an error. Place the file
    at ``dest`` by hand instead of disabling verification."""
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    # TODO: resume partial downloads (Range header). The MIND zip is 50 MB and fails on slow wifi.
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for c in r.iter_content(chunk):
                    f.write(c)
    except requests.exceptions.SSLError as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"TLS certificate verification failed for {url}; refusing an unverified download. "
                           f"Download it manually to {dest}.") from e
    tmp.rename(dest)
    return dest


def _stats_table(name: str, df: pd.DataFrame, user_col="user", item_col="item", time_col="ts") -> pd.DataFrame:
    t = pd.DataFrame([{
        "dataset": name, "#users": df[user_col].nunique(), "#items": df[item_col].nunique(),
        "#interactions": len(df),
        "span": (f"{pd.to_datetime(df[time_col].min(), unit='s').date()} .. "
                 f"{pd.to_datetime(df[time_col].max(), unit='s').date()}")
        if time_col in df else "n/a"}])
    print(t.to_markdown(index=False))
    return t


ML_GENRES = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama",
             "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]


def load_movielens(which: str = "ml-1m") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (ratings[user,item,rating,ts], users[user,gender,age,occupation], movies[item,genres])."""
    cache = DATA_DIR / f"{which}.parquet"
    ucache = DATA_DIR / f"{which}_users.parquet"
    mcache = DATA_DIR / f"{which}_movies.parquet"
    if cache.exists() and ucache.exists() and mcache.exists():
        return pd.read_parquet(cache), pd.read_parquet(ucache), pd.read_parquet(mcache)
    z = download(GROUPLENS + f"{which}.zip", DATA_DIR / f"{which}.zip")
    zf = zipfile.ZipFile(z)
    if which == "ml-1m":
        rat = pd.read_csv(zf.open("ml-1m/ratings.dat"), sep="::", engine="python",
                          names=["user", "item", "rating", "ts"])
        usr = pd.read_csv(zf.open("ml-1m/users.dat"), sep="::", engine="python",
                          names=["user", "gender", "age", "occupation", "zip"])
        mov = pd.read_csv(zf.open("ml-1m/movies.dat"), sep="::", engine="python", encoding="latin-1",
                          names=["item", "title", "genres"])
    elif which == "ml-100k":
        rat = pd.read_csv(zf.open("ml-100k/u.data"), sep="\t", names=["user", "item", "rating", "ts"])
        usr = pd.read_csv(zf.open("ml-100k/u.user"), sep="|", names=["user", "age", "gender", "occupation", "zip"])
        cols = ["item", "title", "release", "video", "url", "unknown"] + ML_GENRES
        m = pd.read_csv(zf.open("ml-100k/u.item"), sep="|", encoding="latin-1", names=cols)
        m["genres"] = m[ML_GENRES].apply(lambda r: "|".join(g for g in ML_GENRES if r[g] == 1), axis=1)
        mov = m[["item", "title", "genres"]]
    else:
        raise ValueError(which)
    DATA_DIR.mkdir(exist_ok=True)
    rat.to_parquet(cache, index=False); usr.to_parquet(ucache, index=False); mov.to_parquet(mcache, index=False)
    return rat, usr, mov


@dataclass
class RatedDataset:
    """A prepared real-data set. Item features, per-user rated streams and cohort priors."""

    name: str
    feats: np.ndarray                     # (n_items_internal, d)
    item_index: Dict[int, int]            # raw item id -> row of feats
    test_users: List[int]
    user_rows: Dict[int, pd.DataFrame]    # user -> chronological rows [item, reward, ts, ...]
    cohort_of: Dict[int, str]
    cohort_rows: Dict[str, Tuple[np.ndarray, np.ndarray]]  # cohort -> (X, r) from TRAIN users
    stats: pd.DataFrame
    d: int = 0
    n_relevant: Dict[int, int] = field(default_factory=dict)  # keyed by RAW dataset user id

    def n_relevant_by_ordinal(self) -> pd.Series:
        """Recall denominators keyed the way the runner logs users, by ordinal index into test_users."""
        return pd.Series({i: self.n_relevant.get(u, 0) for i, u in enumerate(self.test_users)}, dtype=int)

    def prior_for(self, user: int, n_rows: int, rng: np.random.Generator, mismatch: bool = False):
        coh = self.cohort_of[user]
        keys = [k for k in self.cohort_rows if k != coh] if mismatch else [k for k in self.cohort_rows if k == coh]
        if not keys:
            keys = list(self.cohort_rows)
        key = keys[int(rng.integers(len(keys)))]
        X, r = self.cohort_rows[key]
        idx = rng.integers(len(r), size=min(n_rows, len(r)))
        return (X[idx], r[idx]), key


def prepare_movielens(cfg_ds: dict, which: str = "ml-1m", seed: int = 0) -> RatedDataset:
    """MovieLens features. 18 genre one-hots plus a (d-18)-dim SVD item embedding fit on the
    train-period ratings of train users (user-disjoint 85/5/10 split). Users need at least
    min_ratings ratings. Item factors are frozen."""
    from sklearn.decomposition import TruncatedSVD
    from scipy.sparse import csr_matrix

    rat, usr, mov = load_movielens(which)
    stats = _stats_table(which, rat)
    cnt = rat.groupby("user").size()
    keep = cnt[cnt >= int(cfg_ds["min_ratings"])].index.to_numpy()
    rng = np.random.default_rng(seed)
    keep = rng.permutation(keep)
    n = len(keep)
    f_tr, f_va, _ = cfg_ds["split"]
    n_tr, n_va = int(f_tr * n), int(f_va * n)
    train_u, test_u = keep[:n_tr], keep[n_tr + n_va:]  # keep[n_tr:n_tr + n_va] is the held-out val split
    rat = rat[rat["user"].isin(keep)].copy()
    items = np.sort(rat["item"].unique())
    item_index = {int(i): k for k, i in enumerate(items)}
    # genre one-hot
    genre = np.zeros((len(items), len(ML_GENRES)))
    gmap = mov.set_index("item")["genres"].to_dict()
    for i, k in item_index.items():
        for g in str(gmap.get(i, "")).split("|"):
            if g in ML_GENRES:
                genre[k, ML_GENRES.index(g)] = 1.0
    # SVD embedding fit on train users only, using all of their ratings
    tr = rat[rat["user"].isin(train_u)]
    uidx = {u: k for k, u in enumerate(np.sort(tr["user"].unique()))}
    M = csr_matrix((tr["rating"].to_numpy(float) - 3.0,
                    (tr["user"].map(uidx).to_numpy(), tr["item"].map(item_index).to_numpy())),
                   shape=(len(uidx), len(items)))
    svd_dim = int(cfg_ds["svd_dim"])
    svd = TruncatedSVD(n_components=svd_dim, random_state=seed).fit(M)
    emb = svd.components_.T  # (n_items, svd_dim)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    feats = np.hstack([genre / np.sqrt(np.maximum(genre.sum(1, keepdims=True), 1)), emb])
    d = feats.shape[1]
    assert d == int(cfg_ds["d"]), (d, cfg_ds["d"])
    thr = int(cfg_ds["hit_threshold"])
    rat["reward"] = (rat["rating"] >= thr).astype(float)
    # demographic cohorts and cohort prior rows from train users
    usr = usr.set_index("user")
    ck = cfg_ds["cohort_keys"]
    cohort_of = {int(u): "|".join(str(usr.loc[u, k]) for k in ck) for u in keep}
    cohort_rows: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    tr = tr.assign(reward=(tr["rating"] >= thr).astype(float), coh=tr["user"].map(cohort_of))
    for coh, g in tr.groupby("coh"):
        X = feats[g["item"].map(item_index).to_numpy()]
        cohort_rows[coh] = (X, g["reward"].to_numpy(float))
    max_test = int(cfg_ds.get("max_test_users", len(test_u)))
    test_users = [int(u) for u in test_u[:max_test]]
    user_rows = {u: rat[rat["user"] == u].sort_values("ts").reset_index(drop=True) for u in test_users}
    n_rel = {u: int(user_rows[u]["reward"].sum()) for u in test_users}
    return RatedDataset(which, feats, item_index, test_users, user_rows, cohort_of, cohort_rows, stats, d, n_rel)


def movielens_env_factory(ds: RatedDataset, cfg: dict, protocol: str) -> Callable[[int, int], Optional[Env]]:
    """Protocol A is cold start. Protocol B is induced drift, a chronological half-split with the
    prior from the user's own early half. Protocol C is a stale prior from a different demographic
    cohort with probability p_mismatch."""
    T, K = int(cfg["T"]), int(cfg["K_cand"])
    n_rows = int(cfg["n_prior_rows"])
    p_mis = float(cfg.get("protocol_c_p_mismatch", 0.5)) if protocol == "C" else 0.0

    def factory(seed: int, user_i: int) -> Optional[Env]:
        if user_i >= len(ds.test_users):
            return None
        u = ds.test_users[user_i]
        rows = ds.user_rows[u]
        rng = np.random.default_rng([seed, u, 31])
        items = rows["item"].map(ds.item_index).to_numpy()
        rew = rows["reward"].to_numpy(float)
        mismatch = bool(rng.random() < p_mis)
        D_prior, _ = ds.prior_for(u, n_rows, rng, mismatch=mismatch)
        split_at = None
        if protocol == "B":
            split_at = len(rows) // 2
            # stale prior. the user's own early-half rows plus the cohort rows, built before the split
            Xe, re = ds.feats[items[:split_at]], rew[:split_at]
            D_prior = (np.vstack([D_prior[0], Xe]), np.concatenate([D_prior[1], re]))
        return RatedItemEnv(ds.feats, items, rew, T, K, D_prior, seed=seed * 7919 + u, split_at=split_at)

    return factory


def load_mind_small(cfg_ds: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """MIND-small train split. The official Azure blob (MIND_OFFICIAL) refuses anonymous access
    with HTTP 409, so the same raw TSVs are fetched from a public Hugging Face mirror."""
    bcache, ncache = DATA_DIR / "mind_behaviors.parquet", DATA_DIR / "mind_news.parquet"
    if bcache.exists() and ncache.exists():
        return pd.read_parquet(bcache), pd.read_parquet(ncache)
    DATA_DIR.mkdir(exist_ok=True)
    try:
        z = download(MIND_OFFICIAL, DATA_DIR / "MINDsmall_train.zip")
        zf = zipfile.ZipFile(z)
        beh = pd.read_csv(zf.open("behaviors.tsv"), sep="\t", header=None,
                          names=["imp", "user", "time", "history", "impressions"])
        news = pd.read_csv(zf.open("news.tsv"), sep="\t", header=None,
                           names=["item", "category", "subcategory", "title", "abstract", "url", "te", "ae"])
    except Exception as e:  # noqa: BLE001
        print(f"official MIND download failed ({type(e).__name__}); using Hugging Face mirror")
        b = download(MIND_MIRROR + "behaviors.tsv", DATA_DIR / "mind_behaviors.tsv")
        nw = download(MIND_MIRROR + "news.tsv", DATA_DIR / "mind_news.tsv")
        beh = pd.read_csv(b, sep="\t", header=None, names=["imp", "user", "time", "history", "impressions"],
                          quoting=3)
        news = pd.read_csv(nw, sep="\t", header=None, quoting=3,
                           names=["item", "category", "subcategory", "title", "abstract", "url", "te", "ae"])
    beh["time"] = pd.to_datetime(beh["time"], format="%m/%d/%Y %I:%M:%S %p")
    beh = beh[["imp", "user", "time", "impressions"]]
    news = news[["item", "category", "subcategory", "title"]]
    beh.to_parquet(bcache, index=False); news.to_parquet(ncache, index=False)
    return beh, news


def prepare_mind(cfg_ds: dict, seed: int = 0) -> RatedDataset:
    """MIND features. TF-IDF on titles reduced by SVD to svd_dim, plus category one-hots. Reward is
    a click. Candidates are the actual impression list, chronological per user. The cohort is the
    most-clicked category among the user's train-period impressions, since MIND has no demographics."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    beh, news = load_mind_small(cfg_ds)
    beh["ts"] = (beh["time"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
    items = news["item"].to_numpy()
    item_index = {str(i): k for k, i in enumerate(items)}
    tfidf = TfidfVectorizer(max_features=20000, stop_words="english").fit_transform(news["title"].fillna(""))
    svd = TruncatedSVD(n_components=int(cfg_ds["svd_dim"]), random_state=seed).fit_transform(tfidf)
    svd = svd / (np.linalg.norm(svd, axis=1, keepdims=True) + 1e-9)
    cats = sorted(news["category"].unique())
    onehot = np.zeros((len(news), len(cats)))
    onehot[np.arange(len(news)), news["category"].map({c: i for i, c in enumerate(cats)}).to_numpy()] = 1.0
    feats = np.hstack([svd, onehot])
    d = feats.shape[1]
    # parse impressions
    parsed = []
    for imp, u, ts, s in zip(beh["imp"], beh["user"], beh["ts"], beh["impressions"]):
        pairs = [p.rsplit("-", 1) for p in str(s).split()]
        ids = np.array([item_index[p[0]] for p in pairs if p[0] in item_index])
        clk = np.array([int(p[1]) for p in pairs if p[0] in item_index])
        if len(ids) >= 2:
            parsed.append((u, ts, ids, clk))
    P = pd.DataFrame(parsed, columns=["user", "ts", "ids", "clicks"])
    P = P.sort_values(["user", "ts"])
    flat = pd.DataFrame({"user": P["user"], "item": [int(i[0]) for i in P["ids"]], "ts": P["ts"]})
    stats = _stats_table("MIND-small (impressions)", flat)
    cnt = P.groupby("user").size()
    eligible = cnt[cnt >= int(cfg_ds["min_impressions"])].index.to_numpy()
    rng = np.random.default_rng(seed)
    eligible = rng.permutation(eligible)
    n_test = min(int(cfg_ds["max_test_users"]), len(eligible) // 2)
    test_users, train_users = list(eligible[:n_test]), eligible[n_test:]
    train_set = set(train_users)
    # cohort by dominant clicked category, cohort prior rows from train users' impressions
    cat_of_item = news["category"].to_numpy()
    cohort_of: Dict[str, str] = {}
    coh_X: Dict[str, List[np.ndarray]] = {}
    coh_r: Dict[str, List[np.ndarray]] = {}
    user_rows: Dict[int, pd.DataFrame] = {}
    for u, g in P.groupby("user", sort=False):
        ids = np.concatenate(g["ids"].to_list()); clk = np.concatenate(g["clicks"].to_list())
        clicked = ids[clk == 1]
        coh = pd.Series(cat_of_item[clicked]).mode().iloc[0] if len(clicked) else "none"
        cohort_of[u] = coh
        if u in train_set:
            coh_X.setdefault(coh, []).append(feats[ids]); coh_r.setdefault(coh, []).append(clk.astype(float))
    cohort_rows = {c: (np.vstack(coh_X[c]), np.concatenate(coh_r[c])) for c in coh_X}
    for u in test_users:
        g = P[P["user"] == u]
        user_rows[u] = g.reset_index(drop=True)
    ds = RatedDataset("mind_small", feats, {int(k): v for k, v in enumerate(items)}, test_users, user_rows,
                      cohort_of, cohort_rows, stats, d)
    return ds


def mind_env_factory(ds: RatedDataset, cfg: dict) -> Callable[[int, int], Optional[Env]]:
    T, n_rows = int(cfg["T"]), int(cfg["n_prior_rows"])

    def factory(seed: int, user_i: int) -> Optional[Env]:
        if user_i >= len(ds.test_users):
            return None
        u = ds.test_users[user_i]
        g = ds.user_rows[u]
        rng = np.random.default_rng([seed, user_i, 55])
        D_prior, _ = ds.prior_for(u, n_rows, rng)
        impressions = list(zip(g["ids"].to_list(), g["clicks"].to_list()))
        days = pd.to_datetime(g["ts"], unit="s").dt.floor("D").to_numpy()
        day_bounds = [t for t in range(1, len(days)) if days[t] != days[t - 1]]
        rated = np.unique(np.concatenate(g["ids"].to_list()))
        return RatedItemEnv(ds.feats, rated, None, T, 0, D_prior, seed=seed * 7919 + user_i,
                            impressions=impressions, day_boundaries=day_bounds)

    return factory


def load_yahoo_r6(path: str, max_rounds: int = 200000) -> ReplayEnv:
    """Yahoo! R6A format, `ts displayed click |user f... |art1 f... |art2 ...`.
    Uniform-random logging, so the replay estimator applies and the metric is replay_reward."""
    ctxs, acts, rews = [], [], []
    opener = open
    if str(path).endswith(".gz"):
        import gzip
        opener = gzip.open
    with opener(path, "rt") as f:
        for line in f:
            parts = line.strip().split("|")
            head = parts[0].split()
            if len(head) < 3:
                continue
            shown, click = head[1], float(head[2])
            arts, feats = [], []
            for blk in parts[2:]:
                tok = blk.split()
                if not tok:
                    continue
                arts.append(tok[0])
                v = np.zeros(6)
                for kv in tok[1:]:
                    k, val = kv.split(":")
                    v[int(k) - 1] = float(val)
                feats.append(v)
            if shown not in arts:
                continue
            ctxs.append(np.array(feats)); acts.append(arts.index(shown)); rews.append(click)
            if len(ctxs) >= max_rounds:
                break
    return ReplayEnv(ctxs, np.array(acts), np.array(rews))


KUAIRAND_PURE_URL = "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz?download=1"
KUAIRAND_PURE_SHA256 = "c814bf6f3624c0cfae83c57de3df26b2ed206e5c57bab4c4dcbfabbabe20cbf0"


def fetch_kuairand_pure(root: Path = DATA_DIR / "kuairand") -> Path:
    """Download and extract KuaiRand-Pure (Zenodo record 10439422, CC BY-SA 4.0, about 47 MB).

    Returns the ``KuaiRand-Pure/data`` directory. The archive checksum is verified before
    extraction. Only regular files below ``KuaiRand-Pure/`` are extracted."""
    import hashlib
    import tarfile
    root = Path(root)
    data_dir = root / "KuaiRand-Pure" / "data"
    if (data_dir / "log_random_4_22_to_5_08_pure.csv").exists():
        return data_dir
    tgz = download(KUAIRAND_PURE_URL, root / "KuaiRand-Pure.tar.gz")
    h = hashlib.sha256()
    with open(tgz, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != KUAIRAND_PURE_SHA256:
        tgz.unlink()
        raise RuntimeError("KuaiRand-Pure.tar.gz checksum mismatch; partial file removed, re-run to download again")
    with tarfile.open(tgz, "r:gz") as tf:
        members = [m for m in tf.getmembers()
                   if m.isfile() and m.name.startswith("KuaiRand-Pure/") and ".." not in m.name]
        tf.extractall(root, members=members)
    return data_dir


def _kuairand_video_features(data_dir: Path, d: int) -> Dict[int, np.ndarray]:
    """Deterministic video embedding. One-hots of video_type, upload_type, music_type and tags plus
    standardised log1p exposure statistics, reduced by SVD to ``d`` and unit-normalised."""
    basic = pd.read_csv(data_dir / "video_features_basic_pure.csv",
                        usecols=["video_id", "video_type", "upload_type", "music_type", "tag"])
    stat = pd.read_csv(data_dir / "video_features_statistic_pure.csv")
    basic = basic.merge(stat, on="video_id", how="left").fillna(0)
    cat = pd.get_dummies(basic[["video_type", "upload_type", "music_type"]].astype(str), dtype=float)
    tags = basic["tag"].astype(str).str.split(",")
    tag_dum = pd.get_dummies(tags.explode()).groupby(level=0).max().astype(float)
    num = stat.columns.drop("video_id")
    Z = np.log1p(np.clip(basic[num].to_numpy(float), 0, None))
    Z = (Z - Z.mean(0)) / np.maximum(Z.std(0), 1e-9)
    M = np.hstack([cat.to_numpy(), tag_dum.reindex(basic.index).fillna(0).to_numpy(), Z])
    M = M - M.mean(0)
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    E = _unit(U[:, :d] * S[:d])
    return {int(v): E[i] for i, v in enumerate(basic["video_id"])}


def load_kuairand_random(path: Optional[str] = None, d: int = 16, max_rounds: int = 200000, seed: int = 0,
                         K: int = 10, n_prior_rows: int = 200) -> ReplayEnv:
    """KuaiRand-Pure random-exposure slice as a replay environment.

    Rounds are (user, date) groups of the random log ``log_random_4_22_to_5_08_pure.csv``. Every
    exposure in that log was drawn uniformly at random, so any row of a group is a uniform draw from
    the group. The first ``K`` exposures of a group in time order form the candidate set. The logged
    action is a uniform row of that set and its reward is ``is_click``. Rounds are ordered by time.
    ``D_prior`` is ``n_prior_rows`` rows of the earlier standard-policy log (04/08 to 04/21). That is
    a population prior which precedes the evaluation period. ``path`` is the ``KuaiRand-Pure/data``
    directory, or the random-log CSV inside it. When None the archive is fetched from Zenodo."""
    data_dir = Path(path) if path else fetch_kuairand_pure()
    if data_dir.is_file():
        data_dir = data_dir.parent
    emb = _kuairand_video_features(data_dir, d)
    cols = ["user_id", "video_id", "time_ms", "is_click", "date"]
    df = pd.read_csv(data_dir / "log_random_4_22_to_5_08_pure.csv", usecols=cols)
    df = df[df["video_id"].isin(emb.keys())].sort_values("time_ms").copy()
    rng = np.random.default_rng(seed)
    ctxs, acts, rews = [], [], []
    df["rank"] = df.groupby(["user_id", "date"], sort=False).cumcount()
    df = df[df["rank"] < K].copy()
    df["gid"] = df.groupby(["user_id", "date"], sort=False)["time_ms"].transform("min")
    df = df.sort_values(["gid", "user_id", "date", "time_ms"])
    keys = df["user_id"].to_numpy() * 100000000 + df["date"].to_numpy()
    bounds = np.flatnonzero(np.diff(keys, prepend=keys[0] - 1, append=keys[-1] + 1))
    vids = df["video_id"].to_numpy()
    clicks = df["is_click"].to_numpy(float)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi - lo < 2:
            continue
        X = np.stack([emb[int(v)] for v in vids[lo:hi]])
        a = int(rng.integers(hi - lo))
        ctxs.append(X)
        acts.append(a)
        rews.append(float(clicks[lo + a]))
        if len(ctxs) >= max_rounds:
            break
    std = pd.read_csv(data_dir / "log_standard_4_08_to_4_21_pure.csv", usecols=["video_id", "is_click"])
    std = std[std["video_id"].isin(emb.keys())]
    pick = std.iloc[rng.choice(len(std), size=min(n_prior_rows, len(std)), replace=False)]
    Xp = np.stack([emb[int(v)] for v in pick["video_id"]])
    yp = pick["is_click"].to_numpy(float)
    return ReplayEnv(ctxs, np.array(acts), np.array(rews), D_prior=(Xp, yp))


def split_replay(env: ReplayEnv, n_units: int, T: Optional[int] = None) -> List[ReplayEnv]:
    """Cut a replay stream into ``n_units`` consecutive segments of ``T`` rounds. The default is
    equal shares. This gives a paired, seed-clustered CI over segments. All segments share the
    stream's ``D_prior``."""
    if T is None:
        T = env.T // n_units
    segs = []
    for u in range(n_units):
        lo, hi = u * T, (u + 1) * T
        if hi > env.T:
            break
        segs.append(ReplayEnv(env.ctx[lo:hi], env.a[lo:hi], env.r[lo:hi], D_prior=env.D_prior))
    return segs


def load_netflix(path: str, cfg_ds: dict, seed: int = 0) -> pd.DataFrame:
    """Parse Netflix Prize combined_data_*.txt from a local directory. The data cannot be
    downloaded anonymously. Returns ratings[user,item,rating,ts] for a user sample."""
    rows = []
    for fn in sorted(Path(path).glob("combined_data_*.txt")):
        item = None
        with open(fn) as f:
            for line in f:
                if line.endswith(":\n"):
                    item = int(line[:-2])
                else:
                    u, r, dt = line.strip().split(",")
                    rows.append((int(u), item, int(r), dt))
    df = pd.DataFrame(rows, columns=["user", "item", "rating", "date"])
    df["ts"] = pd.to_datetime(df["date"]).astype("int64") // 10**9
    rng = np.random.default_rng(seed)
    users = rng.choice(df["user"].unique(), size=min(int(cfg_ds["sample_users"]), df["user"].nunique()), replace=False)
    return df[df["user"].isin(users)].drop(columns="date")


def amazon_cross_category_prior(cfg_ds: dict, d: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Amazon Reviews 2023 ratings (McAuley lab), used only to build a cross-category D_prior with
    hashed-id features. Review time is not purchase time, so this is never used for drift."""
    warnings.warn("Amazon Reviews 2023: review time != purchase time; this data is used only for D_prior "
                  "and never for drift claims.")
    base = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/0core/rating_only/"
    rng = np.random.default_rng(seed)
    Xs, rs = [], []
    for cat in cfg_ds["categories"]:
        p = download(base + f"{cat}.csv.gz", DATA_DIR / f"amazon_{cat}.csv.gz")
        df = pd.read_csv(p, nrows=200000)
        X = np.stack([_unit(np.random.default_rng(zlib.crc32(str(a).encode())).standard_normal(d))
                      for a in df["parent_asin"]])
        Xs.append(X); rs.append((df["rating"].to_numpy(float) >= 4).astype(float))
    X, r = np.vstack(Xs), np.concatenate(rs)
    idx = rng.integers(len(r), size=int(cfg_ds["n_prior_rows"]))
    return X[idx], r[idx]
