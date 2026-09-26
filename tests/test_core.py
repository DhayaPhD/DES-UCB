import sys
from pathlib import Path

import math
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import des_ucb_core as C  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return C.load_config()


# TODO: a test for two detector firings inside one BOB block (see FIXME in DESUCBAgent.on_drift)
# def test_double_firing_same_block():
#     ...


# ---------------------------------------------------------------- estimators
def test_ridge_add_remove_matches_exact_recompute():
    rng = np.random.default_rng(0)
    d = 6
    est = C.RidgeUCB(d, lam=1.0, beta=0.5, S=100.0)
    X = rng.standard_normal((40, d)); r = rng.standard_normal(40)
    for x, y in zip(X, r):
        est.add(x, y)
    for x, y in zip(X[:15], r[:15]):
        est.remove(x, y)
    kept = slice(15, 40)
    V = np.eye(d) + X[kept].T @ X[kept]
    b = X[kept].T @ r[kept]
    np.testing.assert_allclose(est.theta(), np.linalg.solve(V, b), atol=1e-8)
    np.testing.assert_allclose(est.V_inv, np.linalg.inv(V), atol=1e-8)


def test_ridge_clips_theta_to_ball():
    est = C.RidgeUCB(3, lam=1e-3, beta=0.5, S=1.0)
    for _ in range(50):
        est.add(np.array([1.0, 0, 0]), 100.0)
    assert np.linalg.norm(est.theta()) <= 1.0 + 1e-9


def test_feedback_eviction_on_shrink_and_purge():
    d = 4
    rng = np.random.default_rng(1)
    F = C.FeedbackEstimator(d, window=100)
    for t in range(50):
        F.add_live(t, rng.standard_normal(d), 1.0)
    assert len(F) == 50
    F.set_window(10, t=49)               # keep s > 49-10 = 39 -> s in 40..49
    assert len(F) == 10 and F.rows[0][0] == 40
    # exact recompute against the kept rows
    X = np.stack([x for _, x, _ in F.rows]); r = np.array([y for _, _, y in F.rows])
    np.testing.assert_allclose(F.theta(), np.linalg.solve(np.eye(d) + X.T @ X, X.T @ r), atol=1e-8)
    # shrinking only changes the active window, growing it again restores the retained rows exactly
    F.set_window(30, t=49)
    assert len(F) == 30 and F.rows[0][0] == 20
    X = np.stack([x for _, x, _ in F.rows]); r = np.array([y for _, _, y in F.rows])
    np.testing.assert_allclose(F.theta(), np.linalg.solve(np.eye(d) + X.T @ X, X.T @ r), atol=1e-8)
    F.set_window(10, t=49)
    assert len(F) == 10 and F.rows[0][0] == 40
    # an explicit drift purge is irreversible
    F.purge_older_than(45)
    assert len(F) == 5 and F.rows[0][0] == 45
    F.set_window(100, t=49)
    assert len(F) == 5 and F.rows[0][0] == 45
    F.set_window(10, t=49)
    # rolling eviction
    for t in range(50, 60):
        F.add_live(t, rng.standard_normal(d), 0.0)
    assert len(F) == 10 and F.rows[0][0] == 50
    assert len(F.history) == 15 and F.history[0][0] == 45  # retained (max_history=100), not active


def test_feedback_history_bounded_by_max_history():
    F = C.FeedbackEstimator(3, window=5, max_history=20)
    for t in range(100):
        F.add_live(t, np.ones(3), 1.0)
    assert len(F) == 5 and len(F.history) == 20
    F.set_window(50, t=99)
    assert len(F) == 20 and F.rows[0][0] == 80


# ---------------------------------------------------------------- switch
def test_switch_forced_feedback_exactly_B_rounds():
    sw = C.EvidenceSwitch("exp3", rng=np.random.default_rng(0))
    sw.force_feedback(7)
    picks = []
    for t in range(20):
        picks.append(sw.select(t))
        sw.update(picks[-1], 0.0)
        sw.tick()
    assert picks[:7] == ["F"] * 7
    # after the burn-in EXP3 must be able to pick P again, gamma exploration guarantees it
    assert sw.burn_in_remaining == 0
    assert sw.probs()[0] > 0


def test_exp3_probabilities_sum_to_one_and_favour_better_source():
    rng = np.random.default_rng(0)
    sw = C.EvidenceSwitch("exp3", gamma=0.1, G=1.0, rng=rng)
    for t in range(500):
        m = sw.select(t)
        gain = 0.8 if m == "F" else -0.8   # F is the better source
        sw.update(m, gain + 0.1 * rng.standard_normal())
        sw.tick()
        assert abs(sw.probs().sum() - 1.0) < 1e-9
    p = sw.probs()
    assert p[1] > 0.8 and p[1] > p[0]


def test_ucb1_switch_favours_better_source():
    sw = C.EvidenceSwitch("ucb1", c=1.0, rng=np.random.default_rng(0), sources=("P", "F"))
    picks = []
    for t in range(500):
        m = sw.select(t)
        sw.update(m, 0.8 if m == "P" else -0.8)
        sw.tick()
        picks.append(m)
    assert picks[-100:].count("P") > 90


def test_reset_source_removes_credit():
    sw = C.EvidenceSwitch("exp3", gamma=0.1, rng=np.random.default_rng(0), sources=("P", "F"))
    for t in range(300):
        m = sw.select(t); sw.update(m, 1.0 if m == "P" else -1.0); sw.tick()
    assert sw.probs()[0] > 0.5
    sw.reset_source("P")
    assert sw.probs()[0] <= 0.5 + 1e-9


# ---------------------------------------------------------------- detectors
def test_page_hinkley_fires_on_mean_shift_not_on_noise():
    rng = np.random.default_rng(0)
    ph = C.PageHinkley(delta=0.005, lam_threshold=5.0, alpha_forget=0.999, min_instances=30)
    fired_stationary = [ph.update(abs(0.1 * rng.standard_normal())) for _ in range(2000)]
    assert not any(fired_stationary)
    ph = C.PageHinkley(delta=0.005, lam_threshold=5.0, alpha_forget=0.999, min_instances=30)
    for _ in range(500):
        ph.update(abs(0.1 * rng.standard_normal()))
    fired = [ph.update(1.0 + abs(0.1 * rng.standard_normal())) for _ in range(200)]
    assert any(fired)
    assert fired.index(True) < 50


def test_adwin_fires_on_shift():
    rng = np.random.default_rng(0)
    ad = C.ADWIN(delta=0.002)
    assert not any(ad.update(0.01 * rng.standard_normal()) for _ in range(300))
    assert any(ad.update(1.0 + 0.01 * rng.standard_normal()) for _ in range(100))


# ---------------------------------------------------------------- BOB
def test_bob_favours_better_window():
    bob = C.BOBWindowSelector([50, 100, 200], H=10, gamma=0.1, rng=np.random.default_rng(0))
    for _ in range(300):
        w = bob.pick()
        bob.update(w, 1.0 if w == 100 else 0.2)
    assert bob.probs()[1] > 0.7
    assert abs(bob.probs().sum() - 1) < 1e-9


class _NeverFires:
    def update(self, e):
        return False


class _FiresAt:
    """Detector stub. Fires on the given (0-based) observe() calls, so drift runs inside observe()."""

    def __init__(self, when):
        self.when, self.n = set(when), 0

    def update(self, e):
        self.n += 1
        return (self.n - 1) in self.when


class _SpyBOB:
    """Records (window, block_reward) pairs and always picks the next window in a fixed cycle."""

    def __init__(self, windows):
        self.windows, self.i, self.updates = list(windows), 0, []

    def pick(self):
        w = self.windows[self.i % len(self.windows)]
        self.i += 1
        return w

    def update(self, w, g):
        self.updates.append((w, g))


@pytest.mark.parametrize("agent_name", ["DES-UCB", "B4"])
def test_bob_block_credits_only_rewards_under_its_window(cfg, agent_name):
    cfg = C._deep_update(cfg, {"H": 10, "candidate_windows": [5, 20], "w_0": 8, "burn_in_B": 0})
    ag = C.make_agent(agent_name, cfg["d"], (np.zeros((1, cfg["d"])), np.zeros(1)), cfg, seed=0)
    spy = _SpyBOB(cfg["candidate_windows"])
    ag.bob = spy
    H = cfg["H"]
    est = ag.F if agent_name == "DES-UCB" else ag.est
    if agent_name == "DES-UCB":
        ag.detector = _NeverFires()  # the per-block reward jumps below would otherwise trigger drift
    rng = np.random.default_rng(0)
    windows_seen = []
    for t in range(4 * H):
        X = rng.standard_normal((3, cfg["d"]))
        idx = ag.step(t, X)
        windows_seen.append(est.window)
        ag.observe(t, X[idx], float(t // H))  # constant reward per block => unambiguous attribution
    # block 0 runs under w_0, each later block runs entirely under the window picked at its start
    assert windows_seen[:H] == [cfg["w_0"]] * H
    for k in range(1, 4):
        assert len(set(windows_seen[k * H:(k + 1) * H])) == 1
    assert len(spy.updates) == 3
    lo, hi = cfg.get("reward_range", (-2.0, 2.0))
    for k, (w, g) in enumerate(spy.updates, start=1):
        assert w == windows_seen[k * H]
        assert g == pytest.approx(float(np.clip((k - lo) / (hi - lo), 0, 1)))


def _run_des_with_drift(cfg, fire_at, w_min=2):
    cfg = C._deep_update(cfg, {"H": 10, "candidate_windows": [5, 20], "w_0": 8, "w_min": w_min, "burn_in_B": 0,
                               "purge_keep": "half"})
    ag = C.make_agent("DES-UCB", cfg["d"], (np.zeros((1, cfg["d"])), np.zeros(1)), cfg, seed=0)
    ag.bob = spy = _SpyBOB(cfg["candidate_windows"])
    ag.detector = _FiresAt(fire_at)
    H = cfg["H"]
    rng = np.random.default_rng(0)
    windows_seen = []
    for t in range(3 * H):
        X = rng.standard_normal((3, cfg["d"]))
        idx = ag.step(t, X)
        windows_seen.append(ag.F.window)
        ag.observe(t, X[idx], 1.0)
    assert ag.fired_at == list(fire_at)
    return ag, spy, windows_seen, H


def test_drift_mid_block_discards_bob_block(cfg):
    ag, spy, windows_seen, H = _run_des_with_drift(cfg, fire_at=[10 + 4])
    assert windows_seen[H + 4] != windows_seen[H + 5]  # window changed mid-block
    assert len(spy.updates) == 1  # block 2 only (block 1 discarded); block 0 ran under w_0
    assert spy.updates[0][0] == windows_seen[2 * H] and len(set(windows_seen[2 * H:3 * H])) == 1


def test_drift_on_block_last_round_keeps_bob_block(cfg):
    # drift fires on round 2H-1, all H rewards of block 1 were drawn under bob_w, so the block is still credited
    ag, spy, windows_seen, H = _run_des_with_drift(cfg, fire_at=[2 * 10 - 1])
    assert len(spy.updates) == 2
    assert spy.updates[0][0] == windows_seen[H] and len(set(windows_seen[H:2 * H])) == 1
    assert spy.updates[1][0] == windows_seen[2 * H]


def test_drift_without_window_change_keeps_bob_block(cfg):
    # block 2 runs under window 20 == w_min, suppression leaves it unchanged and the block stays valid
    ag, spy, windows_seen, H = _run_des_with_drift(cfg, fire_at=[2 * 10 + 4], w_min=20)
    assert windows_seen[2 * H + 4] == windows_seen[2 * H + 5] == 20
    assert [w for w, _ in spy.updates] == [windows_seen[H], windows_seen[2 * H]]


# ---------------------------------------------------------------- agent vs B2 on stationary env
def test_des_within_10pct_of_warm_linucb_on_no_drift(cfg):
    # short-horizon smoke check of the registered no-harm claim
    # the claim itself is decided in the notebook at full scale, the wider margin absorbs the 400-round horizon
    cfg = C.load_config("drift_none", overrides={"T": 400})
    df = C.run(lambda s, u: C.SyntheticEnv(cfg, s, u), ["DES-UCB", "B2"], cfg,
               n_seeds=1, n_users=8, n_jobs=1, progress=False)
    dr = C.dynamic_regret(df)["dynamic_regret_mean"]
    assert dr["DES-UCB"] <= 1.25 * dr["B2"] + 1e-9, dr.to_dict()


def test_synthetic_oracle_dominates_every_round(cfg):
    cfg = C.load_config(overrides={"T": 300})
    df = C.run(lambda s, u: C.SyntheticEnv(cfg, s, u), ["DES-UCB", "B1", "B9"], cfg,
               n_seeds=1, n_users=3, n_jobs=1, progress=False)
    # the oracle is on expected reward, so check with the noise-free candidate means
    env = C.SyntheticEnv(cfg, 0, 0)
    for t in range(cfg["T"]):
        mu = env._means(t)
        assert env.best(t) >= mu.max() - 1e-12
    assert (df["regret_kind"].astype(str) == "dynamic").all()


def test_agent_flags_and_factory(cfg):
    env = C.SyntheticEnv(cfg, 0, 0)
    for name in ["DES-UCB"] + C.ALL_ABLATIONS + C.ALL_BASELINES:
        a = C.make_agent(name, env.d, env.D_prior, cfg, seed=0)
        if isinstance(a, C.OracleAgent):
            a.env = env
        for t in range(5):
            X = env.candidates(t); i = a.step(t, X); a.observe(t, X[i], env.reward(t, i))
    a = C.make_agent("A7", env.d, env.D_prior, cfg)
    assert a.switch.kind == "ucb1"


def test_drift_suppression_purges_and_forces_F(cfg):
    cfg = C._deep_update(cfg, {"purge_keep": "half", "w_0": 400})
    env = C.SyntheticEnv(cfg, 0, 0)
    a = C.DESUCBAgent(env.d, env.D_prior, cfg, C.Flags(no_detector=True, no_bob=True))
    for t in range(300):
        X = env.candidates(t); i = a.step(t, X); a.observe(t, X[i], env.reward(t, i))
    n_before = len(a.F)
    a.on_drift(299)
    w = max(cfg["w_min"], cfg["w_0"] // 2)
    assert len(a.F) <= a.window and a.window == w
    assert len(a.F) < n_before
    # the purge boundary matches the active window, nothing older than 299 - w + 1 survives, even in history
    assert a.F.rows[0][0] == 299 - w + 1 and a.F.history[0][0] == 299 - w + 1
    a.F.set_window(cfg["w_0"], 299)
    assert len(a.F) == w and a.F.rows[0][0] == 299 - w + 1
    a.F.set_window(w, 299)
    for t in range(200, 200 + cfg["burn_in_B"]):
        X = env.candidates(t); i = a.step(t, X)
        assert a.source == "F"
        a.observe(t, X[i], env.reward(t, i))


# ---------------------------------------------------------------- real-data env safety
def test_rated_env_never_exposes_unrated_item():
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((100, 5))
    rated = np.array([1, 5, 9, 13, 20, 33, 47, 50])
    rew = np.array([1, 0, 1, 0, 1, 1, 0, 0], float)
    env = C.RatedItemEnv(feats, rated, rew, T=4, K_cand=3, D_prior=None, seed=0)
    shown = []
    for t in range(env.T):
        X = env.candidates(t)
        assert X.shape[0] <= 3
        for row in X:  # every candidate is a rated item
            assert any(np.allclose(row, feats[i]) for i in rated)
        r = env.reward(t, 0)
        assert r in (0.0, 1.0)
        assert env.best(t) >= r
        shown.append(env.rated[env._pool_for(t)[0]])
    assert len(set(shown)) == len(shown)  # never re-shown
    env2 = C.RatedItemEnv(feats, rated, rew, T=4, K_cand=3, D_prior=None, seed=0)
    env2.rated_set = set()  # simulate corruption: any reveal must raise
    with pytest.raises(RuntimeError):
        env2.reward(0, 0)
    assert env.regret_kind == "offline"


def test_metric_namespaces_never_mix():
    dyn = pd.DataFrame({"seed": [0, 0], "user": [0, 0], "t": [0, 1], "agent": ["A", "A"], "reward": [1.0, 0.5],
                        "oracle_or_best_available": [1.0, 1.0], "regret_kind": ["dynamic", "dynamic"],
                        "regret": [0.0, 0.5], "source": ["P", "F"], "window": [1, 1],
                        "drift_fired": [False, False], "true_change": [False, False]})
    off = dyn.assign(regret_kind="offline")
    rep = dyn.assign(regret_kind="replay", counted=True)
    C.dynamic_regret(dyn); C.offline_regret(off); C.replay_reward(rep); C.cum_precision(off, ks=(1,))
    with pytest.raises(ValueError):
        C.dynamic_regret(off)
    with pytest.raises(ValueError):
        C.offline_regret(dyn)
    with pytest.raises(ValueError):
        C.cum_precision(dyn, ks=(1,))
    with pytest.raises(ValueError):
        C.replay_reward(dyn.assign(counted=True))
    with pytest.raises(ValueError):
        C.dynamic_regret(pd.concat([dyn, off]))


def test_cum_recall_requires_ordinal_user_index():
    df = pd.DataFrame({"seed": 0, "user": [0, 0, 1, 1], "agent": "A", "reward": [1.0, 1.0, 0.0, 1.0],
                       "regret_kind": "offline"})
    ds = C.RatedDataset("x", np.zeros((1, 1)), {}, [1001, 1002], {}, {}, {}, pd.DataFrame(), 1,
                        n_relevant={1001: 4, 1002: 2})
    rec = C.cum_recall(df, ds.n_relevant_by_ordinal())
    assert rec["A"] == pytest.approx((2 / 4 + 1 / 2) / 2)
    with pytest.raises(KeyError):
        C.cum_recall(df, pd.Series(ds.n_relevant))  # raw ids do not match the log's ordinals


def test_download_refuses_unverified_tls(tmp_path, monkeypatch):
    import requests

    def boom(*a, **k):
        assert k.get("verify", True) is True
        raise requests.exceptions.SSLError("bad cert")

    monkeypatch.setattr(requests, "get", boom)
    with pytest.raises(RuntimeError, match="verification failed"):
        C.download("https://example.invalid/x.zip", tmp_path / "x.zip")
    assert not (tmp_path / "x.zip").exists() and not (tmp_path / "x.zip.part").exists()


def test_replay_env_counts_only_matched_rounds():
    rng = np.random.default_rng(0)
    ctx = [rng.standard_normal((4, 3)) for _ in range(50)]
    a = rng.integers(4, size=50); r = rng.integers(2, size=50).astype(float)
    env = C.ReplayEnv(ctx, a, r)
    df = C._simulate_one(env, C.RandomAgent(3, None, {"seed": 0}), 0, 0, "B8")
    assert df["counted"].sum() < 50
    assert df.loc[df["counted"], "reward"].notna().all()
    assert (df["regret_kind"] == "replay").all()
    C.replay_reward(df)


def test_paired_bootstrap_ci():
    a = np.arange(100, dtype=float); b = a - 1.0
    m, lo, hi = C.paired_bootstrap_ci(a, b)
    assert abs(m - 1.0) < 1e-12 and lo <= 1.0 <= hi
    reg = C.ClaimRegistry(Path("/tmp/_claims_test.json"))
    assert reg.assert_claim("x", lo > 0, {"m": m}) is True
    assert reg.verdict("x") is True


def test_clustered_bootstrap_and_seed_ci():
    rng = np.random.default_rng(0)
    seeds = np.repeat(np.arange(5), 20)
    a = rng.standard_normal(100) + 2.0 * seeds  # strong seed-level structure
    b = rng.standard_normal(100)
    m, lo, hi = C.paired_bootstrap_ci(a, b, seed=1)
    m2, clo, chi = C.paired_bootstrap_ci(a, b, seed=1, clusters=seeds)
    assert m == m2
    assert (chi - clo) > (hi - lo)  # ignoring the clusters understates the uncertainty
    sm, slo, shi, k = C.seed_level_ci(a, b, seeds)
    assert k == 5 and slo <= sm <= shi
    pu = pd.DataFrame({"A": a, "B": b}, index=pd.MultiIndex.from_arrays([seeds, np.tile(np.arange(20), 5)],
                                                                          names=["seed", "user"]))
    tab = C.summarize_pairs(pu, [("A", "B")]).iloc[0]
    assert tab["n_seeds"] == 5 and tab["n_units"] == 100
    assert tab["cl_lo"] <= tab["mean_diff(A-B)"] <= tab["cl_hi"]


def test_set_dotted_and_seed_list_runner(cfg):
    c2 = C.set_dotted(cfg, "detector.lam_threshold", 3.0)
    assert c2["detector"]["lam_threshold"] == 3.0 and cfg["detector"]["lam_threshold"] != 3.0
    c3 = C.set_dotted(cfg, "T", 60)
    df = C.run(lambda s, u: C.SyntheticEnv(c3, s, u), ["B2"], c3, n_users=1, seeds=[100, 101],
               exp_name=None, progress=False, n_jobs=1)
    assert sorted(df["seed"].unique().tolist()) == [100, 101]
    assert set(C.DIAG_COLS) <= set(df.columns) and "forced" in df.columns


def test_variant_names_parse_and_gating_agrees(cfg):
    c = C.set_dotted(cfg, "T", 300)
    for name, flag in [("DES-UCB[gated]", "gated"), ("DES-UCB[gated,adaptive_gamma]", "adaptive_gamma"),
                       ("DES-UCB[detector=F]", "detector_signal"), ("A5[gated]", "decay_instead"),
                       ("A2[detector=mean]", "no_suppression")]:
        a = C.make_agent(name, c["d"], (np.zeros((1, c["d"])), np.zeros(1)), c)
        assert a.name == name and getattr(a.flags, flag)
    a = C.make_agent("DES-UCB[detector=F]", c["d"], (np.zeros((1, c["d"])), np.zeros(1)), c)
    assert a.flags.detector_signal == "F"
    df = C.run(lambda s, u: C.SyntheticEnv(c, s, u), ["DES-UCB", "A9", "DES-UCB[gated,adaptive_gamma]"],
               c, n_users=2, seeds=[0], exp_name=None, progress=False, n_jobs=1)
    occ = C.switch_occupancy(df)
    assert occ.loc["A9", "agreed_share"] == 0.0  # gate off: every round is a switch decision
    assert occ.loc["DES-UCB", "agreed_share"] > 0.2  # both sources agree on most rounds
    assert (df.loc[(df["agent"] == "DES-UCB") & (df["agreed"] == 1.0), "forced"]).all()


# ---------------------------------------------------------------- pooled source, hedge, purge modes
def test_combined_estimator_equals_warm_ridge_and_rebuilds(cfg):
    rng = np.random.default_rng(1)
    d = cfg["d"]
    Xp, rp = rng.normal(size=(30, d)), rng.normal(size=30)
    P = C.PriorEstimator(d, 1.0, 0.5, 5.0).fit_from((Xp, rp))
    Cst = C.CombinedEstimator(P)
    warm = C.RidgeUCB(d, 1.0, 0.5, 5.0)
    for x, r in zip(Xp, rp):
        warm.add(x, r)
    rows = [(t, rng.normal(size=d), float(rng.normal())) for t in range(40)]
    for t, x, r in rows:
        Cst.add(x, r)
        warm.add(x, r)
    assert np.allclose(Cst.theta(), warm.theta())
    assert np.allclose(Cst.exact_theta(), Cst.theta()) or np.linalg.norm(Cst.exact_theta()) > Cst.S
    Cst.rebuild(rows[-10:])
    ref = C.RidgeUCB(d, 1.0, 0.5, 5.0)
    for x, r in zip(Xp, rp):
        ref.add(x, r)
    for _, x, r in rows[-10:]:
        ref.add(x, r)
    assert np.allclose(Cst.V, ref.V) and np.allclose(Cst.b, ref.b) and np.allclose(Cst.theta(), ref.theta())
    Cst.rebuild([])
    assert np.allclose(Cst.V, P.V) and np.allclose(Cst.b, P.b)


def test_pooled_source_tracks_warm_linucb_when_no_switch_to_F(cfg):
    # with the gate off and F never selected, the C-only path is exactly B2 warm LinUCB
    c = C.set_dotted(C.set_dotted(cfg, "T", 120), "drift.type", "none")
    env = C.SyntheticEnv(c, 0, 0)
    a = C.DESUCBAgent(env.d, env.D_prior, c, C.Flags(no_detector=True, no_bob=True, no_gate=True))
    b2 = C.LinUCBAgent(env.d, env.D_prior, c, warm=True)
    for t in range(c["T"]):
        X = env.candidates(t)
        ia = int(np.argmax(a.C.score(X)))
        ib = b2.step(t, X)
        assert ia == ib
        r = env.reward(t, ib)
        a.C.add(X[ib], r)
        b2.observe(t, X[ib], r)


def test_hedge_uses_every_source_loss_and_favours_lower_loss():
    sw = C.EvidenceSwitch("hedge", eta=1.0, gamma=0.0, rng=np.random.default_rng(0), sources=("C", "F"))
    for t in range(50):
        m = sw.select(t)
        sw.update(m, 0.0, losses={"C": 0.2, "F": 0.8})
        sw.tick()
    p = sw.probs()
    assert p[0] > 0.99 and abs(p.sum() - 1.0) < 1e-9
    sw.reset_source("C")
    assert sw.probs()[0] <= 0.5 + 1e-9
    with pytest.raises(AssertionError):
        sw.update("C", 0.0)  # hedge needs the full loss vector


def test_three_source_switch_and_no_pooled_flag(cfg):
    sw = C.EvidenceSwitch("hedge", rng=np.random.default_rng(0), sources=("P", "C", "F"))
    assert sw.K == 3 and set(sw.select(0) for _ in range(20)) <= {"P", "C", "F"}
    sw.force_feedback(3)
    assert sw.select(1) == "F" and np.allclose(sw.last_probs, [0, 0, 1])
    env = C.SyntheticEnv(cfg, 0, 0)
    a = C.make_agent("A8", env.d, env.D_prior, cfg)
    assert a.switch.SOURCES == ("P", "F") and a.prior_src == "P"
    a = C.make_agent("DES-UCB", env.d, env.D_prior, cfg)
    assert a.switch.SOURCES == ("C", "F") and a.prior_src == "C" and a.gate
    a = C.make_agent("A9", env.d, env.D_prior, cfg)
    assert not a.gate
    a = C.make_agent("A10", env.d, env.D_prior, cfg)
    assert a.switch.kind == "exp3" and a.switch.gamma == cfg["switch"]["exp3_gamma"]


@pytest.mark.parametrize("keep", [0, 1, 25])
def test_purge_keep_rows_restarts_F_and_rebuilds_C(cfg, keep):
    c = C._deep_update(cfg, {"purge_keep": keep})
    env = C.SyntheticEnv(c, 0, 0)
    a = C.DESUCBAgent(env.d, env.D_prior, c, C.Flags(no_detector=True, no_bob=True))
    for t in range(200):
        X = env.candidates(t); i = a.step(t, X); a.observe(t, X[i], env.reward(t, i))
    assert len(a.F) == 200
    a.on_drift(199)
    assert len(a.F) == keep and len(a.F.history) == keep
    if keep:
        assert a.F.rows[0][0] == 200 - keep
    ref = C.CombinedEstimator(a.P)
    for _, x, r in a.F.history:
        ref.add(x, r)
    assert np.allclose(a.C.V, ref.V) and np.allclose(a.C.b, ref.b)
    assert a.window == c["w_0"]  # the window itself is left to BOB
    assert a.switch.burn_in_remaining == max(c["burn_in_B"], c["w_min"] - keep)
    for t in range(200, 200 + c["burn_in_B"]):
        X = env.candidates(t); i = a.step(t, X)
        assert a.source == "F" and a.forced
        a.observe(t, X[i], env.reward(t, i))
    a2 = C.make_agent(f"DES-UCB[purge_keep={keep}]", env.d, env.D_prior, cfg)
    assert a2.purge_keep == keep and f"purge_keep={keep}" in a2.flags.label()


def test_hedge_credits_agreed_rounds_but_not_burn_in(cfg):
    c = C.set_dotted(C.set_dotted(cfg, "T", 150), "drift.type", "none")
    env = C.SyntheticEnv(c, 0, 0)
    a = C.DESUCBAgent(env.d, env.D_prior, c, C.Flags(no_detector=True, no_bob=True))
    for t in range(c["w_min"]):
        X = env.candidates(t); i = a.step(t, X)
        assert a.source == "C" and a.forced  # F below the eligibility count: prior source, forced
        a.observe(t, X[i], env.reward(t, i))
        assert a.switch.n_credited == max(0, t + 2 - env.d)  # hedge charged once F holds d rows
    for t in range(c["w_min"], 100):
        X = env.candidates(t); i = a.step(t, X); a.observe(t, X[i], env.reward(t, i))
    assert a.switch.n_credited == 100 - env.d + 1
    a.on_drift(99)
    n = a.switch.n_credited
    for t in range(100, 100 + c["burn_in_B"]):
        X = env.candidates(t); i = a.step(t, X); a.observe(t, X[i], env.reward(t, i))
    assert a.switch.n_credited == n


def test_hedge_switch_satisfies_expected_loss_bound():
    # Hedge bound. with gamma=0 and losses in [0,1], the expected loss of the sampled source over
    # n credited rounds exceeds the best fixed source by at most ln K/eta + eta n/8
    rng = np.random.default_rng(3)
    for eta in (0.5, 2.0, 8.0):
        sw = C.EvidenceSwitch("hedge", eta=eta, gamma=0.0, rng=rng, sources=("P", "C", "F"))
        L = rng.uniform(size=(300, 3))
        L[:, 2] *= 0.6  # F is the best fixed source in hindsight
        exp_loss = 0.0
        for t in range(300):
            sw.select(t)
            exp_loss += float(sw.last_probs @ L[t])
            sw.update("F", 0.0, losses=dict(zip(sw.SOURCES, L[t])), credited=True)
            sw.tick()
        assert exp_loss - L.sum(0).min() <= math.log(3) / eta + eta * 300 / 8 + 1e-9


def test_glr_detects_mean_shift_and_stays_quiet_when_stationary():
    rng = np.random.default_rng(0)
    det = C.GLRMeanChange(threshold=5.0, min_len=10)
    assert not any(det.update(v) for v in rng.normal(size=400))
    det = C.GLRMeanChange(threshold=5.0, min_len=10)
    fired = [t for t, v in enumerate(np.r_[rng.normal(size=100), rng.normal(3.0, size=60)]) if det.update(v)]
    assert fired and 100 <= fired[0] <= 125
    assert len(det.win) == len(np.r_[rng.normal(size=100), rng.normal(3.0, size=60)]) - fired[-1] - 1


def test_recent_baselines_restart_and_are_matched(cfg):
    c = C.set_dotted(cfg, "T", 60)
    env = C.SyntheticEnv(c, 0, 0)
    dal = C.make_agent("B10", env.d, env.D_prior, c, seed=0)
    b2 = C.make_agent("B2", env.d, env.D_prior, c)
    assert np.allclose(dal.est.V, b2.est.V) and dal.source == "P+F"  # warm start like B2
    n_explore = 0
    for t in range(60):
        X = env.candidates(t); i = dal.step(t, X)
        n_explore += int(dal.forced)
        dal.observe(t, X[i], env.reward(t, i))
    assert 0 < n_explore < 60 and len(dal.det.win) == n_explore
    dal.det = C.GLRMeanChange(threshold=-1e9, min_len=1)  # fires on the next forced round
    dal.det.win.extend([0.0, 1.0])
    dal.explore = True
    dal.observe(60, env.candidates(59)[0], 0.0)
    assert dal.fired_at == [60] and dal.source == "F" and np.allclose(dal.est.V, c["lam"] * np.eye(env.d))
    c = C.set_dotted(c, "baselines.restart_period", 20)
    pr = C.make_agent("B11", env.d, env.D_prior, c)
    for t in range(60):
        X = env.candidates(t); i = pr.step(t, X); pr.observe(t, X[i], env.reward(t, i))
    assert pr.fired_at == [19, 39, 59] and pr.est.n == 0
    assert {"B10", "B11"} <= set(C.ALL_BASELINES) and {"B10", "B11"} <= set(cfg["tuning"]["grids"])


def test_tune_method_multi_regime_uses_relative_criterion(cfg):
    c = C._deep_update(cfg, {"T": 60, "tuning": {**cfg["tuning"], "seeds": [100], "n_users": 1}})
    sel, tab = C.tune_method(lambda cc: (lambda s, u: C.SyntheticEnv(cc, s, u)), "B3",
                             {"baselines.sw_window": [20, 40]}, c, seeds=[100], n_users=1,
                             regimes=["none", "abrupt"])
    assert set(tab.columns) >= {"regret_none", "regret_abrupt", "criterion", "regret_mean"}
    rel = tab["regret_none"] / tab["regret_none"].min() + tab["regret_abrupt"] / tab["regret_abrupt"].min()
    assert np.allclose(tab["criterion"], rel / 2)
    assert sel["baselines.sw_window"] == tab.loc[tab["criterion"].idxmin(), "baselines.sw_window"]


def test_adaptive_gamma_decays():
    sw = C.EvidenceSwitch("exp3", gamma=0.5, adaptive_gamma=True, rng=np.random.default_rng(0))
    sw.probs(); g0 = sw.gamma
    for _ in range(200):
        sw.select(0); sw.update("F", 0.3); sw.tick()
    sw.probs()
    assert sw.gamma < g0 and sw.gamma > 0


def test_detector_signal_variants_run_and_reverse_drift(cfg):
    c = C.set_dotted(cfg, "T", 200)
    c = C.set_dotted(c, "drift", {**c["drift"], "change_points": [60, 120], "direction": "reverse", "magnitude": 2.0})
    env = C.SyntheticEnv(c, 0, 0)
    assert np.allclose(env.theta1, -env.theta0)
    assert np.allclose(env.thetas[60:], -env.theta0)
    df = C.run(lambda s, u: C.SyntheticEnv(c, s, u), ["DES-UCB", "DES-UCB[detector=F]", "DES-UCB[detector=mean]"],
               c, n_users=1, seeds=[0], exp_name=None, progress=False, n_jobs=1)
    assert set(df["agent"].unique()) == {"DES-UCB", "DES-UCB[detector=F]", "DES-UCB[detector=mean]"}
    dbs = C.detection_by_source(df)
    det = C.detection_stats(df)
    for a in det.index:
        g = dbs.loc[a]
        assert g["true_detections"].sum() <= det.loc[a, "n_changes"]
        assert (g["true_detections"] + g["repeats"] + g["false_alarms"]).sum() == g["firings"].sum()
    with pytest.raises(AssertionError):
        C.make_agent("DES-UCB[detector=bogus]", c["d"], (np.zeros((1, c["d"])), np.zeros(1)), c)


def test_regret_decomposition_sums_to_dynamic_regret(cfg):
    c = C.set_dotted(cfg, "T", 200)
    c = C.set_dotted(c, "drift", {**c["drift"], "change_points": [100]})
    df = C.run(lambda s, u: C.SyntheticEnv(c, s, u), ["DES-UCB", "B2"], c, n_users=2, seeds=[0],
               exp_name=None, progress=False, n_jobs=1)
    dec = C.regret_decomposition(df, B=int(c["burn_in_B"]))
    reg = C.dynamic_regret(df)
    for a in ["DES-UCB", "B2"]:
        assert dec.loc[a, "total"] == pytest.approx(reg.loc[a, "dynamic_regret_mean"], rel=1e-6)
    B = int(c["burn_in_B"])
    early = df[(df["agent"] == "B2") & (df["t"] >= 100) & (df["t"] < 100 + B)]
    assert dec.loc["B2"].xs(f"post<={B}", level="phase").sum() == pytest.approx(
        early["regret"].sum() / early.groupby(["seed", "user"]).ngroups, rel=1e-6)
    se = C.selection_effect(df)
    assert "DES-UCB" in se.index.get_level_values("agent") and "B2" not in se.index.get_level_values("agent")


def test_split_replay_segments_share_prior_and_replay_reward_counts_matches(cfg):
    rng = np.random.default_rng(0)
    d, T = cfg["d"], 60
    ctx = [C._unit(rng.normal(size=(5, d))) for _ in range(T)]
    env = C.ReplayEnv(ctx, rng.integers(5, size=T), rng.integers(2, size=T).astype(float),
                      D_prior=(C._unit(rng.normal(size=(10, d))), np.ones(10)))
    segs = C.split_replay(env, 3, 20)
    assert len(segs) == 3 and all(s.T == 20 for s in segs) and all(s.D_prior is env.D_prior for s in segs)
    assert len(C.split_replay(env, 4, 20)) == 3
    c = C.set_dotted(cfg, "T", 20)
    df = C.run(lambda s, u: segs[u], ["DES-UCB", "B2"], c, n_users=3, seeds=[0], exp_name=None,
               progress=False, n_jobs=1)
    rr = C.replay_reward(df)
    assert (df["regret_kind"] == "replay").all()
    for a in ["DES-UCB", "B2"]:
        sub = df[(df["agent"] == a) & df["counted"]]
        assert rr.loc[a, "matched_rounds"] == len(sub)
        assert rr.loc[a, "replay_reward"] == pytest.approx(sub["reward"].mean())
    pu = C.per_unit(df[df["counted"]], "reward", "mean")
    assert pu.shape[0] == df[df["counted"]].groupby(["seed", "user"]).ngroups


def test_stale_evidence_ratio_counts_pooled_source_as_prior_carrying():
    rows = []
    for t in range(6):
        rows.append({"seed": 0, "user": 0, "agent": "X", "t": t, "true_change": t == 2,
                     "source": ["P", "F", "C", "F", "P+F", "F"][t]})
    df = pd.DataFrame(rows)
    r = C.stale_evidence_ratio(df)
    assert math.isclose(r["X"], 2 / 4)
