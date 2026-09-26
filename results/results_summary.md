# Results summary

Built from results/claims.json.

## Supported claims
- **sanity_oracle** — oracle expected reward >= every agent's expected reward, every round. Numbers: min_per_round_regret=0, regret_B9=0, regret_min_other=24.7
- **suppression** — dyn_regret[DES] < dyn_regret[A2 no_suppression] on abrupt drift with a stale (pre-drift) prior. Numbers: regret_DES=24.7, regret_A2=78, mean_diff=-53.3, cl_lo=-55.63, cl_hi=-51.17, flat_lo=-54.98, flat_hi=-51.55, seed_lo=-55.21, seed_hi=-51.39, n_units=1000, n_seeds=10, detector_mean_delay=8.993, detected_events=1996, n_change_events=2000, detected_fraction=0.998
- **stale_ratio** — post-change rounds conditioned on a prior-carrying source (P or C): DES < A2. Numbers: stale_DES=0.0018, stale_A2=0.02614, mean_diff=-0.02434, cl_lo=-0.02547, cl_hi=-0.02306, flat_lo=-0.02518, flat_hi=-0.02348, seed_lo=-0.02537, seed_hi=-0.0233, n_units=1000, n_seeds=10
- **not_redundant** — purge+reset+burn-in beats passive decay of the prior (A5). Numbers: regret_DES=24.7, regret_A5_decay=72.99, mean_diff=-48.29, cl_lo=-50.47, cl_hi=-46.28, flat_lo=-49.99, flat_hi=-46.5, seed_lo=-49.77, seed_hi=-46.81, n_units=1000, n_seeds=10
- **restart_not_redundant** — evidence switching + purge beats a detector-triggered full restart (B6) on abrupt drift with a stale prior. Numbers: regret_DES=24.7, regret_B6=31.29, mean_diff=-6.591, cl_lo=-7.059, cl_hi=-6.079, flat_lo=-6.939, flat_hi=-6.222, seed_lo=-6.995, seed_hi=-6.187, n_units=1000, n_seeds=10
- **recent_baselines** — DES-UCB beats the matched detection-augmented (B10) and periodic-restart (B11) LinUCB baselines on abrupt drift. Numbers: regret_DES=24.7, regret_B10=158.5, regret_B11=82.31, B10_mean_diff=-133.8, B10_cl_lo=-137.3, B10_cl_hi=-130.4, B10_flat_lo=-136.5, B10_flat_hi=-131, B10_seed_lo=-136.5, B10_seed_hi=-131.1, B10_n_units=1000, B10_n_seeds=10, B11_mean_diff=-57.6, B11_cl_lo=-59.07, B11_cl_hi=-56.22, B11_flat_lo=-58.79, B11_flat_hi=-56.4, B11_seed_lo=-58.6, B11_seed_hi=-56.61, B11_n_units=1000, B11_n_seeds=10
- **recurring_not_redundant** — DES-UCB beats the detector-triggered restart (B6) on recurring drift with a stale prior. Numbers: regret_DES=30.89, regret_B6=39.29, regret_A5=68.38, regret_B10=113.4, regret_B11=96.76, mean_diff=-8.395, cl_lo=-8.875, cl_hi=-7.935, flat_lo=-8.741, flat_hi=-8.065, seed_lo=-8.787, seed_hi=-8.004, n_units=1000, n_seeds=10
- **suppression_mis** — same as `suppression` under p_mismatch=0.2 with a population (not pre-drift) prior. Numbers: regret_DES=36.92, regret_A2=82.39, mean_diff=-45.48, cl_lo=-47.55, cl_hi=-43.39, flat_lo=-47.13, flat_hi=-43.84, seed_lo=-47.06, seed_hi=-43.89, n_units=1000, n_seeds=10
- **misleading_prior** — misleading (negated) post-drift prior: purge+reset+burn-in beats passive decay (A5), clustered CI. Numbers: regret_DES=14.53, regret_A5_decay=427, regret_A2=439, regret_B6=16.76, regret_B2=847.1, regret_B10=804.4, regret_B11=368.5, mean_diff=-412.5, cl_lo=-422.4, cl_hi=-402.4, flat_lo=-419, flat_hi=-405.8, seed_lo=-421.4, seed_hi=-403.6, n_units=1000, n_seeds=10, mean_diff_vs_A2=-424.5, cl_excl0_vs_A2=True
- **rate** — log-log regret slope on sinusoidal drift: DES <= B4 + 0.05. Numbers: slope_DES=1.037, slope_B4=1.205, checkpoints=[250, 500, 1000, 2000, 4000]

## Not supported
- **no_harm** [NOT SUPPORTED] — dyn_regret[DES] <= 1.10 * dyn_regret[B2 warm LinUCB] with no drift. Numbers: regret_DES=2.118, regret_B2=1.661, ratio=1.275, regret_A9_no_gate=1.993, regret_A8_no_pooled=7.286, regret_B6=8.326, regret_B10=19.94, regret_B11=8.891, F_fraction_DES=0.8775, agreed_share_DES=0.8743, false_alarms_DES=72
- **real_precision** [NOT SUPPORTED] — ML-1M protocol B: cum_precision@40[DES] >= cum_precision@40[B2]. Numbers: cum_precision@40_DES=0.7179, cum_precision@40_B2=0.7494, mean_diff=-0.03144, cl_lo=-0.03477, cl_hi=-0.02805, flat_lo=-0.03463, flat_hi=-0.02825, seed_lo=-0.03564, seed_hi=-0.02725, n_units=885, n_seeds=3

## Decision table
- NO DECISION-TABLE BRANCH TRIGGERED (claims consistent with the title as stated)