"""Walk-forward intraday forecaster, built so it can fail.

The model: gradient boosting on ~28 point-in-time features, predicting
P(target before stop) for each side, then isotonic-calibrated so the
probability it reports means what it says. The trade decision is not the
probability, it is expected value net of the measured cost:

    EV = p * reward - (1 - p) * 1 - cost_r        (in units of risk)

and nothing trades unless EV > 0 on data the model has never seen.

Every guard here exists because this project has already produced three
convincing edges that were not real:

  * PURGED WALK-FORWARD. Train on sessions strictly before a fold, predict
    the fold, never refit inside it. Plus a one-session embargo, because a
    position opened near the close of day D is still resolving into D+1 and
    training on D+1 would leak its outcome backwards.
  * CALIBRATION FITTED ON A HELD-OUT TAIL of the training window, never on
    the fold being predicted.
  * THE GEOMETRY IS CHOSEN ON TRAIN ONLY, and all twelve candidates are
    counted in the correction.
  * A PERMUTATION NULL. The same pipeline is re-run with labels shuffled
    within each session. If the real result does not clearly exceed the
    shuffled distribution, there is nothing here. This is the test that
    would have caught the two earlier false edges immediately.
  * DAY-BLOCKED CONFIDENCE INTERVALS throughout, because signals inside one
    session are correlated and treating them as independent is exactly how
    noise starts looking significant.

A negative result is the expected outcome and is reported as plainly as a
positive one.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "bar_cache"
OUT_DIR = ROOT / "forecast_cache"
OUT_DIR.mkdir(exist_ok=True)


warnings.filterwarnings("ignore")

# A permutation p-value must be (1 + beats) / (1 + shuffles), never
# beats/shuffles. With 10 shuffles the smallest ATTAINABLE p is 1/11 =
# 0.091, so reporting 0.0000 when no shuffle wins claims a precision the
# test cannot deliver - and 0.091 does not clear 0.05, which reverses the
# conclusion. The +1 counts the observed statistic as one of its own
# reference draws, which is what makes the test exact.

SP = OUT_DIR
DATA = OUT_DIR / "ml_dataset.parquet"

STOP_FRACTIONS = (0.25, 0.5, 0.75)
REWARD_RISKS = (0.75, 1.0, 1.5, 2.0)
GEOMETRIES = [(s, r) for s in STOP_FRACTIONS for r in REWARD_RISKS]

N_FOLDS = 3          # 760k rows x 12 geometries x 2 sides: 3 folds keeps this to ~15 min
EMBARGO_DAYS = 1
CALIB_DAYS = 20
SEED = 20260909
MAX_FIT_ROWS = 250_000
PERMUTATIONS = 10

FEATURES = [
    "vwap_dist_atr", "orb_pos", "orb_break_up", "orb_break_down",
    "range_pos", "day_range_atr", "rvol", "mom6_atr", "mom12_atr",
    "day_change_pct", "gap_pct", "bench_change", "rel_strength",
    "bar_range_atr", "up_bars_7", "vol_burst", "prev_close_dist_atr",
    "cpr_dist_atr", "frac_session_done", "atr_pct", "bars_left",
    "xs_rvol", "xs_vwap_dist_atr", "xs_mom6_atr", "xs_day_change_pct",
    "xs_range_pos", "xs_vol_burst", "xs_rel_strength",
]


def tag(stop_fraction: float, reward: float) -> str:
    return f"s{int(stop_fraction * 100)}r{int(reward * 100)}"


def day_bootstrap(values: np.ndarray, days: np.ndarray,
                  draws: int = 1500) -> tuple:
    """(mean, lo, hi, one-sided p for mean>0) resampling whole session-days."""
    if values.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    order = np.argsort(days, kind="stable")
    values, days = values[order], days[order]
    edges = np.flatnonzero(np.r_[True, days[1:] != days[:-1]])
    blocks = np.split(values, edges[1:])
    if len(blocks) < 5:
        return float(values.mean()), float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(SEED)
    n = len(blocks)
    means = np.empty(draws)
    for d in range(draws):
        pick = rng.integers(0, n, size=n)
        means[d] = np.concatenate([blocks[p] for p in pick]).mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(values.mean()), float(lo), float(hi),
            float((means <= 0.0).mean()))


def build_folds(days: list) -> list:
    """Expanding-window folds with an embargo between train and test."""
    n = len(days)
    start = int(n * 0.45)
    step = max(1, (n - start) // N_FOLDS)
    folds = []
    for k in range(N_FOLDS):
        train_end = start + k * step
        test_start = train_end + EMBARGO_DAYS
        test_end = test_start + step if k < N_FOLDS - 1 else n
        if test_start >= n or test_end <= test_start:
            continue
        folds.append((days[:train_end], days[test_start:test_end]))
    return folds


def fit_side(train: pd.DataFrame, label_column: str, calib_days: set):
    """Train the classifier, then calibrate on a held-out tail of train."""
    fit_mask = ~train["day"].isin(calib_days)
    fit_part, calib_part = train[fit_mask], train[~fit_mask]
    # Cap the fitting set. Sampled uniformly at random over ROWS, not days,
    # so every session stays represented and no period is dropped whole.
    if len(fit_part) > MAX_FIT_ROWS:
        fit_part = fit_part.sample(MAX_FIT_ROWS, random_state=SEED)
    if len(fit_part) < 2000 or len(calib_part) < 500:
        return None, None
    if fit_part[label_column].nunique() < 2:
        return None, None
    model = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.06, max_depth=4,
        min_samples_leaf=200, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15,
        random_state=SEED)
    model.fit(fit_part[FEATURES].to_numpy(dtype=np.float32),
              fit_part[label_column].to_numpy())
    raw = model.predict_proba(
        calib_part[FEATURES].to_numpy(dtype=np.float32))[:, 1]
    if calib_part[label_column].nunique() < 2:
        return model, None
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw, calib_part[label_column].to_numpy())
    return model, iso


def predict(model, iso, frame: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(frame[FEATURES].to_numpy(dtype=np.float32))[:, 1]
    return iso.predict(raw) if iso is not None else raw


def walk_forward(data: pd.DataFrame, stop_fraction: float, reward: float,
                 shuffle: bool = False, rng=None) -> pd.DataFrame:
    """Out-of-fold predictions and realised outcomes for one geometry."""
    suffix = tag(stop_fraction, reward)
    long_label, short_label = f"L_{suffix}", f"S_{suffix}"
    long_r, short_r = f"Lr_{suffix}", f"Sr_{suffix}"
    cost_column = f"cost_{suffix}"
    days = sorted(data["day"].unique())
    out = []
    for train_days, test_days in build_folds(days):
        train = data[data["day"].isin(set(train_days))]
        test = data[data["day"].isin(set(test_days))]
        if train.empty or test.empty:
            continue
        if shuffle:
            # Permute labels WITHIN each session, so the null keeps the same
            # day structure and feature distribution and destroys only the
            # feature-to-outcome link.
            train = train.copy()
            for column in (long_label, short_label):
                train[column] = (
                    train.groupby("day")[column]
                    .transform(lambda s: rng.permutation(s.to_numpy())))
        calib = set(train_days[-CALIB_DAYS:])
        long_model, long_iso = fit_side(train, long_label, calib)
        short_model, short_iso = fit_side(train, short_label, calib)
        if long_model is None or short_model is None:
            continue
        block = test[["symbol", "day", "bar", long_label, short_label,
                      long_r, short_r, cost_column]].copy()
        block["p_long"] = predict(long_model, long_iso, test)
        block["p_short"] = predict(short_model, short_iso, test)
        block["ev_long"] = (block["p_long"] * reward
                            - (1 - block["p_long"]) - block[cost_column])
        block["ev_short"] = (block["p_short"] * reward
                             - (1 - block["p_short"]) - block[cost_column])
        block = block.rename(columns={long_label: "y_long",
                                      short_label: "y_short",
                                      long_r: "r_long", short_r: "r_short",
                                      cost_column: "cost_r"})
        out.append(block)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def net_of_selection(frame: pd.DataFrame) -> tuple:
    """Net R of taking whichever side has positive EV, best side per row."""
    if frame.empty:
        return np.array([]), np.array([])
    take_long = (frame["ev_long"] > 0) & (frame["ev_long"] >= frame["ev_short"])
    take_short = (frame["ev_short"] > 0) & (frame["ev_short"] > frame["ev_long"])
    keep = take_long | take_short
    chosen = frame[keep]
    if chosen.empty:
        return np.array([]), np.array([])
    net = np.where(take_long[keep].to_numpy(),
                   (chosen["r_long"] - chosen["cost_r"]).to_numpy(),
                   (chosen["r_short"] - chosen["cost_r"]).to_numpy())
    return net.astype(float), chosen["day"].to_numpy()


def main() -> int:
    if not DATA.exists():
        print(f"missing {DATA} - run build_dataset2.py first")
        return 1
    wanted = ["symbol", "day", "bar"] + FEATURES
    for stop_fraction, reward in GEOMETRIES:
        suffix = tag(stop_fraction, reward)
        wanted += [f"L_{suffix}", f"Lr_{suffix}", f"S_{suffix}",
                   f"Sr_{suffix}", f"cost_{suffix}"]
    data = pd.read_parquet(DATA, columns=wanted)
    print(f"rows {len(data):,} | days {data.day.nunique()} | "
          f"symbols {data.symbol.nunique()}")
    missing = [f for f in FEATURES if f not in data.columns]
    if missing:
        print(f"FATAL: missing features {missing}")
        return 1
    days = sorted(data["day"].unique())
    folds = build_folds(days)
    print(f"folds: {len(folds)} (embargo {EMBARGO_DAYS}d, "
          f"calibration tail {CALIB_DAYS}d)")
    for k, (tr, te) in enumerate(folds, start=1):
        print(f"  fold {k}: train {len(tr)}d -> test {len(te)}d "
              f"({te[0]} .. {te[-1]})")

    print()
    print("=" * 104)
    print("STEP 1  geometry selection on TRAINING data only")
    print("=" * 104)
    tune = data[data["day"].isin(set(folds[0][0]))]
    scored = []
    for stop_fraction, reward in GEOMETRIES:
        suffix = tag(stop_fraction, reward)
        need = (1.0 + tune[f"cost_{suffix}"].median()) / (1.0 + reward)
        have = float(tune[f"L_{suffix}"].mean())
        scored.append((stop_fraction, reward, have, need, have - need))
        print(f"  s{stop_fraction} r{reward}: base hit {have:.4f}  "
              f"needs {need:.4f} to clear cost  gap {have - need:+.4f}")
    best = max(scored, key=lambda row: row[4])
    print(f"\n  chosen on train: stop {best[0]} sigma, reward {best[1]}:1")

    print()
    print("=" * 104)
    print("STEP 2  walk-forward out-of-fold, every geometry (all 12 counted)")
    print("=" * 104)
    print(f"{'geometry':>12} {'trades':>8} {'of rows':>8} {'AUC':>7} "
          f"{'Brier':>8} {'base':>7} {'net R':>9} {'95% CI':>21} {'p':>7}")
    print("-" * 104)
    results = []
    for stop_fraction, reward in GEOMETRIES:
        frame = walk_forward(data, stop_fraction, reward)
        if frame.empty:
            print(f"{'s' + str(stop_fraction) + ' r' + str(reward):>12} "
                  f"{'no folds':>8}")
            continue
        net, net_days = net_of_selection(frame)
        auc = (roc_auc_score(frame["y_long"], frame["p_long"])
               if frame["y_long"].nunique() > 1 else float("nan"))
        brier = brier_score_loss(frame["y_long"], frame["p_long"])
        obs, lo, hi, p = day_bootstrap(net, net_days)
        results.append({"geom": (stop_fraction, reward), "n": int(net.size),
                        "net": obs, "p": p, "auc": auc})
        print(f"{'s' + str(stop_fraction) + ' r' + str(reward):>12} "
              f"{net.size:>8,} {net.size / len(frame) * 100:>7.1f}% "
              f"{auc:>7.4f} {brier:>8.4f} {float(frame['y_long'].mean()):>7.4f} "
              f"{obs:>9.4f} [{lo:>8.4f},{hi:>8.4f}] {p:>7.4f}")

    print()
    print("=" * 104)
    print(f"STEP 3  permutation null on the train-chosen geometry "
          f"(s{best[0]} r{best[1]}), {PERMUTATIONS} shuffles")
    print("=" * 104)
    real_frame = walk_forward(data, best[0], best[1])
    real_net, real_days = net_of_selection(real_frame)
    real_mean = float(real_net.mean()) if real_net.size else float("nan")
    real_auc = (roc_auc_score(real_frame["y_long"], real_frame["p_long"])
                if not real_frame.empty and real_frame["y_long"].nunique() > 1
                else float("nan"))
    rng = np.random.default_rng(SEED)
    null_means, null_aucs = [], []
    for k in range(PERMUTATIONS):
        frame = walk_forward(data, best[0], best[1], shuffle=True, rng=rng)
        if frame.empty:
            continue
        net, _ = net_of_selection(frame)
        if net.size:
            null_means.append(float(net.mean()))
        if frame["y_long"].nunique() > 1:
            null_aucs.append(roc_auc_score(frame["y_long"], frame["p_long"]))
        print(f"    shuffle {k + 1}/{PERMUTATIONS}: "
              f"net {null_means[-1]:+.4f}" if null_means else "", flush=True)
    if null_means:
        arr = np.array(null_means)
        print(f"\n  real  net R {real_mean:+.4f}   AUC {real_auc:.4f}   "
              f"trades {real_net.size:,}")
        print(f"  null  net R {arr.mean():+.4f} +/- {arr.std():.4f}  "
              f"range [{arr.min():+.4f}, {arr.max():+.4f}]")
        if null_aucs:
            print(f"  null  AUC {np.mean(null_aucs):.4f} "
                  f"+/- {np.std(null_aucs):.4f}")
        beats = int((arr >= real_mean).sum())
        exact_p = (1 + beats) / (1 + arr.size)
        print(f"  permutation p = (1+{beats})/(1+{arr.size}) = "
              f"{exact_p:.4f}   (floor with {arr.size} shuffles: "
              f"{1 / (1 + arr.size):.4f})")
        print(f"  excess over null: {real_mean - arr.mean():+.4f} R")

    print()
    print("=" * 104)
    print("STEP 4  verdict")
    print("=" * 104)
    good = [r for r in results if r["net"] > 0 and r["p"] < 0.05]
    print(f"  net R > 0 with p < 0.05 before correction: "
          f"{len(good)} of {len(results)} geometries")
    if good:
        m = len(results)
        ranked = sorted(good, key=lambda r: r["p"])
        survivors = [r for i, r in enumerate(ranked, start=1)
                     if r["p"] <= 0.05 * i / m]
        print(f"  surviving Benjamini-Hochberg across all {m}: "
              f"{len(survivors)}")
        for r in ranked:
            mark = "SURVIVES" if r in survivors else "rejected"
            print(f"    {mark:>8}  s{r['geom'][0]} r{r['geom'][1]}: "
                  f"net {r['net']:+.4f} p={r['p']:.4f} n={r['n']:,} "
                  f"AUC={r['auc']:.4f}")
    else:
        print("  none. No geometry produced a profitable out-of-fold result.")
    print("=" * 104)
    return 0


if __name__ == "__main__":
    sys.exit(main())
