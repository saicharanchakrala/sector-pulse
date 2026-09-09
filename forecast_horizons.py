"""Short, mid and long horizon models, evaluated against holding the index.

The label is already excess of NIFTY 50 and net of delivery costs, so a
positive result here means "beat the index after charges", not "made money
because Indian equities rose". That distinction is the whole point: a
raw-return label would have handed back the market's decade as though it
were the model's skill.

THE EMBARGO IS SIZED TO THE LABEL. A 252-session label opened on the last
training date does not resolve until 252 sessions later, so training up to
date T and testing from T+1 would let the model see outcomes it could not
have known. Each horizon therefore embargoes its own length. This is
expensive - it is why the long horizon gets few test periods - and skipping
it is the most common way published backtests leak.

Sample-size honesty is printed per horizon: the number of NON-OVERLAPPING
periods, which is the real sample size. Rows overlap heavily, so a long
horizon showing 90,000 rows still rests on roughly ten independent
observations of the future. Read the long result as weak evidence whatever
it says.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

import forecast_stats

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
# Which dataset to score. Pass a filename as argv[1]; defaults to the
# survivorship-corrected point-in-time build.
DATA = OUT_DIR / (sys.argv[1] if len(sys.argv) > 1 else "pit_dataset.parquet")

HORIZONS = {"short": 10, "mid": 63, "long": 252}
# The embargo is stated in TRADING SESSIONS but applied as POSITIONS in the
# sorted list of unique dates, so the two must be reconciled - and the
# conversion factor is a property of the dataset, not a constant.
#
# This was hardcoded twice and wrong twice. First 5, assuming the date list
# was sparse. Then 0.974, measured on a build whose per-symbol sampling
# offsets made the union of dates dense. The point-in-time build samples
# from a COMMON date index instead, so its real spacing is one date per five
# sessions - and a 252-session embargo became 1,258, which left the long
# horizon with no folds at all.
#
# So it is now measured from the dates themselves every run. Over-embargoing
# only wastes data; under-embargoing leaks the future, so the ceiling is
# deliberate.
def embargo_positions(dates: list, span_sessions: int) -> int:
    """How many positions in `dates` cover `span_sessions` trading days."""
    if len(dates) < 3:
        return span_sessions
    gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
    gaps = [g for g in gaps if g > 0]
    if not gaps:
        return span_sessions
    calendar_per_position = sorted(gaps)[len(gaps) // 2]
    # 252 trading sessions to 365 calendar days.
    span_calendar = span_sessions * 365.0 / 252.0
    return max(1, int(-(-span_calendar // calendar_per_position)))
N_FOLDS = 3
CALIB_FRACTION = 0.15
SEED = 20260909
# 30, not 10. The exact permutation p is (1+beats)/(1+shuffles), so ten
# shuffles cannot produce a p below 1/11 = 0.091 - it is arithmetically
# incapable of showing significance at 0.05 however strong the effect.
# Thirty gives a floor of 1/31 = 0.032, which can.
PERMUTATIONS = 30
TOP_FRACTION = 0.10          # size of the "buy" basket, per date

BASE_FEATURES = [
    "mom5", "mom21", "mom63", "mom126", "mom252", "mom252_ex21",
    "rel_mom21", "rel_mom63", "rel_mom126", "rel_mom252",
    "vol252", "vol21", "vol_ratio", "pos_52w", "drawdown_from_high",
    "volume_ratio", "turnover_log", "price_log",
    "bench_mom63", "bench_mom252",
]
XS_FEATURES = [
    "xs_mom21", "xs_mom63", "xs_mom252", "xs_mom252_ex21", "xs_rel_mom63",
    "xs_rel_mom252", "xs_vol252", "xs_vol_ratio", "xs_pos_52w",
    "xs_volume_ratio", "xs_turnover_log", "xs_drawdown_from_high",
]


def date_bootstrap(values: np.ndarray, dates: np.ndarray,
                   draws: int = 2000) -> tuple:
    """(mean, lo, hi, one-sided p) resampling whole DATES as blocks.

    Every name selected on the same date shares that date's market move, so
    the date is the unit of independence, not the row.

    This was its own copy of the routine, which meant the figures this
    project publishes were produced by code with no tests while the tested
    implementation in forecast_stats had no callers. Verified identical to
    zero absolute difference on all four returned values across 40
    randomised trials before being replaced by the call.
    """
    return forecast_stats.block_bootstrap(values, dates, draws=draws,
                                          seed=SEED)


def folds_for(dates: list, embargo_sessions: int) -> list:
    """Expanding-window folds, embargo equal to the label horizon.

    embargo_sessions is in trading sessions; the gap is converted to
    positions using the dataset's own date spacing.
    """
    embargo = embargo_positions(dates, embargo_sessions)
    n = len(dates)
    start = int(n * 0.45)
    remaining = n - start - embargo
    if remaining <= 20:
        return []
    step = max(20, remaining // N_FOLDS)
    out = []
    for k in range(N_FOLDS):
        train_end = start + k * step
        test_start = train_end + embargo
        test_end = min(n, test_start + step)
        if test_start >= n or test_end <= test_start:
            continue
        out.append((dates[:train_end], dates[test_start:test_end]))
    return out


def fit_and_calibrate(train: pd.DataFrame, features: list, label: str):
    """Classifier plus isotonic calibration on a held-out tail of train."""
    dates = sorted(train["date"].unique())
    if len(dates) < 40:
        return None, None
    split = int(len(dates) * (1 - CALIB_FRACTION))
    fit_dates, calib_dates = set(dates[:split]), set(dates[split:])
    fit_part = train[train["date"].isin(fit_dates)]
    calib_part = train[train["date"].isin(calib_dates)]
    if len(fit_part) < 1000 or len(calib_part) < 300:
        return None, None
    if fit_part[label].nunique() < 2 or calib_part[label].nunique() < 2:
        return None, None
    model = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.05, max_depth=3,
        min_samples_leaf=100, l2_regularization=2.0,
        early_stopping=True, validation_fraction=0.15, random_state=SEED)
    model.fit(fit_part[features].to_numpy(dtype=np.float32),
              fit_part[label].to_numpy())
    raw = model.predict_proba(
        calib_part[features].to_numpy(dtype=np.float32))[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw, calib_part[label].to_numpy())
    return model, iso


def run_horizon(data: pd.DataFrame, name: str, span: int, features: list,
                shuffle: bool = False, rng=None) -> pd.DataFrame:
    """Out-of-fold calibrated predictions for one horizon."""
    target = f"y_{name}"
    frame = data.dropna(subset=[target]).copy()
    if frame.empty:
        return pd.DataFrame()
    frame["label"] = (frame[target] > 0).astype(int)
    dates = sorted(frame["date"].unique())
    out = []
    for train_dates, test_dates in folds_for(dates, span):
        train = frame[frame["date"].isin(set(train_dates))]
        test = frame[frame["date"].isin(set(test_dates))]
        if train.empty or test.empty:
            continue
        if shuffle:
            train = train.copy()
            train["label"] = (train.groupby("date")["label"]
                              .transform(lambda s: rng.permutation(s.to_numpy())))
        model, iso = fit_and_calibrate(train, features, "label")
        if model is None:
            continue
        block = test[["symbol", "date", target, f"mdd_{name}",
                      f"raw_{name}", f"bench_{name}"]].copy()
        raw = model.predict_proba(
            test[features].to_numpy(dtype=np.float32))[:, 1]
        block["p"] = iso.predict(raw)
        block["label"] = test["label"].to_numpy()
        out.append(block)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def evaluate(frame: pd.DataFrame, name: str, label: str) -> dict:
    """Excess return of the top-decile basket, versus every alternative."""
    target = f"y_{name}"
    if frame.empty:
        return {}
    # The basket: highest calibrated probability within each date. This is
    # a relative choice, which is how factor strategies actually trade -
    # an absolute probability threshold would load up in bull regimes and
    # hold nothing in bear ones, confounding timing with selection.
    frame = frame.copy()
    frame["rank"] = frame.groupby("date")["p"].rank(pct=True)
    picked = frame[frame["rank"] >= 1.0 - TOP_FRACTION]
    if picked.empty:
        return {}
    obs, lo, hi, p = date_bootstrap(picked[target].to_numpy(),
                                    picked["date"].to_numpy())
    everything, e_lo, e_hi, _ = date_bootstrap(frame[target].to_numpy(),
                                               frame["date"].to_numpy())
    auc = (roc_auc_score(frame["label"], frame["p"])
           if frame["label"].nunique() > 1 else float("nan"))
    return {
        "label": label, "rows": len(frame), "picked": len(picked),
        "dates": frame["date"].nunique(), "auc": auc,
        "excess": obs, "lo": lo, "hi": hi, "p": p,
        "all_names": everything, "all_lo": e_lo, "all_hi": e_hi,
        "spread": obs - everything,
        "mdd": float(picked[f"mdd_{name}"].mean()),
        "hit": float((picked[target] > 0).mean()),
    }


def main() -> int:
    if not DATA.exists():
        print(f"missing {DATA} - run build_daily.py first")
        return 1
    data = pd.read_parquet(DATA)
    features = [f for f in BASE_FEATURES + XS_FEATURES if f in data.columns]
    print(f"rows {len(data):,} | dates {data.date.nunique():,} | "
          f"symbols {data.symbol.nunique()} | features {len(features)}")
    total_sessions = data["date"].nunique()

    # Dates divided by the measured density recovers calendar length.
    all_dates = sorted(data["date"].unique())
    gaps = [(all_dates[i + 1] - all_dates[i]).days
            for i in range(len(all_dates) - 1)]
    gaps = [g for g in gaps if g > 0]
    spacing = sorted(gaps)[len(gaps) // 2] if gaps else 1
    calendar_sessions = int(total_sessions * spacing * 252.0 / 365.0)
    print(f"sampled dates {total_sessions:,} -> ~{calendar_sessions:,} "
          f"trading sessions of calendar covered")
    summary = []
    for name, span in HORIZONS.items():
        print()
        print("=" * 104)
        independent = calendar_sessions // span
        print(f"HORIZON {name.upper()}  ({span} sessions)   "
              f"embargo {embargo_positions(all_dates, span)} dates "
              f"~ {span} sessions   "
              f"non-overlapping periods in sample: ~{independent}")
        print("=" * 104)
        frame = run_horizon(data, name, span, features)
        if frame.empty:
            print("  no usable folds after the embargo - horizon too long "
                  "for the history available")
            summary.append({"label": name, "excess": float("nan"),
                            "p": float("nan"), "note": "no folds"})
            continue
        result = evaluate(frame, name, name)
        if not result:
            print("  nothing selected")
            continue
        print(f"  out-of-fold rows {result['rows']:,} over "
              f"{result['dates']:,} dates, AUC {result['auc']:.4f}")
        print(f"  top-{int(TOP_FRACTION * 100)}% basket: "
              f"{result['picked']:,} positions")
        print(f"    excess over index, net of cost : "
              f"{result['excess']:+.3f}%  "
              f"[{result['lo']:+.3f}, {result['hi']:+.3f}]  p={result['p']:.4f}")
        print(f"    same for ALL names (equal weight): "
              f"{result['all_names']:+.3f}%  "
              f"[{result['all_lo']:+.3f}, {result['all_hi']:+.3f}]")
        print(f"    selection spread (basket - all)  : "
              f"{result['spread']:+.3f}%")
        print(f"    share of positions beating index : {result['hit']:.3f}")
        print(f"    worst forward close vs entry     : {result['mdd']:.2f}%")

        rng = np.random.default_rng(SEED)
        nulls = []
        for k in range(PERMUTATIONS):
            shuffled = run_horizon(data, name, span, features,
                                   shuffle=True, rng=rng)
            if shuffled.empty:
                continue
            null_result = evaluate(shuffled, name, name)
            if null_result:
                nulls.append(null_result["excess"])
        if nulls:
            arr = np.array(nulls)
            beats = int((arr >= result["excess"]).sum())
            beat = (1 + beats) / (1 + arr.size)
            print(f"  permutation null ({len(nulls)} shuffles): "
                  f"{arr.mean():+.3f}% +/- {arr.std():.3f}  "
                  f"range [{arr.min():+.3f}, {arr.max():+.3f}]")
            print(f"  permutation p: {beat:.4f}   "
                  f"excess over null: {result['excess'] - arr.mean():+.3f}%")
            result["perm_p"] = beat
        summary.append(result)

    print()
    print("=" * 104)
    print("SUMMARY  (positive excess = beat holding the index, after charges)")
    print("=" * 104)
    print("Excess is PER HOLDING PERIOD, and the periods differ, so the rows "
          "are NOT")
    print("directly comparable: 10 sessions is about 25 round trips a year "
          "while 252")
    print("is one, and each row is charged for a single round trip. The "
          "trips/yr column")
    print("is the conversion factor. Selection spread is the like-for-like "
          "number.")
    print()
    print(f"{'horizon':>8} {'excess %':>10} {'95% CI':>22} {'boot p':>8} "
          f"{'perm p':>8} {'spread':>8} {'mdd %':>8} {'trips/yr':>9}")
    for row in summary:
        if not row or row.get("excess") != row.get("excess"):
            print(f"{row.get('label', '?'):>8} {'n/a':>10}  "
                  f"{row.get('note', '')}")
            continue
        trips = 252.0 / HORIZONS[row["label"]]["sessions"]
        print(f"{row['label']:>8} {row['excess']:>+10.3f} "
              f"[{row['lo']:+9.3f},{row['hi']:+9.3f}] {row['p']:>8.4f} "
              f"{row.get('perm_p', float('nan')):>8.4f} "
              f"{row['spread']:>+8.3f} {row['mdd']:>8.2f} {trips:>9.1f}")
    print()
    print("'worst forward close' is the lowest CLOSE over the horizon relative")
    print("to entry, not a true intraday drawdown - highs and lows are not")
    print("loaded - so it UNDERSTATES how far underwater a position went.")
    print("Read the LONG row with the most suspicion: its rows overlap so")
    print("heavily that the effective sample is roughly the non-overlapping")
    print("period count printed above, not the row count.")
    print("Caveat not controlled here: excess of the index still carries beta.")
    print("A high-beta basket beats the index in a rising decade without any")
    print("skill. Beta-neutralising is the next refinement, not done yet.")
    print("=" * 104)
    return 0


if __name__ == "__main__":
    sys.exit(main())
