"""Fetch ~10 years of daily bars for the whole NSE mainboard, not just the
210 names that happen to be in today's F&O list.

This is the survivorship fix. The previous universe was today's F&O
membership, so every name that was liquid in 2018 and has since shrunk out
of the list was missing from the backtest - and their absence is not random,
it is precisely the losers. An equal-weight basket of that universe "beat"
the index by 20.66% per year, which is what selecting a decade of winners
looks like.

The replacement: fetch a broad pool once, then define the universe AT EACH
DATE by trailing turnover computed from data available on that date. A name
that falls out of the top ranks in 2020 simply stops appearing from 2020,
which is what a point-in-time universe means.

RESIDUAL BIAS, stated plainly because this does not eliminate it: Kite's
instrument master lists what is listed TODAY. A company fully delisted -
acquired, gone private, or thrown off by the exchange - is absent from the
master and therefore from this pool too. So the fix removes the "fell out of
the F&O list" bias but not the "vanished entirely" bias. The direction is
known (results still flattered) and the remaining gap should be treated as
an upper bound on any measured edge, not as noise.
"""
from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

import kite_instruments as ki      # noqa: E402
import market_source as ms         # noqa: E402

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "bar_cache"
OUT_DIR = ROOT / "forecast_cache"
OUT_DIR.mkdir(exist_ok=True)


START = date(2016, 1, 1)
LOG = OUT_DIR / "broad_pool_missing.txt"
# Funds, not companies. A stock-selection universe should not contain them,
# and they are easier to exclude by name here than to detect later.
FUND_PATTERN = re.compile(
    r"(BEES|ETF|IETF|GSEC|SDL|BHARATBOND|LIQUID|GILT|MAFANG|MON100|"
    r"SILVER|GOLD|MOM100|MOM50|LOWVOL|QUAL30|VAL30|CONSUMBEES|"
    r"EVINDIA|MIDSELIETF|TOP100CASE|AXISNIFTY|SETFNIF|UTINIFTETF)")


# Operating companies whose names collide with the fund pattern. Audited by
# listing every symbol the pattern excluded and checking which lacked a fund
# marker (BEES/ETF/INAV/LIQUID/BOND): these four are real businesses -
# jewellery exporters and an IT services firm - not gold or silver funds.
NOT_FUNDS = {"GOLDIAM", "SHANTIGOLD", "SKYGOLD", "SILVERTUC"}


def pool() -> list:
    """Plain-series NSE equities, funds removed."""
    rows = ki.fetch_master()
    names = []
    for contract in ki.nse_equities(rows):
        symbol = contract.tradingsymbol
        if "-" in symbol:
            continue                      # SME, govt sec, bonds, rights
        if contract.instrument_type != "EQ":
            continue
        if FUND_PATTERN.search(symbol) and symbol not in NOT_FUNDS:
            continue
        names.append(symbol)
    return sorted(set(names))


def already_cached(symbol: str) -> bool:
    """Whether a deep daily history is already on disk for this symbol."""
    safe = symbol.replace("/", "_").replace(":", "_").replace(" ", "_")
    return any(CACHE.glob(f"kite__{safe}__day__2016*.parquet"))


def main() -> int:
    symbols = pool()
    end = date.today()
    todo = [s for s in symbols if not already_cached(s)]
    print(f"mainboard pool: {len(symbols):,} symbols")
    print(f"already cached: {len(symbols) - len(todo):,}")
    print(f"to fetch      : {len(todo):,}  "
          f"(~{len(todo) * 2 / 3 / 60:.0f} min at 3 req/s, 2 chunks each)")
    if not ms.session_available():
        print("FATAL: no Kite session. Run: .venv\\Scripts\\python -m kite_login")
        return 1

    ok = empty = errors = 0
    missing = []
    for index, symbol in enumerate(todo, start=1):
        try:
            got = ms.bars([symbol], START, end, interval="day")
        except ms.NoSession as exc:
            print(f"SESSION LOST after {ok} fetched: {exc}")
            break
        except Exception as exc:
            errors += 1
            if errors <= 10:
                print(f"  {symbol}: {type(exc).__name__} {exc}")
            continue
        frame = got.get(symbol)
        if frame is None or frame.empty:
            empty += 1
            missing.append(symbol)
            continue
        ok += 1
        if index % 100 == 0:
            print(f"  {index}/{len(todo)}  ok={ok} empty={empty} "
                  f"err={errors}  last={symbol} ({len(frame)} bars)",
                  flush=True)
    print(f"\ncomplete: {ok} fetched, {empty} returned nothing, "
          f"{errors} errored")
    deep = list(CACHE.glob("kite__*__day__2016*.parquet"))
    print(f"deep daily files on disk: {len(deep):,}")
    try:
        LOG.write_text("\n".join(missing), encoding="utf-8")
        print(f"symbols with no data listed in {LOG.name}")
    except Exception as exc:
        print(f"could not write log: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
