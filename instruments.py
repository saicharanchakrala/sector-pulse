"""Discover every tradeable NSE instrument at a point in time.

Nothing in this module carries a hardcoded list of symbols. The universe is
fetched from NSE on each run and persisted with a timestamp, so a scan can
say exactly which instruments existed when it ran rather than which ones
somebody typed into a constant. That is not pedantry: the hand-written index
list this replaces was missing NIFTYFPI, so every scan requested it as an
equity and logged a 404, and no test could have caught it because the list
was the definition of truth.

What is actually obtainable, measured rather than assumed:

  equities        all listed symbols from NSE's EQUITY_L master (about 2,570)
  F&O underlying  216 names with front-month lot sizes, split into indices
                  and stocks by the master file's own section row
  F&O state       spot, aggregate open interest, OI change, and the split of
                  turnover between futures and options, for all 216 in one
                  request
  option chains   per underlying on demand, roughly 0.36s each
  futures         index futures per contract, plus a 20-name stock-futures
                  watch. There is no bulk per-contract stock futures feed:
                  the derivatives bhavcopy 404s on every published URL and
                  /api/quote-derivative has been withdrawn. Futures turnover
                  and OI per underlying is what remains, and it is enough to
                  tell whether a move is being driven in the derivatives
                  segment.

Everything returned by NSE is treated as data. Fields are coerced and
discarded when they do not parse; nothing here executes or trusts content.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime

import requests

import config

logger = logging.getLogger(__name__)

KIND_EQUITY = "EQUITY"
KIND_FO_INDEX = "FO_INDEX"
KIND_FO_STOCK = "FO_STOCK"

_SECTION_MARKER = "DERIVATIVES ON"

# Index underlyings, used only to label a futures contract FUTIDX vs FUTSTK.
# Kept as a set of names rather than a hardcoded universe: the scannable
# list still comes entirely from discovery.
_INDEX_NAMES = frozenset({"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY",
                          "NIFTYNXT50", "NIFTYFPI", "SENSEX", "BANKEX"})

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass(frozen=True)
class Instrument:
    """One tradeable name as NSE currently lists it."""

    symbol: str
    kind: str
    name: str = ""
    series: str = ""
    lot_size: "int | None" = None
    isin: str = ""

    @property
    def is_index(self) -> bool:
        """Whether this is an index rather than a company."""
        return self.kind == KIND_FO_INDEX

    @property
    def has_derivatives(self) -> bool:
        """Whether futures and options exist on this name."""
        return self.kind in (KIND_FO_INDEX, KIND_FO_STOCK)


@dataclass(frozen=True)
class FoState:
    """One underlying's derivatives state at the snapshot instant.

    Units are in the field names because NSE mixes them in one payload and
    summing the wrong pair is silent. Verified against the feed's own
    arithmetic: total == futures + options premium, to the paisa, in lakhs.
    optValue is options *notional* in rupees, a different quantity again -
    adding it to a lakh figure made every futures share read 0.0%.
    """

    symbol: str
    spot: float
    open_interest: float                 # contracts
    prev_open_interest: float            # contracts, prior session
    futures_turnover_lakh: float         # futValue
    options_premium_lakh: float          # premValue
    options_notional_rupees: float       # optValue
    total_turnover_lakh: float           # total, == futures + options premium
    volume: float                        # contracts traded

    @property
    def oi_change(self) -> float:
        """Absolute change in aggregate open interest, in contracts."""
        return self.open_interest - self.prev_open_interest

    @property
    def oi_change_pct(self) -> "float | None":
        """Percent change in open interest; matches the feed's own avgInOI."""
        if self.prev_open_interest <= 0.0:
            return None
        return (self.open_interest / self.prev_open_interest - 1.0) * 100.0

    @property
    def derivatives_turnover_rupees(self) -> float:
        """Futures plus options premium, converted from lakhs to rupees."""
        return self.total_turnover_lakh * 1e5

    @property
    def futures_share(self) -> "float | None":
        """Futures share of derivatives turnover, 0..1.

        A move carried in futures reads differently from one carried in
        options premium, so the mix is kept rather than summed away.
        """
        if self.total_turnover_lakh <= 0.0:
            return None
        return self.futures_turnover_lakh / self.total_turnover_lakh

    @property
    def totals_reconcile(self) -> bool:
        """Whether total still equals futures plus options premium.

        Guards the unit assumption above: if NSE redefines a column this
        goes False and the share becomes untrustworthy rather than wrong.
        """
        parts = self.futures_turnover_lakh + self.options_premium_lakh
        if self.total_turnover_lakh <= 0.0:
            return parts <= 0.0
        return abs(parts - self.total_turnover_lakh) <= 0.01 * max(
            1.0, self.total_turnover_lakh)


@dataclass(frozen=True)
class FuturesContract:
    """One futures contract, where a per-contract feed still answers."""

    symbol: str
    instrument_type: str
    expiry: str
    last_price: float
    open_interest: float
    change_pct: float


@dataclass
class Universe:
    """Everything discovered in one pass, with what failed recorded."""

    captured_at: datetime
    equities: list[Instrument] = field(default_factory=list)
    fo_indices: list[Instrument] = field(default_factory=list)
    fo_stocks: list[Instrument] = field(default_factory=list)
    fo_state: dict[str, FoState] = field(default_factory=dict)
    futures: list[FuturesContract] = field(default_factory=list)
    option_expiries: dict[str, list[str]] = field(default_factory=dict)
    gaps: list[str] = field(default_factory=list)

    @property
    def fo_underlyings(self) -> list[Instrument]:
        """Indices and stocks that carry derivatives."""
        return self.fo_indices + self.fo_stocks

    @property
    def scannable_equities(self) -> list[Instrument]:
        """Equities eligible for the intraday scan.

        Indices are excluded because they cannot be bought as equity, and
        that exclusion now comes from the discovered F&O master rather than
        from a maintained list of index names.
        """
        return [inst for inst in self.equities if not inst.is_index]

    def counts(self) -> dict[str, int]:
        """Headline counts for the discovery report."""
        return {
            "equities": len(self.equities),
            "fo_indices": len(self.fo_indices),
            "fo_stocks": len(self.fo_stocks),
            "fo_state": len(self.fo_state),
            "futures_contracts": len(self.futures),
            "option_expiries": len(self.option_expiries),
        }


def open_session() -> requests.Session:
    """A session carrying the cookies NSE hands out on its landing page."""
    session = requests.Session()
    session.headers.update(_HEADERS)
    try:
        session.get(config.NSE_BASE_URL, timeout=config.NSE_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        logger.warning("NSE bootstrap failed: %s", exc)
    return session


def _get(session: requests.Session, url: str, referer: str = "") -> "requests.Response | None":
    """One GET that returns None instead of raising."""
    headers = {"Referer": referer} if referer else {}
    try:
        response = session.get(url, timeout=config.NSE_TIMEOUT_SECONDS,
                               headers=headers)
    except requests.RequestException as exc:
        logger.warning("Request failed (%s): %s", url, exc)
        return None
    if response.status_code != 200:
        logger.warning("HTTP %d from %s", response.status_code, url)
        return None
    return response


def _number(value: object) -> float:
    """Coerce an NSE field to a float, defaulting to 0.0."""
    if value is None:
        return 0.0
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return 0.0


def parse_equity_master(text: str) -> list[Instrument]:
    """Parse NSE's listed-equity master CSV. Pure: no network."""
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 2:
        logger.warning("Equity master parsed to %d rows", len(rows))
        return []
    header = [cell.strip().upper() for cell in rows[0]]

    def column(*names: str) -> "int | None":
        for name in names:
            if name in header:
                return header.index(name)
        return None

    sym_col = column("SYMBOL")
    if sym_col is None:
        logger.warning("Equity master has no SYMBOL column: %s", header)
        return []
    name_col = column("NAME OF COMPANY")
    series_col = column("SERIES")
    isin_col = column("ISIN NUMBER", "ISIN")
    out: list[Instrument] = []
    seen: set[str] = set()
    for row in rows[1:]:
        if len(row) <= sym_col:
            continue
        symbol = row[sym_col].strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        out.append(Instrument(
            symbol=symbol,
            kind=KIND_EQUITY,
            name=row[name_col].strip() if name_col is not None and len(row) > name_col else "",
            series=row[series_col].strip().upper() if series_col is not None and len(row) > series_col else "",
            isin=row[isin_col].strip() if isin_col is not None and len(row) > isin_col else "",
        ))
    return out


def fetch_listed_equities(session: requests.Session) -> list[Instrument]:
    """Every symbol on NSE's listed-equity master; [] on failure."""
    response = _get(session, config.NSE_EQUITY_LIST_URL)
    return parse_equity_master(response.text) if response is not None else []


def _lot_from_row(row: list[str], symbol_col: int) -> "int | None":
    """Front-month lot size: the first positive number after the symbol."""
    for cell in row[symbol_col + 1:]:
        raw = cell.strip().replace(",", "")
        if not raw:
            continue
        try:
            size = int(float(raw))
        except ValueError:
            continue
        return size if size > 0 else None
    return None


def parse_fo_master(text: str) -> tuple[list[Instrument], list[Instrument]]:
    """F&O indices and stocks with lot sizes, split by the file's section row.

    The master file separates index derivatives from single-stock
    derivatives with a "Derivatives on Individual Securities" row. Using
    that boundary means the index list is whatever NSE says it is today,
    which is how NIFTYFPI stopped being silently treated as an equity.
    Pure: no network.
    """
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return [], []
    header = [cell.strip().upper() for cell in rows[0]]
    symbol_col = header.index("SYMBOL") if "SYMBOL" in header else 1
    boundary = None
    for index, row in enumerate(rows):
        if row and _SECTION_MARKER in row[0].strip().upper():
            boundary = index
            break
    if boundary is None:
        logger.warning("No index/stock section row in the F&O master; "
                       "treating every underlying as a stock")
        boundary = 0

    def collect(subset: list[list[str]], kind: str) -> list[Instrument]:
        found: list[Instrument] = []
        seen: set[str] = set()
        for row in subset:
            if len(row) <= symbol_col:
                continue
            symbol = row[symbol_col].strip().upper()
            if (not symbol or " " in symbol or symbol in seen
                    or symbol in ("SYMBOL", "UNDERLYING")):
                continue
            seen.add(symbol)
            found.append(Instrument(
                symbol=symbol, kind=kind,
                name=row[0].strip() if row else "",
                lot_size=_lot_from_row(row, symbol_col)))
        return found

    indices = collect(rows[1:boundary], KIND_FO_INDEX)
    stocks = collect(rows[boundary + 1:], KIND_FO_STOCK)
    return indices, stocks


def fetch_fo_master(session: requests.Session
                    ) -> tuple[list[Instrument], list[Instrument]]:
    """Download and parse the F&O master; ([], []) on failure."""
    response = _get(session, config.FO_MKTLOTS_URL)
    return parse_fo_master(response.text) if response is not None else ([], [])


def parse_fo_state(rows: list[dict]) -> dict[str, FoState]:
    """Parse the OI snapshot rows into FoState records. Pure: no network."""
    out: dict[str, FoState] = {}
    for row in rows:
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        out[symbol] = FoState(
            symbol=symbol,
            spot=_number(row.get("underlyingValue")),
            open_interest=_number(row.get("latestOI")),
            prev_open_interest=_number(row.get("prevOI")),
            futures_turnover_lakh=_number(row.get("futValue")),
            options_premium_lakh=_number(row.get("premValue")),
            options_notional_rupees=_number(row.get("optValue")),
            total_turnover_lakh=_number(row.get("total")),
            volume=_number(row.get("volume")),
        )
    broken = [s.symbol for s in out.values() if not s.totals_reconcile]
    if broken:
        logger.warning("Turnover columns stopped reconciling for %d underlying(s) "
                       "(e.g. %s); the futures/options split may have changed "
                       "units upstream", len(broken), ", ".join(broken[:5]))
    return out


def fetch_fo_state(session: requests.Session) -> dict[str, FoState]:
    """Derivatives state for every F&O underlying, in one request.

    This is the cross-segment view: spot alongside aggregate open interest
    and the futures-versus-options turnover split, for all 216 underlyings
    at once. It is the only bulk derivatives feed still answering, so it is
    also the only practical way to let F&O activity inform an equity scan.
    """
    response = _get(session, config.NSE_BASE_URL + config.NSE_OI_SPURTS_PATH,
                    referer=config.NSE_BASE_URL + "/market-data/oi-spurts")
    if response is None:
        return {}
    try:
        rows = response.json().get("data") or []
    except ValueError as exc:
        logger.warning("OI snapshot was not JSON: %s", exc)
        return {}
    return parse_fo_state(rows)


def fetch_futures(session: requests.Session) -> list[FuturesContract]:
    """Per-contract futures from whichever live keys still answer."""
    out: list[FuturesContract] = []
    for key in config.NSE_FUTURES_INDEX_KEYS:
        url = f"{config.NSE_BASE_URL}{config.NSE_LIVE_DERIVATIVES_PATH}?index={key}"
        response = _get(session, url, referer=config.NSE_BASE_URL
                        + "/market-data/equity-derivatives-watch")
        if response is None:
            continue
        try:
            rows = response.json().get("data") or []
        except ValueError:
            continue
        for row in rows:
            meta = row.get("metadata") or row
            symbol = str(meta.get("identifier") or meta.get("contract") or "").strip()
            if not symbol:
                continue
            out.append(FuturesContract(
                symbol=symbol,
                instrument_type=str(meta.get("instrumentType") or ""),
                expiry=str(meta.get("expiryDate") or ""),
                last_price=_number(meta.get("lastPrice")),
                open_interest=_number(meta.get("openInterest")),
                change_pct=_number(meta.get("pChange") or meta.get("change")),
            ))
    return out


def fetch_futures_from_kite() -> list[FuturesContract]:
    """Every NSE futures contract from Zerodha's public instrument master.

    This closes a gap that looked closed for good. NSE's own endpoints serve
    no bulk per-contract futures data - the derivatives bhavcopy 404s on
    every published URL and /api/quote-derivative has been withdrawn - so
    futures activity had to be inferred from per-underlying turnover. Kite
    publishes 647 contracts across 216 underlyings in one unauthenticated
    CSV, with lot sizes that cross-check exactly against NSE's own
    market-lot file on all 216.

    Prices are not included, which is honest rather than limiting: the
    master describes what exists, and what it last traded at needs a
    session.
    """
    try:
        import kite_instruments
    except ImportError as exc:
        logger.warning("kite_instruments unavailable: %s", exc)
        return []
    contracts = kite_instruments.fetch_master()
    if not contracts:
        return []
    out: list[FuturesContract] = []
    for contract in kite_instruments.nse_futures(contracts):
        out.append(FuturesContract(
            symbol=contract.tradingsymbol,
            instrument_type="FUTSTK" if contract.name not in _INDEX_NAMES
                            else "FUTIDX",
            expiry=contract.expiry.isoformat() if contract.expiry else "",
            last_price=0.0,      # the master carries no prices
            open_interest=0.0,
            change_pct=0.0,
        ))
    return out


def fetch_option_expiries(session: requests.Session, symbols: list[str],
                          pause: float = config.NSE_REQUEST_PAUSE_SECONDS
                          ) -> dict[str, list[str]]:
    """Available option expiries per underlying, paced to avoid throttling."""
    out: dict[str, list[str]] = {}
    for symbol in symbols:
        url = (f"{config.NSE_BASE_URL}{config.NSE_CONTRACT_INFO_PATH}"
               f"?symbol={requests.utils.quote(symbol)}")
        response = _get(session, url,
                        referer=config.NSE_BASE_URL + "/option-chain")
        if response is not None:
            try:
                dates = response.json().get("expiryDates") or []
            except ValueError:
                dates = []
            if dates:
                out[symbol] = [str(value) for value in dates]
        if pause > 0:
            time.sleep(pause)
    return out


def discover(with_expiries: bool = False,
             session: "requests.Session | None" = None) -> Universe:
    """Fetch the whole instrument universe as it stands right now."""
    live = open_session() if session is None else session
    universe = Universe(captured_at=datetime.now().astimezone())

    universe.equities = fetch_listed_equities(live)
    if not universe.equities:
        universe.gaps.append("listed-equity master unavailable")

    indices, stocks = fetch_fo_master(live)
    universe.fo_indices, universe.fo_stocks = indices, stocks
    if not stocks:
        universe.gaps.append("F&O master unavailable")

    universe.fo_state = fetch_fo_state(live)
    if not universe.fo_state:
        universe.gaps.append("F&O open-interest snapshot unavailable")

    # Zerodha publishes the whole contract master with no auth, and it
    # contains what NSE will not give: every futures contract with its lot
    # size and instrument token. Preferred over NSE's live-derivatives keys,
    # which answer for index futures and a 20-name watch only.
    universe.futures = fetch_futures_from_kite()
    kite_sourced = bool(universe.futures)
    if not kite_sourced:
        universe.futures = fetch_futures(live)
        if universe.futures:
            universe.gaps.append(
                f"Kite's instrument master was unreachable, so futures fell "
                f"back to NSE's live keys and cover only "
                f"{len(universe.futures)} contracts rather than the full list")
        else:
            universe.gaps.append("no futures feed answered at all")

    if with_expiries:
        wanted = [inst.symbol for inst in universe.fo_underlyings]
        universe.option_expiries = fetch_option_expiries(live, wanted)
        missing = len(wanted) - len(universe.option_expiries)
        if missing > 0:
            universe.gaps.append(f"option expiries missing for {missing} underlyings")
    return universe


def snapshot_path(root=None):
    """The single file holding the current instrument universe."""
    base = config.INSTRUMENT_DIR if root is None else root
    return base / config.INSTRUMENT_FILE.name


def _legacy_snapshots(root=None) -> list:
    """Timestamped files from when this store kept a history."""
    base = config.INSTRUMENT_DIR if root is None else root
    try:
        return sorted(base.glob("universe_*.json"))
    except OSError:
        return []


def save(universe: Universe, root=None):
    """Overwrite the current universe file; returns the path, or None.

    Written to a temporary file and moved into place, because this is now
    the only copy: a process killed halfway through a direct write would
    leave a truncated file and no history to fall back on. os.replace is
    atomic on Windows and POSIX alike.
    """
    base = config.INSTRUMENT_DIR if root is None else root
    target = snapshot_path(base)
    payload = {
        "captured_at": universe.captured_at.isoformat(),
        "equities": [asdict(i) for i in universe.equities],
        "fo_indices": [asdict(i) for i in universe.fo_indices],
        "fo_stocks": [asdict(i) for i in universe.fo_stocks],
        "fo_state": {k: asdict(v) for k, v in universe.fo_state.items()},
        "futures": [asdict(f) for f in universe.futures],
        "option_expiries": universe.option_expiries,
        "gaps": universe.gaps,
    }
    staging = target.with_suffix(".json.tmp")
    try:
        base.mkdir(parents=True, exist_ok=True)
        with staging.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1)
        os.replace(staging, target)
    except OSError as exc:
        logger.warning("Could not write %s: %s", target, exc)
        try:
            staging.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    for stale in _legacy_snapshots(base):
        try:
            stale.unlink()
            logger.info("Removed superseded snapshot %s", stale.name)
        except OSError as exc:
            logger.warning("Could not remove %s: %s", stale.name, exc)
    return target


def _read(path) -> "Universe | None":
    """Parse one universe file, or None when it cannot be read."""
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None
    return _to_universe(payload)


def load_latest(root=None) -> "Universe | None":
    """The current universe, or None when nothing has been synced.

    Falls back to the newest timestamped file so an install that predates
    the single-file store keeps working until its next sync.
    """
    base = config.INSTRUMENT_DIR if root is None else root
    current = snapshot_path(base)
    if current.exists():
        return _read(current)
    legacy = _legacy_snapshots(base)
    if not legacy:
        return None
    logger.info("Reading a superseded timestamped snapshot; the next sync "
                "will replace it with a single current file")
    return _read(legacy[-1])


def _to_universe(payload: dict) -> Universe:
    """Rebuild a Universe from its stored JSON."""
    stamp = datetime.fromisoformat(payload["captured_at"])
    if stamp.tzinfo is None:
        # A hand-edited or legacy snapshot without an offset would make
        # `captured_at <= now` raise and take the scan down with it.
        stamp = stamp.astimezone()
    return Universe(
        captured_at=stamp,
        equities=[Instrument(**row) for row in payload.get("equities", [])],
        fo_indices=[Instrument(**row) for row in payload.get("fo_indices", [])],
        fo_stocks=[Instrument(**row) for row in payload.get("fo_stocks", [])],
        fo_state={k: FoState(**v)
                  for k, v in (payload.get("fo_state") or {}).items()},
        futures=[FuturesContract(**row) for row in payload.get("futures", [])],
        option_expiries=payload.get("option_expiries") or {},
        gaps=payload.get("gaps") or [],
    )


def to_ticker(symbol: str, suffix: str = ".NS") -> str:
    """Map a discovered NSE symbol to its yfinance ticker.

    A rule, not a table: every listed equity is SYMBOL.NS. Indices are not
    mapped at all, because they cannot be bought as equity and the scan
    excludes them.
    """
    cleaned = symbol.strip().upper()
    if not cleaned or cleaned.startswith("^") or "." in cleaned:
        return cleaned
    return f"{cleaned}{suffix}"
