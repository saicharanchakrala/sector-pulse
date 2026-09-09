"""Tests for the Zerodha modules: instrument master and the tick protocol.

Every test here is offline. That is essential for the tick parsers in
particular: Kite's wire format is not self-describing, because a packet's
MEANING comes from its byte LENGTH, and prices arrive as integers whose
divisor depends on the segment encoded in the instrument token. A misread
produces plausible wrong prices rather than an error, so the format is
pinned against hand-built frames instead of against a live market.

Run with: .venv\\Scripts\\python -m pytest test_kite.py -q
"""
from __future__ import annotations

import asyncio
import json
import struct
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest
import websockets

import kite_instruments as ki
import kite_ticker as kt
import tick_recorder as tr

# Token low bytes select the segment: 1 NSE cash, 2 NFO, 9 indices.
RELIANCE = 738561
RELIANCE_FUT = 17606914
NIFTY_INDEX = 256265

_MASTER = (
    "instrument_token,exchange_token,tradingsymbol,name,last_price,expiry,"
    "strike,tick_size,lot_size,instrument_type,segment,exchange\n"
    "738561,2885,RELIANCE,RELIANCE,0,,0,0.05,1,EQ,NSE,NSE\n"
    "17606914,68777,RELIANCE26SEPFUT,RELIANCE,0,2026-09-29,0,0.1,500,FUT,"
    "NFO-FUT,NFO\n"
    "12540674,48987,RELIANCE26OCTFUT,RELIANCE,0,2026-10-27,0,0.1,500,FUT,"
    "NFO-FUT,NFO\n"
    "16035586,62639,RELIANCE26SEP1300CE,RELIANCE,0,2026-09-29,1300,0.05,500,"
    "CE,NFO-OPT,NFO\n"
    "16035842,62640,RELIANCE26SEP1300PE,RELIANCE,0,2026-09-29,1300,0.05,500,"
    "PE,NFO-OPT,NFO\n"
    "256265,1001,NIFTY 50,NIFTY 50,0,,0,0.05,0,EQ,INDICES,NSE\n"
)


# --- instrument master ---------------------------------------------------

def test_master_parses_every_instrument_kind() -> None:
    rows = ki.parse_master(_MASTER)
    assert len(rows) == 6
    by_symbol = {c.tradingsymbol: c for c in rows}
    assert by_symbol["RELIANCE26SEPFUT"].is_future
    assert by_symbol["RELIANCE26SEP1300CE"].is_option
    assert not by_symbol["RELIANCE"].is_future
    assert by_symbol["RELIANCE26SEP1300CE"].strike == pytest.approx(1300.0)
    assert by_symbol["RELIANCE26SEPFUT"].expiry == date(2026, 9, 29)
    assert by_symbol["RELIANCE"].expiry is None


def test_futures_are_separated_from_options_and_cash() -> None:
    rows = ki.parse_master(_MASTER)
    assert [c.tradingsymbol for c in ki.nse_futures(rows)] == [
        "RELIANCE26SEPFUT", "RELIANCE26OCTFUT"]
    assert len(ki.nse_options(rows)) == 2
    assert [c.tradingsymbol for c in ki.nse_equities(rows)] == ["RELIANCE"]


def test_options_can_be_filtered_by_underlying() -> None:
    rows = ki.parse_master(_MASTER)
    assert len(ki.nse_options(rows, "RELIANCE")) == 2
    assert ki.nse_options(rows, "TCS") == []


def test_lot_size_comes_from_the_nearest_expiry() -> None:
    # Both futures carry 500; the point is that the nearest is chosen rather
    # than whichever row happened to come last.
    rows = ki.parse_master(_MASTER)
    assert ki.lot_sizes(rows) == {"RELIANCE": 500}


def test_expiries_and_strikes_are_sorted_and_deduplicated() -> None:
    rows = ki.parse_master(_MASTER)
    assert ki.expiries(rows, "RELIANCE") == [date(2026, 9, 29)]
    assert ki.strikes(rows, "RELIANCE") == [1300.0]
    assert ki.strikes(rows, "RELIANCE", date(2026, 10, 27)) == []


def test_a_malformed_row_is_skipped_not_fatal() -> None:
    # The master is third-party data; one bad row must not lose the rest.
    broken = _MASTER + "notanumber,,BADROW,BAD,0,,0,0,0,EQ,NSE,NSE\n"
    assert len(ki.parse_master(broken)) == 6
    assert ki.parse_master("") == []
    assert ki.parse_master("garbage\n") == []


def test_summary_counts_what_the_report_prints() -> None:
    summary = ki.summary(ki.parse_master(_MASTER))
    assert summary["nse_futures"] == 2
    assert summary["nse_options"] == 2
    assert summary["calls"] == 1 and summary["puts"] == 1
    assert summary["nse_equities"] == 1
    assert summary["fo_underlyings"] == 1


# --- tick protocol: divisors --------------------------------------------

def test_divisor_is_derived_from_the_token_segment() -> None:
    # Getting this wrong yields prices off by a factor of a hundred, with no
    # error, so it is derived from the token rather than assumed.
    assert kt.divisor_for(RELIANCE) == 100.0
    assert kt.divisor_for(RELIANCE_FUT) == 100.0
    assert kt.divisor_for(0x0300 | 3) == 10_000_000.0     # currency
    assert kt.divisor_for(0x0600 | 6) == 10_000.0         # BSE currency


# --- tick protocol: framing ---------------------------------------------

def _frame(*payloads: bytes) -> bytes:
    """Build one binary frame exactly as Kite sends it."""
    out = struct.pack(">H", len(payloads))
    for payload in payloads:
        out += struct.pack(">H", len(payload)) + payload
    return out


def _ltp(token: int, paise: int) -> bytes:
    return struct.pack(">ii", token, paise)


# Prices are signed on the wire; quantities, volume and open interest are
# UNSIGNED. Packing them signed cannot even represent the values that expose
# the wrap bug, so the format string matters to the test as much as to the code.
_QUOTE_FMT = ">iiIiIIIiiii"


def _equity_quote(token: int, ltp: int, open_: int, high: int, low: int,
                  close: int, volume: int) -> bytes:
    return struct.pack(_QUOTE_FMT, token, ltp, 25, ltp, volume, 1200,
                       1300, open_, high, low, close)


def _equity_full(token: int, ltp: int, oi: int, stamp: int,
                 last_traded: int = 1_700_000_001,
                 oi_high: int = 999_999_999, oi_low: int = 7) -> bytes:
    """A 184-byte packet with a DISTINCT value in every field.

    Writing the same value into open interest, its day high and its day low
    made the offsets indistinguishable: reading OI from 52 or 56 instead of
    48 passed silently, which is exactly the highest-risk misread in this
    format.
    """
    head = _equity_quote(token, ltp, ltp, ltp, ltp, ltp, 1_000_000)
    block = struct.pack(">IIIII", last_traded, oi, oi_high, oi_low, stamp)
    depth = b"".join(struct.pack(">iihh", 100, ltp, 3, 0) for _ in range(10))
    packet = head + block + depth
    assert len(packet) == 184
    return packet


def _index(token: int, ltp: int, high: int, low: int, open_: int,
           close: int) -> bytes:
    return struct.pack(">iiiiiii", token, ltp, high, low, open_, close, 0)


NOW = datetime(2026, 9, 8, 10, 0)


def test_ltp_packet_converts_paise_to_rupees() -> None:
    ticks = kt.parse_frame(_frame(_ltp(RELIANCE, 129490)), NOW)
    assert len(ticks) == 1
    assert ticks[0].instrument_token == RELIANCE
    assert ticks[0].last_price == pytest.approx(1294.90)
    assert ticks[0].mode == kt.MODE_LTP


def test_equity_quote_packet_maps_every_field() -> None:
    ticks = kt.parse_frame(_frame(_equity_quote(
        RELIANCE, 129490, 130410, 130680, 128800, 130950, 9_799_528)), NOW)
    tick = ticks[0]
    assert tick.last_price == pytest.approx(1294.90)
    assert tick.open == pytest.approx(1304.10)
    assert tick.high == pytest.approx(1306.80)
    assert tick.low == pytest.approx(1288.00)
    # Kite's `close` is the PREVIOUS session's close, not today's.
    assert tick.close == pytest.approx(1309.50)
    assert tick.volume == 9_799_528
    assert tick.mode == kt.MODE_QUOTE
    assert tick.change_pct == pytest.approx((1294.90 / 1309.50 - 1) * 100)


def test_full_packet_carries_open_interest() -> None:
    # Open interest is the field NSE never publishes historically, so it
    # arriving here is the whole reason for using full mode. Every
    # neighbouring field carries a different value, so reading OI from 52 or
    # 56, or the timestamp from 56, now fails instead of passing quietly.
    ticks = kt.parse_frame(
        _frame(_equity_full(RELIANCE_FUT, 129620, 128_549_500, 1757325600)), NOW)
    tick = ticks[0]
    assert tick.last_price == pytest.approx(1296.20)
    assert tick.open_interest == 128_549_500, "offset 48, not 52 or 56"
    assert tick.mode == kt.MODE_FULL
    assert tick.exchange_timestamp == datetime(2025, 9, 8, 10, 0,
                                               tzinfo=timezone.utc)


def test_volume_and_open_interest_are_read_unsigned() -> None:
    # These fields are unsigned on the wire. Read as signed they wrap above
    # 2^31: a 2.5 billion share volume parsed as -1,794,967,296, which
    # inverts relative volume downstream with no error anywhere. NSE penny
    # names routinely trade past that.
    big_volume, big_oi = 2_500_000_000, 3_000_000_000
    quote = kt.parse_frame(_frame(_equity_quote(
        RELIANCE, 100, 100, 100, 100, 100, big_volume)), NOW)[0]
    assert quote.volume == big_volume
    full = kt.parse_frame(_frame(_equity_full(
        RELIANCE_FUT, 100, big_oi, 1757325600)), NOW)[0]
    assert full.open_interest == big_oi


def test_index_packet_has_no_volume_and_is_flagged() -> None:
    ticks = kt.parse_frame(_frame(_index(NIFTY_INDEX, 2363510, 2375895,
                                         2362310, 2374310, 2377915)), NOW)
    tick = ticks[0]
    assert tick.is_index
    assert tick.last_price == pytest.approx(23635.10)
    assert tick.high == pytest.approx(23758.95)
    assert tick.close == pytest.approx(23779.15)
    assert tick.volume == 0


def test_one_frame_can_carry_many_packets_in_order() -> None:
    ticks = kt.parse_frame(_frame(_ltp(RELIANCE, 1),
                                  _ltp(RELIANCE_FUT, 2),
                                  _index(NIFTY_INDEX, 3, 1, 1, 1, 1)), NOW)
    assert [t.instrument_token for t in ticks] == [
        RELIANCE, RELIANCE_FUT, NIFTY_INDEX]


def test_change_pct_is_none_without_a_previous_close() -> None:
    ticks = kt.parse_frame(_frame(_ltp(RELIANCE, 129490)), NOW)
    assert ticks[0].close == 0.0
    assert ticks[0].change_pct is None


# --- tick protocol: malformed input must never raise --------------------

def test_a_heartbeat_yields_no_ticks() -> None:
    assert kt.parse_frame(b"\x00", NOW) == []
    assert kt.parse_frame(b"", NOW) == []
    assert kt.parse_frame(None, NOW) == []


def test_a_truncated_packet_is_dropped_not_guessed() -> None:
    # Half a packet has no valid reading, so it must be discarded rather
    # than parsed from whatever bytes happen to be there.
    whole = _frame(_ltp(RELIANCE, 129490))
    assert kt.parse_frame(whole[:-3], NOW) == []


def test_an_unknown_packet_length_is_ignored() -> None:
    assert kt.parse_frame(_frame(b"\x00" * 12), NOW) == []


def test_a_frame_that_overstates_its_packet_count_still_parses_what_it_has() -> None:
    lying = struct.pack(">H", 5) + struct.pack(">H", 8) + _ltp(RELIANCE, 1)
    assert len(kt.parse_frame(lying, NOW)) == 1


def test_an_implausible_exchange_timestamp_becomes_none() -> None:
    # Reading the field unsigned means there are no negative timestamps, so
    # the guard has to bound plausibility instead: 0xFFFFFFFF is a valid
    # datetime in 2106 and would otherwise be accepted as a real print.
    for seconds in (0, 1, 0xFFFFFFFF, 4_200_000_000):
        ticks = kt.parse_frame(
            _frame(_equity_full(RELIANCE_FUT, 100, 0, seconds)), NOW)
        assert ticks[0].exchange_timestamp is None, seconds
    good = kt.parse_frame(
        _frame(_equity_full(RELIANCE_FUT, 100, 0, 1757325600)), NOW)[0]
    assert good.exchange_timestamp is not None


# --- tick protocol: control messages ------------------------------------

def test_control_messages_match_the_documented_shape() -> None:
    assert kt.subscribe_message([1, 2]) == '{"a": "subscribe", "v": [1, 2]}'
    assert kt.mode_message("full", [1]) == '{"a": "mode", "v": ["full", [1]]}'


def test_an_unknown_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="ltp, quote or full"):
        kt.mode_message("bogus", [1])


def test_the_stream_url_requires_both_credentials() -> None:
    with pytest.raises(ValueError, match="required"):
        kt.stream_url("", "token")
    with pytest.raises(ValueError, match="required"):
        kt.stream_url("key", "")
    assert kt.stream_url("k", "t").startswith("wss://")


def test_split_packets_reports_the_segment_from_the_token() -> None:
    ticks = kt.parse_frame(_frame(_ltp(RELIANCE_FUT, 1)), NOW)
    assert ticks[0].segment == 2


# --- 184-byte packets: what is parsed, and what is honestly ignored -----

def test_market_depth_bytes_are_not_parsed_at_all() -> None:
    # The module docstring says the depth block in bytes 64-184 is ignored
    # and that Tick has no depth field. Rewriting every depth byte must
    # therefore change nothing, and this fails the day that stops being
    # true - which is the day the docstring would start lying.
    packet = bytearray(_equity_full(RELIANCE_FUT, 129620, 128_549_500,
                                    1757325600))
    plain = kt.parse_frame(_frame(bytes(packet)), NOW)[0]
    packet[64:184] = b"\xAB" * 120
    scrambled = kt.parse_frame(_frame(bytes(packet)), NOW)[0]
    assert scrambled == plain
    assert not hasattr(plain, "depth")


# --- stream resilience: a bounded read and a responsive stop ------------

_FAKE_SESSION = SimpleNamespace(api_key="key", access_token="tok")


class _FakeSocket:
    """A scripted stand-in for a websockets connection.

    One script entry is consumed per recv(): bytes or str come back as a
    frame, an exception instance is raised, a callable is invoked (used to
    trip the stop event mid-stream) and then skipped, and None blocks
    forever, which is what a half-open TCP connection looks like from here.
    """

    def __init__(self, script: list) -> None:
        self._script = list(script)
        self.sent: list = []

    async def send(self, message) -> None:
        """Record a control message instead of sending it."""
        self.sent.append(message)

    async def recv(self):
        """The next scripted frame, or a wait that never ends."""
        while self._script:
            item = self._script.pop(0)
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            if callable(item):
                item()
                continue
            return item
        await asyncio.sleep(3600)   # cancelled by the caller's bounded read


class _FakeConnect:
    """websockets.connect, handing out one scripted socket per call."""

    def __init__(self, sockets: list) -> None:
        self._sockets = list(sockets)
        self.urls: list = []
        self.kwargs: list = []
        self._current = None

    def __call__(self, url, **kwargs):
        """Open the next scripted connection."""
        self.urls.append(url)
        self.kwargs.append(kwargs)
        self._current = self._sockets[min(len(self.urls) - 1,
                                          len(self._sockets) - 1)]
        return self

    async def __aenter__(self):
        return self._current

    async def __aexit__(self, *exc_info) -> bool:
        return False


def test_next_frame_is_bounded_and_a_stop_beats_a_silent_socket() -> None:
    # await recv() on a half-open connection never returns and never
    # raises, so the read has to be bounded or the recorder parks for the
    # rest of the session.
    async def go() -> None:
        stop = asyncio.Event()
        waiter = asyncio.ensure_future(stop.wait())
        socket = _FakeSocket([b"\x00\x00", None])
        assert await kt._next_frame(socket.recv, waiter, 1.0) == b"\x00\x00"
        with pytest.raises(kt.StreamStalled):
            await kt._next_frame(socket.recv, waiter, 0.02)
        stop.set()
        assert await kt._next_frame(socket.recv, waiter, 30.0) is None
        waiter.cancel()

    asyncio.run(asyncio.wait_for(go(), timeout=5))


def test_a_silent_socket_is_dropped_and_reconnected(monkeypatch) -> None:
    # The first socket is half-open: it accepts the subscription and then
    # says nothing, forever, without raising. The stream has to notice and
    # redial rather than sit there.
    received: list = []
    alive = _FakeSocket([])
    connect = _FakeConnect([])

    async def go():
        nonlocal alive, connect
        stop = asyncio.Event()
        alive = _FakeSocket([_frame(_ltp(RELIANCE, 129490)), stop.set, None])
        connect = _FakeConnect([_FakeSocket([None]), alive])
        monkeypatch.setattr(websockets, "connect", connect)
        return await asyncio.wait_for(
            kt.stream([RELIANCE], received.extend, session=_FAKE_SESSION,
                      stop_event=stop, reconnect_delay=0.0,
                      read_timeout=0.02), timeout=5)

    stats = asyncio.run(go())
    assert stats.stalls == 1
    assert len(connect.urls) == 2, "the stalled socket must be redialled"
    assert connect.kwargs[0]["ping_interval"] is None
    assert [t.instrument_token for t in received] == [RELIANCE]
    assert alive.sent == [kt.subscribe_message([RELIANCE]),
                          kt.mode_message(kt.MODE_FULL, [RELIANCE])]


def test_a_stop_does_not_wait_for_the_next_tick(monkeypatch) -> None:
    # stop_event used to be read only after a frame arrived, so on a quiet
    # instrument a stop request hung until the next print. A 30s read
    # timeout against a 2s deadline proves the stop ended it, not a timeout.
    connect = _FakeConnect([])

    async def go():
        nonlocal connect
        stop = asyncio.Event()
        connect = _FakeConnect([_FakeSocket([None])])
        monkeypatch.setattr(websockets, "connect", connect)
        streamer = asyncio.ensure_future(
            kt.stream([RELIANCE], lambda ticks: None, session=_FAKE_SESSION,
                      stop_event=stop, reconnect_delay=0.0,
                      read_timeout=30.0))
        await asyncio.sleep(0.05)      # let it connect and block on recv
        stop.set()
        return await asyncio.wait_for(streamer, timeout=2)

    stats = asyncio.run(go())
    assert stats.stalls == 0 and stats.errors == 0
    assert len(connect.urls) == 1


def test_a_failing_callback_is_fatal_rather_than_a_reconnect_loop(
        monkeypatch) -> None:
    # A full disk raises OSError from inside the stream loop, where it used
    # to be counted as a stream error and answered with a reconnect,
    # forever, recording nothing. Reconnecting cannot fix a disk.
    def boom(ticks) -> None:
        raise OSError(28, "No space left on device")

    connect = _FakeConnect([])

    async def go() -> None:
        nonlocal connect
        connect = _FakeConnect([_FakeSocket([_frame(_ltp(RELIANCE, 1))])])
        monkeypatch.setattr(websockets, "connect", connect)
        with pytest.raises(kt.CallbackError, match="No space left"):
            await asyncio.wait_for(
                kt.stream([RELIANCE], boom, session=_FAKE_SESSION,
                          reconnect_delay=0.0, read_timeout=1.0), timeout=5)

    asyncio.run(go())
    assert len(connect.urls) == 1


def test_a_zero_read_timeout_is_refused() -> None:
    # A non-positive timeout would stall on the first read and reconnect in
    # a tight loop, which is worse than the hang it replaced.
    async def go() -> None:
        await kt.stream([RELIANCE], lambda ticks: None,
                        session=_FAKE_SESSION, read_timeout=0.0)

    with pytest.raises(ValueError, match="read_timeout"):
        asyncio.run(go())


# --- the recorder's writer: a persistent write failure must stop it -----

class _FakeHandle:
    """A file stand-in whose writes can be made to fail on demand."""

    def __init__(self, fail: bool = False) -> None:
        self.lines: list = []
        self.flushes = 0
        self.fail = fail

    def write(self, text: str) -> None:
        """Append a line, or raise what a full disk would."""
        if self.fail:
            raise OSError(28, "No space left on device")
        self.lines.append(text)

    def flush(self) -> None:
        """Flush, or raise what a full disk would."""
        if self.fail:
            raise OSError(28, "No space left on device")
        self.flushes += 1


def _one_tick() -> list:
    """A single parsed tick, for driving the writer."""
    return kt.parse_frame(_frame(_ltp(RELIANCE, 129490)), NOW)


def test_the_writer_labels_and_serialises_every_tick() -> None:
    handle = _FakeHandle()
    stop = asyncio.Event()
    writer = tr.TickWriter(handle, {RELIANCE: "RELIANCE"}, stop)
    writer(_one_tick())
    row = json.loads(handle.lines[0])
    assert row["symbol"] == "RELIANCE"
    assert row["last_price"] == pytest.approx(1294.90)
    assert row["received_at"] == NOW.isoformat()
    assert row["exchange_timestamp"] is None
    assert (writer.rows, writer.failures, handle.flushes) == (1, 0, 1)
    assert not stop.is_set() and not writer.gave_up


def test_a_persistent_write_failure_stops_the_recorder() -> None:
    # Reconnecting silently around a dead disk is the failure being fixed:
    # the recorder looked alive at WARNING level and recorded nothing.
    handle = _FakeHandle(fail=True)
    stop = asyncio.Event()
    writer = tr.TickWriter(handle, {}, stop, max_failures=2)
    writer(_one_tick())
    assert not stop.is_set(), "one failure can be momentary"
    assert writer.failures == 1
    writer(_one_tick())
    assert stop.is_set() and writer.gave_up
    assert isinstance(writer.last_error, OSError)
    assert writer.rows == 0


def test_a_recovered_write_clears_the_run_of_failures() -> None:
    handle = _FakeHandle(fail=True)
    stop = asyncio.Event()
    writer = tr.TickWriter(handle, {}, stop, max_failures=2)
    writer(_one_tick())
    handle.fail = False
    writer(_one_tick())
    handle.fail = True
    writer(_one_tick())
    assert not stop.is_set(), "the successful write broke the run"
    assert (writer.failures, writer.consecutive_failures) == (2, 1)
    writer(_one_tick())
    assert stop.is_set() and writer.gave_up


def test_the_exit_code_reports_a_write_failure_and_a_dead_stream() -> None:
    # A run that recorded nothing must not exit 0 and look successful.
    handle = _FakeHandle(fail=True)
    writer = tr.TickWriter(handle, {}, asyncio.Event(), max_failures=1)
    writer(_one_tick())
    assert tr._report(kt.TickerStats(), writer, None) == 1
    clean = tr.TickWriter(_FakeHandle(), {}, asyncio.Event())
    assert tr._report(kt.TickerStats(), clean, None) == 0
    dead = RuntimeError("stream died")
    assert tr._report(kt.TickerStats(), clean, dead) == 1


# --- the recorder end to end, with the socket stubbed out ---------------

def _recorder_args(**overrides):
    """Parsed recorder arguments, starting from the module's own defaults."""
    args = tr._parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _stub_recorder(monkeypatch, tmp_path, fake_stream) -> None:
    """Point the recorder at a temp directory and a stubbed stream."""
    monkeypatch.setattr(tr.kite_client, "load_session", lambda: _FAKE_SESSION)
    monkeypatch.setattr(tr, "resolve_tokens",
                        lambda args: {RELIANCE: "RELIANCE"})
    monkeypatch.setattr(tr, "TICK_DIR", tmp_path)
    monkeypatch.setattr(tr, "STATUS_FILE", tmp_path / "status.json")
    monkeypatch.setattr(tr.kt, "stream", fake_stream)


def test_the_recorder_records_a_tick_and_releases_its_file(
        monkeypatch, tmp_path) -> None:
    async def fake_stream(tokens, on_ticks, *, mode, session, stats,
                          stop_event):
        on_ticks(_one_tick())
        stop_event.set()
        return stats

    _stub_recorder(monkeypatch, tmp_path, fake_stream)
    code = asyncio.run(asyncio.wait_for(tr.run(_recorder_args()), timeout=5))
    assert code == 0
    written = list(tmp_path.glob("ticks_*.jsonl"))
    assert len(written) == 1
    assert json.loads(written[0].read_text(encoding="utf-8"))["symbol"] == (
        "RELIANCE")
    assert "stalls" in json.loads(
        (tmp_path / "status.json").read_text(encoding="utf-8"))
    # Windows refuses to unlink a file that is still open, so this is what
    # proves the handle was closed rather than leaked.
    written[0].unlink()


def test_a_dead_stream_ends_the_run_instead_of_hanging(
        monkeypatch, tmp_path) -> None:
    # Without --minutes the run waited on the stop event alone, so a stream
    # task that died on its first connection left the recorder parked for
    # the session: no ticks, no error, nothing to see.
    async def fake_stream(tokens, on_ticks, *, mode, session, stats,
                          stop_event):
        raise RuntimeError("subscription rejected")

    _stub_recorder(monkeypatch, tmp_path, fake_stream)
    code = asyncio.run(asyncio.wait_for(tr.run(_recorder_args()), timeout=5))
    assert code == 1
