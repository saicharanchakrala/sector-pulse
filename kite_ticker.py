"""Kite WebSocket tick stream: genuinely live prices, parsed from binary.

This is the only source in this project that is actually live. yfinance
serves NSE with roughly a fifteen-minute delay, so polling it faster never
made data fresher - a point worth keeping in mind, because it was the
premise of an earlier plan to run a background poller.

Kite streams binary frames over one socket. The wire format is not
self-describing: a frame carries a packet count, then length-prefixed
packets whose MEANING is inferred from their byte length alone. So the
parsers below are the specification as far as this project is concerned,
and they are written as pure functions over bytes precisely so they can be
tested without a market open or a credential.

    2 bytes            number of packets in this frame
    per packet:
      2 bytes          packet length
      N bytes          payload, interpreted by N:
         8             last price only
        28, 32         an index
        44             equity quote
       184             equity quote, open interest, and market depth

Two things the 184-byte packet carries are deliberately NOT parsed: the
last-traded timestamp at offset 44, and the market depth in bytes 64 to
184, which is ten 12-byte entries - five bids then five offers. Tick has
no field for either and nothing in this project reads them, so parsing
them would widen every recorded row for no reader. A mode of "full"
therefore describes what was asked of Kite, not what a Tick holds: a Tick
has no depth field at all.

Prices arrive as integers in the instrument's minor unit - paise for NSE
and NFO - and the divisor depends on the segment encoded in the low byte of
the instrument token. Getting that wrong yields prices off by a factor of a
hundred, which is why it is derived rather than assumed.

A one-byte frame is a heartbeat and carries no data.
"""
from __future__ import annotations

import asyncio
import json
import logging
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

WS_URL = "wss://ws.kite.trade"

MODE_LTP = "ltp"
MODE_QUOTE = "quote"
MODE_FULL = "full"

# Segment is the low byte of the instrument token. Only the divisor differs;
# everything else about the packet layout is the same.
_SEGMENT_NSE_CM = 1
_SEGMENT_NFO_FO = 2
_SEGMENT_CDS = 3
_SEGMENT_BSE_CM = 4
_SEGMENT_BFO_FO = 5
_SEGMENT_BCD = 6
_SEGMENT_INDICES = 9

_DIVISORS = {_SEGMENT_CDS: 10_000_000.0, _SEGMENT_BCD: 10_000.0}
_DEFAULT_DIVISOR = 100.0

# One instrument's subscription cap and connection cap, per Kite's docs.
MAX_INSTRUMENTS_PER_CONNECTION = 3000


@dataclass(frozen=True)
class Tick:
    """One instrument's state at one instant, as the stream reported it."""

    instrument_token: int
    last_price: float
    mode: str
    received_at: datetime
    last_quantity: int = 0
    average_price: float = 0.0
    volume: int = 0
    buy_quantity: int = 0
    sell_quantity: int = 0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    open_interest: int = 0
    exchange_timestamp: "datetime | None" = None
    is_index: bool = False

    @property
    def segment(self) -> int:
        """The segment encoded in the instrument token's low byte."""
        return self.instrument_token & 0xFF

    @property
    def change_pct(self) -> "float | None":
        """Percent change against the previous close the stream supplied."""
        if self.close <= 0:
            return None
        return (self.last_price / self.close - 1.0) * 100.0


def divisor_for(instrument_token: int) -> float:
    """Price divisor for an instrument, from its segment."""
    return _DIVISORS.get(instrument_token & 0xFF, _DEFAULT_DIVISOR)


def split_packets(frame: bytes) -> list[bytes]:
    """Split one binary frame into its length-prefixed packets.

    A frame shorter than two bytes is a heartbeat. A truncated tail is
    dropped rather than guessed at: a partial packet has no valid reading.
    """
    if frame is None or len(frame) < 2:
        return []
    count = struct.unpack(">H", frame[0:2])[0]
    packets: list[bytes] = []
    cursor = 2
    for _ in range(count):
        if cursor + 2 > len(frame):
            logger.warning("Frame ended mid-header after %d packet(s)",
                           len(packets))
            break
        length = struct.unpack(">H", frame[cursor:cursor + 2])[0]
        cursor += 2
        if cursor + length > len(frame):
            logger.warning("Frame ended mid-packet (wanted %d bytes)", length)
            break
        packets.append(frame[cursor:cursor + length])
        cursor += length
    return packets


def _int32(payload: bytes, offset: int) -> int:
    """One big-endian SIGNED 32-bit integer, for prices."""
    return struct.unpack(">i", payload[offset:offset + 4])[0]


def _uint32(payload: bytes, offset: int) -> int:
    """One big-endian UNSIGNED 32-bit integer.

    Quantities, volume, open interest and timestamps are unsigned on the
    wire. Read as signed they wrap above 2^31: a 2.5 billion share volume
    parsed as -1,794,967,296, which inverts relative volume downstream with
    no error anywhere. NSE penny names routinely trade past that.
    """
    return struct.unpack(">I", payload[offset:offset + 4])[0]


# A market timestamp outside this window is not a timestamp. The upper
# bound matters because these fields are read UNSIGNED: what would have
# been -1 signed arrives as 4294967295, which is a perfectly valid datetime
# in 2106 and would otherwise be accepted as real.
_STAMP_MIN = 946_684_800          # 2000-01-01
_STAMP_MAX = 4_102_444_800        # 2100-01-01


def _stamp(seconds: int) -> "datetime | None":
    """An exchange timestamp, or None when the field is unset or implausible."""
    if seconds <= 0 or not (_STAMP_MIN <= seconds <= _STAMP_MAX):
        return None
    try:
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def parse_packet(payload: bytes, received_at: datetime) -> "Tick | None":
    """Interpret one packet by its length; None when the length is unknown."""
    size = len(payload)
    if size < 8:
        return None
    token = _int32(payload, 0)
    scale = divisor_for(token)

    if size == 8:
        return Tick(instrument_token=token,
                    last_price=_int32(payload, 4) / scale,
                    mode=MODE_LTP, received_at=received_at)

    if size in (28, 32):
        # Index packets carry no volume or depth, and their field order
        # differs from equities: the change field sits where volume would.
        tick = Tick(
            instrument_token=token, last_price=_int32(payload, 4) / scale,
            mode=MODE_QUOTE if size == 28 else MODE_FULL,
            received_at=received_at, high=_int32(payload, 8) / scale,
            low=_int32(payload, 12) / scale, open=_int32(payload, 16) / scale,
            close=_int32(payload, 20) / scale, is_index=True,
            exchange_timestamp=_stamp(_uint32(payload, 28)) if size == 32 else None)
        return tick

    if size in (44, 184):
        tick_kwargs = dict(
            instrument_token=token, last_price=_int32(payload, 4) / scale,
            last_quantity=_uint32(payload, 8),
            average_price=_int32(payload, 12) / scale,
            volume=_uint32(payload, 16), buy_quantity=_uint32(payload, 20),
            sell_quantity=_uint32(payload, 24),
            open=_int32(payload, 28) / scale, high=_int32(payload, 32) / scale,
            low=_int32(payload, 36) / scale, close=_int32(payload, 40) / scale,
            received_at=received_at,
        )
        if size == 184:
            tick_kwargs.update(
                mode=MODE_FULL,
                open_interest=_uint32(payload, 48),
                exchange_timestamp=_stamp(_uint32(payload, 60)))
        else:
            tick_kwargs.update(mode=MODE_QUOTE)
        return Tick(**tick_kwargs)

    logger.warning("Unrecognised packet length %d; ignoring", size)
    return None


def parse_frame(frame: bytes, received_at: "datetime | None" = None
                ) -> list[Tick]:
    """Every tick in one binary frame."""
    stamp = received_at or datetime.now().astimezone()
    return [tick for tick in
            (parse_packet(p, stamp) for p in split_packets(frame))
            if tick is not None]


def subscribe_message(tokens: list[int]) -> str:
    """The JSON Kite expects to add instruments to a subscription."""
    return json.dumps({"a": "subscribe", "v": [int(t) for t in tokens]})


def mode_message(mode: str, tokens: list[int]) -> str:
    """The JSON Kite expects to set the detail level for instruments."""
    if mode not in (MODE_LTP, MODE_QUOTE, MODE_FULL):
        raise ValueError(f"mode must be ltp, quote or full, got {mode!r}")
    return json.dumps({"a": "mode", "v": [mode, [int(t) for t in tokens]]})


def stream_url(api_key: str, access_token: str) -> str:
    """The authenticated socket URL.

    The access token is a credential. It has to travel in the query string
    because that is the only place Kite accepts it, so this URL must never
    be logged or printed.

    The query is percent-encoded rather than interpolated. A token
    containing a '#' made websockets raise InvalidURI, whose message embeds
    the entire URI - so the credential landed in a WARNING line despite the
    handler deliberately logging only the exception text.
    """
    if not api_key or not access_token:
        raise ValueError("Both api_key and access_token are required")
    query = urlencode({"api_key": api_key, "access_token": access_token})
    return f"{WS_URL}?{query}"


@dataclass
class TickerStats:
    """What the stream has done so far, for reporting without leaking data."""

    frames: int = 0
    ticks: int = 0
    heartbeats: int = 0
    errors: int = 0
    stalls: int = 0
    connected_at: "datetime | None" = None
    last_tick_at: "datetime | None" = None
    tokens: set = field(default_factory=set)

    def summary(self) -> dict:
        """Plain counters for a status line."""
        return {"frames": self.frames, "ticks": self.ticks,
                "heartbeats": self.heartbeats, "errors": self.errors,
                "stalls": self.stalls, "instruments": len(self.tokens),
                "connected_at": (self.connected_at.isoformat()
                                 if self.connected_at else None),
                "last_tick_at": (self.last_tick_at.isoformat()
                                 if self.last_tick_at else None)}


# Kite sends a one-byte heartbeat about once a second on a live connection,
# so silence for this long means the socket is gone even when TCP has not
# noticed. Long enough to survive a slow moment, short enough that a dead
# feed is caught inside one status refresh.
#
# The trade this makes: if Kite ever goes quiet WITHOUT dropping the
# connection, this reconnects every timeout instead of hanging. That is the
# intended direction. A reconnect is counted, logged and visible in the
# status file; a hang looks exactly like a market with nothing to say.
READ_TIMEOUT = 30.0


class StreamStalled(Exception):
    """Nothing arrived inside the read timeout; the socket is dead."""


class CallbackError(Exception):
    """`on_ticks` raised.

    Wrapped so the reconnect loop cannot mistake a consumer failure for a
    network failure. Reconnecting does not fix a full disk: it hides it
    behind an endless retry that records nothing.
    """


async def _next_frame(recv, stop_waiter: "asyncio.Future",
                      read_timeout: float) -> "bytes | str | None":
    """The next frame from `recv()`, or None when a stop has been requested.

    The read is bounded on purpose. A half-open TCP connection delivers
    nothing and raises nothing, so an unbounded `await recv()` parks the
    caller for the rest of the session; StreamStalled turns that silence
    into something the caller can act on. Racing the stop waiter alongside
    the read is what makes a stop take effect between frames instead of
    after the next one, which on a quiet instrument could be minutes away.
    """
    receiver = asyncio.ensure_future(recv())
    try:
        done, _pending = await asyncio.wait(
            {receiver, stop_waiter}, timeout=read_timeout,
            return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        # A cancelled caller must not leave a read running on a socket it
        # is about to close.
        receiver.cancel()
        raise
    if receiver in done:
        return receiver.result()
    # Both remaining paths abandon this socket, so a frame lost to this
    # cancellation cannot matter.
    receiver.cancel()
    if stop_waiter in done:
        return None
    raise StreamStalled(f"no frame in {read_timeout:g}s")


async def _dispatch_frame(frame, on_ticks, counters: TickerStats) -> None:
    """Count one frame and hand its ticks to `on_ticks`."""
    if isinstance(frame, str):
        # Kite sends JSON for errors and order updates.
        logger.info("Stream message: %s", frame[:200])
        return
    counters.frames += 1
    if len(frame) < 2:
        counters.heartbeats += 1
        return
    ticks = parse_frame(frame)
    if not ticks:
        return
    counters.ticks += len(ticks)
    counters.last_tick_at = datetime.now().astimezone()
    try:
        result = on_ticks(ticks)
        if asyncio.iscoroutine(result):
            await result
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        raise CallbackError(
            f"on_ticks failed: {type(exc).__name__}: {exc}") from exc


async def _run_connection(socket, tokens: list[int], mode: str, on_ticks,
                          counters: TickerStats, stop: asyncio.Event,
                          stop_waiter: "asyncio.Future",
                          read_timeout: float) -> None:
    """Subscribe on a fresh socket, then pump frames until stop or trouble."""
    counters.connected_at = datetime.now().astimezone()
    counters.tokens.update(int(t) for t in tokens)
    await socket.send(subscribe_message(tokens))
    await socket.send(mode_message(mode, tokens))
    logger.info("Subscribed to %d instrument(s) in %s mode", len(tokens), mode)
    while not stop.is_set():
        frame = await _next_frame(socket.recv, stop_waiter, read_timeout)
        if frame is None:
            return
        await _dispatch_frame(frame, on_ticks, counters)


def _scrub(text: str, *secrets: str) -> str:
    """`text` with any of `secrets` replaced by stars.

    Defence in depth. Some exception messages embed the whole URI, which
    carries the access token, so the values are removed rather than trusted
    to be absent.
    """
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


async def stream(tokens: list[int], on_ticks, mode: str = MODE_FULL,
                 session=None, stats: "TickerStats | None" = None,
                 stop_event: "asyncio.Event | None" = None,
                 reconnect_delay: float = 5.0,
                 read_timeout: float = READ_TIMEOUT) -> TickerStats:
    """Connect, subscribe, and hand every batch of ticks to `on_ticks`.

    Reconnects on a socket that drops OR one that goes silent, rather than
    exiting, because a recorder that dies on the first network blip records
    nothing useful. It deliberately does NOT reconnect when `on_ticks`
    raises: that is the consumer failing, and retrying the socket would
    only hide it. `on_ticks` receives a list of Tick and may be sync or
    async.

    The library keepalive stays off and liveness is judged from the receive
    side instead. Kite already sends a heartbeat about once a second, so a
    bounded read detects a half-open connection from behaviour this module
    observes and counts, whereas a ping only detects it if the server
    answers pongs - which Kite does not document, and a server that ignored
    them would have ping_timeout tearing down healthy sockets every twenty
    seconds. The bounded read is also the only one of the two that can be
    tested without a live market.
    """
    import websockets  # local import: only the live path needs it

    import kite_client
    live = session or kite_client.load_session()
    if live is None:
        raise kite_client.KiteError("No Kite session. Run kite_login first.")
    if not tokens:
        raise ValueError("No instrument tokens to subscribe to")
    if len(tokens) > MAX_INSTRUMENTS_PER_CONNECTION:
        raise ValueError(
            f"Kite caps one connection at {MAX_INSTRUMENTS_PER_CONNECTION} "
            f"instruments, got {len(tokens)}. Split across connections.")
    if read_timeout <= 0:
        raise ValueError(f"read_timeout must be positive, got {read_timeout}")
    counters = stats or TickerStats()
    url = stream_url(live.api_key, live.access_token)

    # An internal event when the caller supplied none, so that every wait
    # below has one thing to race against and no branch has to ask whether
    # a stop is even possible.
    stop = stop_event if stop_event is not None else asyncio.Event()
    stop_waiter = asyncio.ensure_future(stop.wait())
    try:
        while not stop.is_set():
            try:
                async with websockets.connect(url,
                                              ping_interval=None) as socket:
                    await _run_connection(socket, tokens, mode, on_ticks,
                                          counters, stop, stop_waiter,
                                          read_timeout)
            except (asyncio.CancelledError, CallbackError):
                # CancelledError is not an Exception, but naming it says the
                # omission is deliberate. CallbackError is fatal by design:
                # see the class docstring.
                raise
            except StreamStalled as exc:
                counters.errors += 1
                counters.stalls += 1
                logger.warning("Stream silent (%s); reconnecting in %gs",
                               exc, reconnect_delay)
            except Exception as exc:
                counters.errors += 1
                reason = _scrub(str(exc)[:400], live.access_token,
                                live.api_key)
                logger.warning("Stream dropped (%s: %s); reconnecting in %gs",
                               type(exc).__name__, reason[:160],
                               reconnect_delay)
            if stop.is_set():
                break
            # A sleep that a stop request can cut short.
            await asyncio.wait({stop_waiter}, timeout=reconnect_delay)
    finally:
        stop_waiter.cancel()
    return counters
