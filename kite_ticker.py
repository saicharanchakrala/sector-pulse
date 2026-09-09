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
       184             equity quote plus open interest and market depth

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
    connected_at: "datetime | None" = None
    last_tick_at: "datetime | None" = None
    tokens: set = field(default_factory=set)

    def summary(self) -> dict:
        """Plain counters for a status line."""
        return {"frames": self.frames, "ticks": self.ticks,
                "heartbeats": self.heartbeats, "errors": self.errors,
                "instruments": len(self.tokens),
                "connected_at": (self.connected_at.isoformat()
                                 if self.connected_at else None),
                "last_tick_at": (self.last_tick_at.isoformat()
                                 if self.last_tick_at else None)}


async def stream(tokens: list[int], on_ticks, mode: str = MODE_FULL,
                 session=None, stats: "TickerStats | None" = None,
                 stop_event: "asyncio.Event | None" = None,
                 reconnect_delay: float = 5.0) -> TickerStats:
    """Connect, subscribe, and hand every batch of ticks to `on_ticks`.

    Reconnects on a dropped socket rather than exiting, because a recorder
    that dies on the first network blip records nothing useful. `on_ticks`
    receives a list of Tick and may be sync or async.
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
    counters = stats or TickerStats()
    url = stream_url(live.api_key, live.access_token)

    while stop_event is None or not stop_event.is_set():
        try:
            async with websockets.connect(url, ping_interval=None) as socket:
                counters.connected_at = datetime.now().astimezone()
                counters.tokens.update(int(t) for t in tokens)
                await socket.send(subscribe_message(tokens))
                await socket.send(mode_message(mode, tokens))
                logger.info("Subscribed to %d instrument(s) in %s mode",
                            len(tokens), mode)
                while stop_event is None or not stop_event.is_set():
                    frame = await socket.recv()
                    if isinstance(frame, str):
                        # Kite sends JSON for errors and order updates.
                        logger.info("Stream message: %s", frame[:200])
                        continue
                    counters.frames += 1
                    if len(frame) < 2:
                        counters.heartbeats += 1
                        continue
                    ticks = parse_frame(frame)
                    if not ticks:
                        continue
                    counters.ticks += len(ticks)
                    counters.last_tick_at = datetime.now().astimezone()
                    result = on_ticks(ticks)
                    if asyncio.iscoroutine(result):
                        await result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            counters.errors += 1
            # Defence in depth. Some exception messages embed the whole URI,
            # which carries the token, so the values are scrubbed rather
            # than trusted to be absent.
            reason = str(exc)[:400]
            for secret in (live.access_token, live.api_key):
                if secret:
                    reason = reason.replace(secret, "***")
            logger.warning("Stream dropped (%s: %s); reconnecting in %.0fs",
                           type(exc).__name__, reason[:160], reconnect_delay)
            if stop_event is not None and stop_event.is_set():
                break
            await asyncio.sleep(reconnect_delay)
    return counters
