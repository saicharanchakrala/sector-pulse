"""Tests for the generated attention chimes.

Small file, but the two failures worth catching are silent ones: a WAV
that no browser will decode, and a chime handed to a state that should
make no noise. Both would be discovered at the moment a stop went through,
which is the worst possible moment to discover them.
"""
from __future__ import annotations

import io
import wave

import position_watch as pw
import sound


def read(payload: bytes) -> tuple:
    """(channels, sample width, frame rate, frame count) of a WAV blob."""
    with wave.open(io.BytesIO(payload), "rb") as handle:
        return (handle.getnchannels(), handle.getsampwidth(),
                handle.getframerate(), handle.getnframes())


def test_every_announced_state_has_a_playable_chime() -> None:
    # position_watch.ANNOUNCE is the contract: anything worth interrupting
    # someone for must have a sound, or the alert is silent.
    for state in pw.ANNOUNCE:
        payload = sound.chime_for(state)
        assert payload, f"{state} has no chime"
        channels, width, rate, frames = read(payload)
        assert (channels, width, rate) == (1, 2, sound.SAMPLE_RATE)
        assert frames > sound.SAMPLE_RATE * 0.1, f"{state} is too short to hear"


def test_a_quiet_state_makes_no_noise() -> None:
    for state in (pw.SUPPORTED, pw.UNSUPPORTED, pw.UNKNOWN, "", "NONSENSE"):
        assert sound.chime_for(state) == b""


def test_the_four_alerts_are_audibly_different() -> None:
    # Identical bytes would mean a stop and a target sound the same, which
    # defeats the point of not having to look.
    payloads = {state: sound.chime_for(state) for state in pw.ANNOUNCE}
    assert len(set(payloads.values())) == len(payloads)


def test_a_stop_is_the_longest_and_most_insistent() -> None:
    # Ordering, not absolute lengths: the alarm must not be the quietest
    # thing in the set.
    frames = {state: read(sound.chime_for(state))[3]
              for state in pw.ANNOUNCE}
    assert frames[pw.STOP_BREACHED] == max(frames.values())
    assert frames[pw.NEAR_TARGET] == min(frames.values())


def test_samples_stay_inside_the_16_bit_range() -> None:
    # Clipping would turn a tone into a buzz on some players.
    values = sound._samples(880, 0.05)
    assert values and max(values) <= 32_767 and min(values) >= -32_768
    assert abs(values[0]) < 2_000, "no fade in - this clicks"
    assert abs(values[-1]) < 2_000, "no fade out - this clicks"


def test_a_gap_is_silence_rather_than_a_tone() -> None:
    assert set(sound._samples(0, 0.05)) == {0}
