"""Attention chimes for the position watch, generated rather than shipped.

WHY A CHIME AT ALL, when the watch also speaks. Speech goes through the
Web Speech API inside a component iframe, and Chrome has refused
speechSynthesis.speak() in a document that has never been clicked since
M71. A component iframe is rebuilt on the rerun that carries the alert, so
it may well be such a document, and a stop going through is exactly the
case where "the browser suppressed it" is not an acceptable outcome. The
chime plays in the page itself, which by the time an alert can fire has
usually been clicked - though not certainly, since a page left open from
before the open has had no interaction either. Neither channel is
guaranteed, which is why there are two of them plus a toast and a table.

WHY GENERATED. These are sine tones with a fade - a few dozen lines of
stdlib rather than four binary blobs nobody can review or diff. The
patterns stay distinguishable without looking: falling and repeated for a
stop, rising for a target, one short note for nearing it, and an
alternating pair for a reversal.
"""
from __future__ import annotations

import io
import math
import struct
import wave
from functools import lru_cache

import position_watch

SAMPLE_RATE = 22_050
AMPLITUDE = 0.35            # comfortable over laptop speakers, not startling
FADE_SECONDS = 0.012        # long enough to kill the click at each edge

# (frequency in Hz, seconds). A gap is a frequency of 0.
PATTERNS: dict[str, tuple[tuple[float, float], ...]] = {
    # Falling and repeated - the one pattern that should be unpleasant.
    position_watch.STOP_BREACHED: ((880, 0.16), (0, 0.05), (740, 0.16),
                                   (0, 0.05), (554, 0.30)),
    # Rising and resolved.
    position_watch.TARGET_REACHED: ((659, 0.14), (784, 0.14), (988, 0.26)),
    # Deliberately slight: nearing a target is information, not an alarm.
    position_watch.NEAR_TARGET: ((784, 0.12),),
    # Alternating, because a reversal is two things swapping places.
    position_watch.FLIPPED: ((622, 0.14), (466, 0.14), (622, 0.14),
                             (466, 0.20)),
}


def _samples(frequency: float, seconds: float) -> list[int]:
    """One tone as 16-bit signed samples, faded in and out at the edges."""
    total = max(1, int(SAMPLE_RATE * seconds))
    fade = max(1, int(SAMPLE_RATE * FADE_SECONDS))
    out = []
    for index in range(total):
        if frequency <= 0:
            out.append(0)
            continue
        envelope = min(1.0, index / fade, (total - index) / fade)
        value = math.sin(2 * math.pi * frequency * index / SAMPLE_RATE)
        out.append(int(32_767 * AMPLITUDE * envelope * value))
    return out


def wav_bytes(pattern) -> bytes:
    """A pattern of (frequency, seconds) tones as a mono 16-bit WAV."""
    frames = []
    for frequency, seconds in pattern:
        frames.extend(_samples(frequency, seconds))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(struct.pack(f"<{len(frames)}h", *frames))
    return buffer.getvalue()


# How many distinguishable payloads exist per state, via trailing silence.
# One alert is one payload, so a session would need this many alerts of the
# SAME state before an id repeats. 400 x 1 frame is 18 milliseconds of
# silence at the longest, which is inaudible.
NONCE_SPACE = 400


@lru_cache(maxsize=None)
def chime_for(state: str, nonce: int = 0) -> bytes:
    """The chime for one watch state, or b"" for a state that gets none.

    Returning empty rather than a default tone is deliberate: a position
    that is merely still supported must make no noise at all, or the whole
    thing gets muted within an hour.

    `nonce` appends inaudible trailing silence, and it is not decoration.
    st.audio derives the element id from a hash of the audio CONTENT, and
    the frontend refuses to autoplay an id it has already seen - so with
    identical bytes the first STOP BREACHED of a session sounded and every
    one after it was silent. Pass a counter that rises with each alert.
    """
    pattern = PATTERNS.get(state)
    if not pattern:
        return b""
    padding = ((0, (1 + nonce % NONCE_SPACE) / SAMPLE_RATE),) if nonce else ()
    return wav_bytes(tuple(pattern) + padding)


def _example_sentence(state: str) -> str:
    """What the watch would say out loud alongside this chime.

    Built from position_watch rather than retyped here, so the demo cannot
    drift from what the app actually says.
    """
    example = position_watch.Position(
        symbol="HDFCLIFE", side=position_watch.SHORT, entry=511.20,
        stop=516.00, target=501.60)
    return position_watch.Status(position=example, state=state,
                                 headline="").spoken


def main(argv=None) -> int:
    """Play every chime once, so they can be heard without risking money.

    Exists because the alternative way to hear these is to hold a position
    until its stop goes through, and nobody should have to audition an
    alarm that way. Playback is winsound, which is stdlib on Windows and
    absent everywhere else; with --save (or off Windows) the WAVs are
    written out instead so any player can open them.
    """
    import argparse
    import time

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", metavar="DIR", nargs="?", const=".",
                        help="write the WAVs here instead of playing them")
    parser.add_argument("--only", metavar="STATE",
                        help="one state only, e.g. \"STOP BREACHED\"")
    args = parser.parse_args(argv)

    from pathlib import Path

    wanted = [s for s in position_watch.ANNOUNCE
              if not args.only or s.upper() == args.only.upper()]
    if not wanted:
        print(f"No such state. Choose from: "
              f"{', '.join(position_watch.ANNOUNCE)}")
        return 1
    try:
        import winsound
    except ImportError:
        winsound = None
    if args.save or winsound is None:
        folder = Path(args.save or ".")
        folder.mkdir(parents=True, exist_ok=True)
        for state in wanted:
            path = folder / f"{state.lower().replace(chr(32), chr(95))}.wav"
            path.write_bytes(chime_for(state))
            print(f"  {state:<15} -> {path}")
        return 0
    for state in wanted:
        # Printed before playing, so the sound and the sentence it comes
        # with arrive together rather than the label trailing the tone.
        print(f"  {state:<17} {_example_sentence(state)}", flush=True)
        winsound.PlaySound(chime_for(state), winsound.SND_MEMORY)
        time.sleep(0.6)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
