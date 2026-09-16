"""Publish the turnover ranking the containerised feed ranks its universe on.

WHY A SEPARATE FILE AND NOT THE DAILY STORE. live_feed.by_turnover ranks
symbols on 20-session turnover read from forecast_cache/bars_day.parquet,
which is 559 MB. A container has no such file:

  * baking it into the image makes every deploy carry 559 MB of history
    that is stale the moment the next nightly fold runs, and
  * downloading it at task start costs a minute and a half of the window
    between the task starting and the 09:15 open.

The ranking itself is about 2,500 rows of symbol and turnover - roughly
60 KB. That is the only part of those 559 MB the feed ever reads, so it is
the only part that crosses the network. Same principle as the live bars:
move what is actually needed, leave the history where the reader is.

THE ARITHMETIC IS NOT REPEATED HERE. It comes from live_feed.turnover_values,
so a change to how the feed ranks cannot leave the publisher computing the
old thing while both continue to run.

WHEN TO RUN IT. Nightly, after fetch_tail.py has folded the day into the
store - the ranking is only as current as the store it reads. A ranking a
few days stale is not serious, since 20-session turnover moves slowly, but
one computed before a fold describes yesterday's universe.
"""
from __future__ import annotations

import io
import sys

import object_store

# From object_store, so the writer and the feed that reads it
# cannot end up on two different keys.
OBJECT = object_store.TURNOVER_OBJECT


def build() -> "object | None":
    """The ranking as a frame, or None when the daily store cannot be read."""
    import pandas as pd

    import bar_store
    import live_feed

    try:
        frames = bar_store.load("day")
    except Exception as exc:
        print(f"Could not read the daily store: {exc}")
        return None
    ranked = live_feed.turnover_values(frames)
    if not ranked:
        print("The daily store produced no ranking - is it empty?")
        return None
    return pd.DataFrame({"symbol": [s for _, s in ranked],
                         "turnover": [v for v, _ in ranked]})


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not object_store.enabled():
        print(f"{object_store.BUCKET_ENV} is not set, so there is nowhere to "
              f"publish to.\nSet it to the bucket the feed reads, for "
              f"example:\n  set {object_store.BUCKET_ENV}=original-image-test")
        return 2

    frame = build()
    if frame is None:
        return 1

    buffer = io.BytesIO()
    frame.to_parquet(buffer)
    payload = buffer.getvalue()
    try:
        object_store.put(OBJECT, payload)
    except object_store.StorageError as exc:
        print(f"Publish failed: {exc}")
        return 1

    top = ", ".join(frame["symbol"].head(5).tolist())
    print(f"published {len(frame):,} symbols ({len(payload) / 1024:.0f} KB) "
          f"to {object_store.describe()}/{OBJECT}")
    print(f"  most traded: {top}")
    if "--show" in argv:
        print(frame.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
