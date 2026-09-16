"""Test-wide isolation.

WHY THIS EXISTS. The suite describes LOCAL-DISK behaviour: 688 tests that
read and write live_bars/ and bar_cache/ and assert on what lands there.
Once object_store existed, three environment variables could silently
redirect all of it at an S3 bucket - and the machine most likely to have
them set is this one, because it is the machine that talks to the feed.

Verified before this file was added: `SECTOR_PULSE_S3_BUCKET=fake pytest
test_live_bars.py` produced three failures that had nothing to do with the
code under test. A developer seeing those would reasonably go looking in
live_bars.py, where there is nothing to find.

So the variables are removed for every test, always. A test that wants the
object store sets it explicitly with monkeypatch, which is both clearer and
the only way it can now happen.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _local_disk_by_default(monkeypatch):
    """Ensure no test is silently pointed at a real object store."""
    import object_store

    for name in (object_store.BUCKET_ENV, object_store.PREFIX_ENV,
                 object_store.REGION_ENV):
        monkeypatch.delenv(name, raising=False)
    # The client is cached per process, so a test that built one against a
    # fake bucket would otherwise hand it to the next test.
    object_store.reset()
    yield
    object_store.reset()
