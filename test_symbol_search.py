"""Every table with an instrument column gets the same search box.

WHY ONE HELPER. show_table already exists so a column "cannot be
explained in one table and left bare in another" - a filter present on
some tables and missing on others is worse, because from the screen you
cannot tell which case you are looking at.

THE DANGEROUS PART is watch_table, which maps a row SELECTION back to a
position with frame.iloc[index]. A selection index is positional within
the frame that was rendered, so filtering for display while resolving
against the unfiltered frame would add a different instrument than the
one clicked, straight into Positions, with nothing on screen wrong. The
test for that is the reason this file exists.
"""
from __future__ import annotations

import pandas as pd
import pytest

import app


class _Surface:
    """A stand-in for st / a column, recording what was rendered."""

    def __init__(self, query=""):
        self.query = query
        self.captions: list = []
        self.rendered = None

    def text_input(self, label, **kwargs):
        return self.query

    def caption(self, text, **kwargs):
        self.captions.append(text)

    def dataframe(self, frame, **kwargs):
        self.rendered = frame
        return type("R", (), {"selection": {"rows": []}})()


def _frame(symbols=None, column="symbol"):
    symbols = symbols or [f"SYM{i:02d}" for i in range(12)]
    return pd.DataFrame({column: symbols,
                         "score": range(len(symbols))})


# --- which column ---------------------------------------------------------

@pytest.mark.parametrize("column", ["symbol", "Symbol", "ticker", "Ticker"])
def test_the_instrument_column_is_found_whatever_it_is_called(column) -> None:
    """The scan publishes "symbol"; the horizon tables title-case it."""
    assert app.symbol_column(_frame(column=column)) == column


def test_a_table_with_no_instrument_column_has_none() -> None:
    assert app.symbol_column(pd.DataFrame({"price": [1.0]})) is None


def test_a_non_frame_is_none_rather_than_an_error() -> None:
    assert app.symbol_column(None) is None


# --- when the box appears -------------------------------------------------

def test_a_short_table_gets_no_search_box() -> None:
    """Clutter on a table already entirely on screen."""
    surface = _Surface()
    small = _frame([f"S{i}" for i in range(3)])
    got = app.symbol_search(small, "k", target=surface)
    assert got is small
    assert surface.captions == []


def test_a_long_table_gets_one() -> None:
    surface = _Surface(query="")
    big = _frame()
    assert len(big) >= app.SEARCH_MIN_ROWS
    got = app.symbol_search(big, "k", target=surface)
    assert len(got) == len(big), "an empty query must not filter"


def test_a_table_without_symbols_gets_no_box() -> None:
    surface = _Surface(query="ANY")
    frame = pd.DataFrame({"price": [1.0] * 20})
    got = app.symbol_search(frame, "k", target=surface)
    assert len(got) == 20, "filtered a table with nothing to filter on"


# --- matching -------------------------------------------------------------

def test_a_substring_matches() -> None:
    frame = _frame(["COLPAL", "HDFCBANK", "HDFCLIFE", "RELIANCE",
                    "TCS", "INFY", "WIPRO", "ITC", "SBIN"])
    got = app.symbol_search(frame, "k", target=_Surface(query="HDFC"))
    assert sorted(got["symbol"]) == ["HDFCBANK", "HDFCLIFE"]


def test_matching_ignores_case() -> None:
    frame = _frame(["COLPAL", "HDFCBANK", "RELIANCE", "TCS", "INFY",
                    "WIPRO", "ITC", "SBIN"])
    got = app.symbol_search(frame, "k", target=_Surface(query="colpal"))
    assert list(got["symbol"]) == ["COLPAL"]


@pytest.mark.parametrize("query", ["COLPAL TCS", "COLPAL,TCS",
                                   "  COLPAL ,, TCS  "])
def test_several_terms_are_a_union(query) -> None:
    """Separated by spaces or commas, so a handful can be pulled up."""
    frame = _frame(["COLPAL", "HDFCBANK", "RELIANCE", "TCS", "INFY",
                    "WIPRO", "ITC", "SBIN"])
    got = app.symbol_search(frame, "k", target=_Surface(query=query))
    assert sorted(got["symbol"]) == ["COLPAL", "TCS"]


def test_a_query_is_not_a_regex() -> None:
    """A stray bracket must not raise, and must not match everything."""
    frame = _frame(["COLPAL", "HDFCBANK", "RELIANCE", "TCS", "INFY",
                    "WIPRO", "ITC", "SBIN"])
    got = app.symbol_search(frame, "k", target=_Surface(query="A.*L"))
    assert got.empty, "treated the query as a pattern"


def test_no_match_says_so_rather_than_showing_an_empty_table() -> None:
    """An empty table reads as "no data", which is a different fact."""
    surface = _Surface(query="NOSUCHNAME")
    frame = _frame()
    got = app.symbol_search(frame, "k", target=surface)
    assert got.empty
    assert any("Nothing matching" in c for c in surface.captions)


def test_a_narrowed_table_reports_the_counts() -> None:
    surface = _Surface(query="SYM0")
    app.symbol_search(_frame(), "k", target=surface)
    assert any(" of " in c for c in surface.captions)


# --- the one that could lose money ---------------------------------------

def test_a_filtered_selection_resolves_to_the_row_that_was_clicked(
        monkeypatch) -> None:
    """THE REGRESSION TEST.

    watch_table renders a frame, then maps the selection index back with
    frame.iloc[index]. Both must be the SAME frame: rendering the
    filtered rows while resolving against the unfiltered ones would add a
    different instrument than the one clicked, into Positions, with
    nothing on screen looking wrong.
    """
    frame = _frame(["AAA", "BBB", "CCC", "TARGET", "EEE", "FFF", "GGG",
                    "HHH", "III", "JJJ"])
    seen: dict = {}

    class _Picky(_Surface):
        def dataframe(self, frame, **kwargs):
            self.rendered = frame
            # The user clicks the first visible row.
            return type("R", (), {"selection": {"rows": [0]}})()

        def button(self, *args, **kwargs):
            return False

    def capture(row, symbol=None, entry=None, note=""):
        seen["symbol"] = row[app.symbol_column(frame)]
        return None

    monkeypatch.setattr(app.position_watch, "position_from_row", capture)

    surface = _Picky(query="TARGET")
    app.watch_table(frame, "watch_key", target=surface)

    assert list(surface.rendered[app.symbol_column(frame)]) == ["TARGET"]
    assert seen["symbol"] == "TARGET", (
        f"clicked the only visible row and got {seen.get('symbol')!r}")


def test_an_out_of_range_selection_is_ignored(monkeypatch) -> None:
    """A stale selection index after filtering must not raise."""
    frame = _frame()
    called: dict = {"n": 0}

    class _Stale(_Surface):
        def dataframe(self, frame, **kwargs):
            self.rendered = frame
            return type("R", (), {"selection": {"rows": [99]}})()

    def counted(row, **kwargs):
        called["n"] += 1
        return None

    monkeypatch.setattr(app.position_watch, "position_from_row", counted)
    app.watch_table(frame, "k", target=_Stale(query="SYM00"))
    assert called["n"] == 0


# --- every table is wired ------------------------------------------------

def test_every_show_table_caller_passes_a_key() -> None:
    """A table without a key silently has no search box.

    The whole point is that the filter is not present on some tables and
    absent on others, so this asserts the wiring rather than trusting it.
    """
    import inspect

    lines = inspect.getsource(app).splitlines()
    calls = [line.strip() for line in lines
             if "show_table(" in line
             and not line.lstrip().startswith(("def ", "#", "*"))
             and "show_table, not st.dataframe" not in line]
    assert len(calls) >= 6, f"expected several call sites, found {calls}"

    # Two callers pass no key, both deliberately:
    #   - the premarket panel calls symbol_search itself, because its
    #     search must see the whole published list rather than the
    #     slider's slice;
    #   - the pivot ladder's rows are LEVELS, not instruments, and there
    #     are five of them.
    # Anything else without a key has silently lost its search box.
    bare = [c for c in calls if "key=" not in c]
    assert len(bare) == 2, bare
    assert any("found" in c for c in bare), bare
    assert any(c == "show_table(frame)" for c in bare), bare
