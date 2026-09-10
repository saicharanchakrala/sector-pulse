"""Plain-English meanings for every technical term the UI shows.

One place, so a term cannot be explained two different ways in two tabs,
and so the wording can be improved without hunting through app.py.

The house rule for writing these: say what the number MEANS to someone
deciding what to do, not how it is calculated. "Volume vs normal - 2.0
means twice the usual number of shares changing hands by this time of day"
is useful. "Ratio of cumulative volume to the time-matched median of prior
sessions" is the same fact and helps nobody.

Where a term carries a warning, the warning is part of the definition
rather than a separate note somewhere else. Someone hovering over "Our
ranking" should learn there and then that it was measured and found no
better than random.
"""
from __future__ import annotations

# --- what the columns mean ------------------------------------------------
COLUMNS = {
    # --- the Positions tab. Deliberately its own names: every number in
    # that table is one the USER recorded, so the scan's wording - "sell
    # here if it goes against you", "set at twice the distance to the
    # stop" - is both advice and false there.
    "Held": "Whether you are long or short this position, as you recorded "
            "it.",
    "State": "What is true about the position right now: whether price has "
             "passed your stop or reached your exit, whether it is nearing "
             "the exit, and whether the scan still points your way. A "
             "statement about where price is, never a suggestion.",
    "Your fill": "The price you told this app you actually got in at. "
                 "Correct it with Edit levels if it is wrong - every "
                 "percentage in the row is measured from it.",
    "Price now": "The last live price this check could get. Blank means no "
                 "price came back, which is not the same as nothing having "
                 "changed.",
    "Move %": "How far price has moved IN YOUR FAVOUR since your fill: "
              "positive means up for a long and down for a short.",
    "Your stop": "The stop you recorded. This app only tells you when "
                 "price passes it; it places no orders and gives no "
                 "instruction about what to do.",
    "Your exit": "The exit you recorded, and the level the NEARING TARGET "
                 "warning is measured against.",
    "Shares": "The quantity you recorded, used only to turn a percentage "
              "into rupees. Optional.",
    "Why you took it": "Your own note, or which table the position came "
                       "from.",
    "Symbol": "The stock's ticker on the NSE.",
    "Stock": "The stock's ticker on the NSE.",
    "Side": "Whether the setup is to buy (LONG) or to sell short (SHORT). "
            "This tool only ever proposes buying.",
    "View": "Whether the stock is currently stronger (LONG) or weaker "
            "(SHORT) than the market. Not a recommendation.",
    "Entry": "The price the calculations assume you get in at. It is the "
             "last traded price, not a guaranteed fill.",
    "Entry price": "The price the calculations assume you get in at. It is "
                   "the last traded price, not a guaranteed fill - a real "
                   "order may fill a little away from it.",
    "Stop loss at": "Sell here if it goes against you. Placed using how "
                    "much this instrument typically moves, so ordinary "
                    "wobble should not reach it. Not a prediction that it "
                    "will hold - it is where you have decided to stop "
                    "losing money.",
    "Exit price": "Where the position would be closed in profit. Set at "
                  "twice the distance to the stop, so one winner pays for "
                  "two losers. It is arithmetic from recent volatility, "
                  "not a price forecast.",
    "Stop": "The price at which the trade would be abandoned. Set from how "
            "much this stock typically moves, so it is far enough away not "
            "to be hit by ordinary wobble.",
    "Target": "The price the trade would aim for. Set at twice the "
              "distance to the stop, so one win covers two losses.",
    "Price": "Most recent closing price.",
    "Qty": "How many shares fit your capital and risk settings. Not a "
           "suggestion to buy that many.",
    "Risk": "Rupees you would lose if the stop is hit exactly. Comes from "
            "your risk-per-trade setting.",
    "Cost": "Total charges to get in and out once: brokerage, taxes and "
            "exchange fees. Unavoidable.",
    "Cost %": "Charges to get in and out once, as a percentage of the "
              "position. You start every trade this far behind.",
    "Round trip": "Charges to get in and out once, as a percentage. You "
                  "start this far behind on every trade.",
    "Win % needed": "How often this trade must work just to break even "
                    "after charges. Above about 55% is not realistic.",
    "Can pay for itself?": "'no' means the charges are so large relative "
                           "to the likely move that being right about "
                           "direction still would not make money.",
    "Pays for itself?": "'yes' means the likely move is at least three "
                        "times the charges. 'no' means the fees eat it.",
    "Plausible move %": "How far this stock might reasonably move over the "
                        "holding period, based on how much it has been "
                        "moving lately. Not a prediction of direction.",
    "Plausible move": "How far this stock might reasonably move over the "
                      "holding period, based on recent behaviour. Says "
                      "nothing about which way.",
    "Covers cost": "How many times the likely move covers the charges. "
                   "39x is comfortable; under 3x is not worth doing.",
    "Move vs fees": "How many times the likely move covers the charges. "
                    "Under 3x, fees dominate and the trade is not worth "
                    "doing however right you are.",
    "Stop %": "How far the stop sits from entry, in percent.",
    "Target %": "How far the target sits from entry, in percent.",
    "RVOL": "Volume vs normal. 1.0 means today is trading at its usual "
            "pace for this time of day; 2.0 means twice as busy. Heavy "
            "volume means more people agree something is happening.",
    "Volume vs normal": "1.0 means today is trading at its usual pace for "
                        "this time of day; 2.0 means twice as busy.",
    "RS vs Nifty": "How much this stock has beaten the Nifty index today, "
                   "in percentage points. Positive means outperforming.",
    "Strength vs Nifty": "How far ahead of the Nifty index this stock is, "
                         "in percentage points. Positive is ahead.",
    "vs index pp": "How far ahead of the Nifty this stock is over the "
                   "holding period, in percentage points. Negative means "
                   "the index did better.",
    "Beat index by": "How far ahead of the Nifty this stock is over the "
                     "holding period, in percentage points. Negative means "
                     "the index did better and you would have been better "
                     "off in an index fund.",
    "OI chg %": "Change in open interest: how many futures and options "
                "contracts are open. Rising alongside price suggests new "
                "money coming in rather than old positions closing.",
    "Trend %": "How much the stock has moved over the holding period so "
               "far. Past movement, not a forecast.",
    "Volatility %": "How much this stock swings about in a year, in "
                    "percent. Higher means bigger moves both ways.",
    "Off high %": "How far below its recent peak the stock is. -20% means "
                  "it has fallen a fifth from its high.",
    "Below recent peak": "How far under its highest recent price the stock "
                         "sits. -20% means it has fallen a fifth.",
    "In range": "Where the price sits in its recent range. 1.0 is at the "
                "top, 0.0 at the bottom, 0.5 in the middle.",
    "Where in range": "1.0 means at the top of its recent range, 0.0 at "
                      "the bottom, 0.5 in the middle.",
    "Rank (unvalidated)": "Our ordering of these names. IMPORTANT: this was "
                          "tested against its own data and came out no "
                          "better than random, so do not read the top of "
                          "the list as 'the best ones'.",
    "Our ranking": "Our ordering. IMPORTANT: tested against its own data "
                   "and found no better than random - and at the longer "
                   "horizons it picked WORSE than simply buying everything. "
                   "Do not read the top as 'the best ones'.",
    "Horizon": "How long the position is meant to be held.",
    "Held": "How long the position is meant to be held.",
    "Verdict": "BUY means every check passed. NO BUY names the check that "
               "failed. Neither is a prediction of the price.",
    "Blocked by": "The specific check that failed. This is the useful part "
                  "of a NO BUY.",
    "Detail": "The most relevant supporting number for this horizon.",
    # --- the end-of-day sector tab ---
    "Sector": "The market sector being scored.",
    "ETF": "The fund used to price this sector. Blank means no tradable "
           "fund is mapped to it.",
    "Composite": "News sentiment and price momentum blended into one "
                 "number. Higher means more attention and more strength at "
                 "once. A description of today, not a forecast.",
    "News score": "How positive or negative recent coverage of this sector "
                  "reads, from -1 to +1. Zero means neutral or no news.",
    "Articles": "How many news items fed this sector's score.",
    "N": "How many news items about this sector arrived today.",
    "News today": "How positive or negative today's coverage reads, from "
                  "-1 to +1.",
    "Buzz %": "This sector's share of all the news collected. High buzz "
              "means the story is concentrated here.",
    "5d %": "Price change over the last 5 trading days, in percent.",
    "21d %": "Price change over the last 21 trading days - about a month.",
    "63d %": "Price change over the last 63 trading days - about a "
             "quarter.",
    "Momentum": "Recent price strength across several windows, combined. "
                "Past movement, not a forecast.",
    "Day %": "Price change so far today, in percent.",
    "Last hr %": "Price change over the last hour, in percent.",
    "Rank": "Where this sector sits in today's ordering. Not validated as "
            "predictive - read it as a description of current conditions.",
    "Action": "What the end-of-day rule suggests for this sector. It is a "
              "rule applied to today's numbers, not advice.",
    "Illiquid": "True means this fund trades too thinly to enter or exit "
                "reliably at the price shown.",
    # --- the pivot levels expander ---
    "Level": "A pivot level worked out from yesterday's high, low and "
             "close. S1-S3 sit below, R1-R3 above. Landmarks, not signals - "
             "we tested whether price turns at them and could not show it "
             "does.",
    "vs now": "How far this level sits from the current price, in percent. "
              "Negative means the level is below where price is now.",
}

# --- concepts worth an expander ------------------------------------------
CONCEPTS = {
    "How a stop and target are set":
        "The stop is placed using how much the stock typically moves, so "
        "normal wobble does not trigger it. The target is set at twice the "
        "stop distance, so one winner pays for two losers. Neither is a "
        "prediction - both are arithmetic from recent volatility.",
    "Why charges matter so much":
        "Every round trip costs brokerage, taxes and exchange fees. "
        "Intraday that is roughly 0.08% of the position; holding overnight "
        "it is about 0.23% because the tax applies to both the buy and the "
        "sell. On a small option position the flat per-order fee can be "
        "over 6% of the premium. You must clear that before you make "
        "anything, which is why the cost columns come first here.",
    "What the horizons mean":
        "Intraday means in and out the same day. Short is about two weeks, "
        "mid about three months, long about a year. They get different "
        "answers for the same stock, because a stock can be strong this "
        "fortnight and weak over the year.",
    "Why this tool will not predict for you":
        "We tested it properly. Four holding periods, about 855,000 past "
        "examples, using a model free to find any pattern it liked. It "
        "performed the same as the identical model trained on deliberately "
        "SCRAMBLED answers - and over a year, the scrambled version did "
        "better. So the ordering carries no forecasting power. What does "
        "hold up is the arithmetic: what a trade costs, how far the stock "
        "is likely to move, and whether the second covers the first.",
    "What the pivot levels are":
        "Every broker shows the same seven numbers, worked out from "
        "yesterday's high, low and close: a central pivot, three levels "
        "above it (R1-R3) and three below (S1-S3). The usual claim is that "
        "price tends to stall or turn at them. We checked on 79,000 cases "
        "and could not show it - comparing within the same trading day, a "
        "stop placed on a pivot was hit slightly MORE often, not less. So "
        "we show them as landmarks for orientation, and we do not place "
        "stops or targets on them.",
    "What 'volume vs normal' tells you":
        "Volume is how many shares changed hands. Comparing it to the same "
        "time of day in recent sessions tells you whether today is unusual. "
        "A price move on heavy volume has more people behind it than the "
        "same move on light volume.",
}


def column_help(name: str) -> str:
    """Plain-English tooltip for a column, or '' when there is none."""
    return COLUMNS.get(name, "")


def config_for(frame, st) -> dict:
    """Streamlit column_config giving every known column its tooltip.

    Built from the frame's actual columns so a renamed or removed column
    silently drops out rather than raising.
    """
    config = {}
    for name in getattr(frame, "columns", []):
        text = column_help(str(name))
        if text:
            config[name] = st.column_config.Column(str(name), help=text)
    return config
