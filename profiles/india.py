"""India (NSE) market profile: Nifty sectoral indices + Indian financial press.

Every feed URL below was live-verified on 2026-06-12. Symbols are Kite
tradingsymbols: all thirteen index names and all eleven ETFs were resolved
against Kite's public instrument master on 2026-09-09, when the last of the
Yahoo spellings was retired. Three global feeds are included because Indian
markets move on global cues.
"""
from __future__ import annotations

from models import SectorDef
from profiles import MarketProfile

_FEEDS: list[dict] = [
    {
        "name": "Economic Times Top Stories",
        "url": "https://economictimes.indiatimes.com/rssfeedstopstories.cms",
        "category": "business",
    },
    {
        "name": "Economic Times Markets",
        "url": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "category": "markets",
    },
    {
        "name": "Moneycontrol Top News",
        "url": "https://www.moneycontrol.com/rss/MCtopnews.xml",
        "category": "business",
    },
    {
        "name": "Moneycontrol Business",
        "url": "https://www.moneycontrol.com/rss/business.xml",
        "category": "business",
    },
    {
        "name": "Moneycontrol Buzzing Stocks",
        "url": "https://www.moneycontrol.com/rss/buzzingstocks.xml",
        "category": "markets",
    },
    {
        "name": "LiveMint Markets",
        "url": "https://www.livemint.com/rss/markets",
        "category": "markets",
    },
    {
        "name": "BusinessLine Markets",
        "url": "https://www.thehindubusinessline.com/markets/feeder/default.rss",
        "category": "markets",
    },
    {
        "name": "NDTV Profit",
        "url": "https://feeds.feedburner.com/ndtvprofit-latest",
        "category": "business",
    },
    {
        "name": "Google News India Business",
        "url": "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-IN&gl=IN&ceid=IN:en",
        "category": "business",
    },
    {
        "name": "Google News: Nifty & Sensex",
        "url": "https://news.google.com/rss/search?q=nifty+OR+sensex&hl=en-IN&gl=IN&ceid=IN:en",
        "category": "markets",
    },
    {
        "name": "Google News: RBI & FII",
        "url": "https://news.google.com/rss/search?q=RBI+OR+%22repo+rate%22+OR+FII&hl=en-IN&gl=IN&ceid=IN:en",
        "category": "markets",
    },
    # Global cues — Indian markets track US/world risk sentiment closely.
    {
        "name": "Google News Business",
        "url": "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en",
        "category": "world",
    },
    {
        "name": "BBC Business",
        "url": "http://feeds.bbci.co.uk/news/business/rss.xml",
        "category": "world",
    },
    {
        "name": "CNBC Top News",
        "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "category": "world",
    },
]

_SECTORS: dict[str, SectorDef] = {
    "Banks": SectorDef(
        name="Banks",
        etf="NIFTY BANK",
        keywords=[
            "hdfc bank", "icici", "icici bank", "kotak", "kotak mahindra",
            "axis bank", "indusind", "indusind bank", "yes bank", "idfc first",
            "federal bank", "rbl bank", "bandhan bank", "bank nifty", "nifty bank",
            "private bank", "private banks", "private lender", "private lenders",
            "npa", "npas", "bad loans", "gross npa", "asset quality",
            "credit growth", "loan growth", "deposit growth", "deposit rates",
            "casa", "casa ratio", "net interest margin", "nim",
            "repo rate", "rbi", "reserve bank of india", "monetary policy",
            "rate cut", "rate hike", "crr", "cash reserve ratio", "slr",
            "priority sector", "banking sector", "lender", "lenders",
            "provisioning", "slippages",
        ],
    ),
    "Financial Services": SectorDef(
        name="Financial Services",
        etf="NIFTY FIN SERVICE",
        keywords=[
            "bajaj finance", "bajaj finserv", "hdfc life", "sbi life", "icici lombard",
            "icici prudential", "lic", "life insurance corporation", "hdfc amc",
            "nippon amc", "cholamandalam", "chola", "shriram finance",
            "muthoot finance", "manappuram", "l&t finance", "piramal",
            "aditya birla capital", "paytm", "policybazaar", "pb fintech",
            "jio financial", "nbfc", "nbfcs", "shadow bank", "microfinance",
            "housing finance", "gold loan", "gold loans", "insurance premium",
            "insurer", "insurers", "general insurance", "life insurance",
            "mutual fund", "mutual funds", "sip inflows", "amc", "asset management",
            "broking", "brokerage", "zerodha", "demat", "wealth management",
            "credit card spends", "fintech", "irdai", "sebi",
        ],
    ),
    "IT": SectorDef(
        name="IT",
        etf="NIFTY IT",
        keywords=[
            "tcs", "tata consultancy", "infosys", "wipro", "hcl tech", "hcltech",
            "tech mahindra", "ltimindtree", "lti mindtree", "mphasis", "coforge",
            "persistent systems", "l&t technology", "ltts", "cyient", "birlasoft",
            "zensar", "happiest minds", "tata elxsi", "kpit", "oracle financial",
            "it services", "it sector", "it stocks", "it companies", "indian it",
            "software services", "software exporter", "software exporters",
            "outsourcing", "offshoring", "gcc", "global capability center",
            "deal wins", "deal pipeline", "large deals", "attrition",
            "h-1b", "h1b", "visa fees", "discretionary spending", "tech spending",
            "digital transformation", "artificial intelligence", "generative ai",
            "cloud migration", "client spending", "billing rates",
        ],
    ),
    "Pharma": SectorDef(
        name="Pharma",
        etf="NIFTY PHARMA",
        keywords=[
            "sun pharma", "cipla", "dr reddy", "dr. reddy", "divis", "divi's",
            "lupin", "aurobindo", "aurobindo pharma", "zydus", "torrent pharma",
            "alkem", "glenmark", "biocon", "laurus labs", "ipca", "natco",
            "mankind pharma", "abbott india", "gland pharma", "granules",
            "apollo hospitals", "max healthcare", "fortis", "narayana health",
            "usfda", "us fda", "fda inspection", "form 483", "warning letter",
            "anda", "abbreviated new drug", "api", "active pharmaceutical",
            "generics", "generic drugs", "complex generics", "biosimilar",
            "biosimilars", "speciality drugs", "drug pricing", "price erosion",
            "cdmo", "contract manufacturing", "clinical trial", "pharma sector",
            "pharma stocks", "vaccine", "oncology", "formulations", "nppa",
        ],
    ),
    "Auto": SectorDef(
        name="Auto",
        etf="NIFTY AUTO",
        keywords=[
            "maruti", "maruti suzuki", "tata motors", "mahindra", "m&m",
            "mahindra & mahindra", "bajaj auto", "hero motocorp", "tvs",
            "tvs motor", "eicher", "eicher motors", "royal enfield", "ashok leyland",
            "ola electric", "ather", "hyundai india", "kia india", "mg motor",
            "bosch india", "motherson", "samvardhana", "bharat forge", "exide",
            "amara raja", "sona blw", "sona comstar", "uno minda", "tyre",
            "tyres", "apollo tyres", "mrf", "ceat",
            "auto sales", "vehicle sales", "passenger vehicle", "passenger vehicles",
            "two-wheeler", "two-wheelers", "three-wheeler", "commercial vehicle",
            "commercial vehicles", "suv", "suvs", "ev", "electric vehicle",
            "electric vehicles", "electric two-wheeler", "auto sector",
            "automaker", "automakers", "auto ancillary", "pli scheme",
            "fame scheme", "festive demand", "dealer inventory",
        ],
    ),
    "FMCG": SectorDef(
        name="FMCG",
        etf="NIFTY FMCG",
        keywords=[
            "hul", "hindustan unilever", "itc", "britannia", "nestle india",
            "dabur", "marico", "godrej consumer", "colgate-palmolive",
            "colgate india", "emami", "jyothy labs", "tata consumer",
            "varun beverages", "united spirits", "united breweries", "radico",
            "patanjali", "adani wilmar", "bikaji", "honasa", "mamaearth",
            "fmcg", "fast-moving consumer", "consumer goods", "consumer staples",
            "rural demand", "rural consumption", "urban demand", "volume growth",
            "price hikes", "input costs", "palm oil", "monsoon", "rainfall",
            "soaps", "detergents", "personal care", "packaged food",
            "packaged foods", "biscuits", "noodles", "beverages", "cigarettes",
            "staples demand", "distribution network", "general trade",
            "quick commerce",
        ],
    ),
    "Metal": SectorDef(
        name="Metal",
        etf="NIFTY METAL",
        keywords=[
            "tata steel", "jsw steel", "hindalco", "vedanta", "sail",
            "steel authority", "nmdc", "jindal", "jindal steel", "jspl",
            "nalco", "national aluminium", "hindustan zinc", "hindustan copper",
            "moil", "ratnamani", "apl apollo", "welspun corp",
            "steel prices", "steel demand", "steel output", "steel exports",
            "aluminium", "aluminium prices", "copper", "copper prices", "zinc",
            "iron ore", "coking coal", "base metals", "metal stocks",
            "metal sector", "lme", "china demand", "china stimulus",
            "export duty", "import duty", "safeguard duty", "anti-dumping",
            "smelter", "mining", "ferrous", "non-ferrous", "alloy",
            "galvanised", "pellet",
        ],
    ),
    "Energy": SectorDef(
        name="Energy",
        etf="NIFTY ENERGY",
        keywords=[
            "reliance", "reliance industries", "ril", "ongc", "oil india",
            "ioc", "indian oil", "bpcl", "hpcl", "gail", "petronet",
            "igl", "indraprastha gas", "mgl", "mahanagar gas", "gujarat gas",
            "coal india", "ntpc", "power grid", "nhpc", "sjvn", "tata power",
            "adani green", "adani power", "adani energy", "adani total gas",
            "jsw energy", "torrent power", "suzlon", "inox wind",
            "crude", "crude oil", "brent", "oil prices", "lng", "natural gas",
            "refinery", "refining", "refining margins", "grm", "petrochemical",
            "petrochemicals", "fuel prices", "petrol", "diesel", "opec",
            "windfall tax", "city gas", "solar", "solar power", "renewable",
            "renewables", "wind energy", "green hydrogen", "thermal power",
            "power demand", "electricity demand", "discom", "coal production",
        ],
    ),
    "Realty": SectorDef(
        name="Realty",
        etf="NIFTY REALTY",
        keywords=[
            "dlf", "godrej properties", "oberoi realty", "lodha", "macrotech",
            "prestige estates", "brigade", "brigade enterprises", "sobha",
            "phoenix mills", "sunteck", "mahindra lifespace", "signature global",
            "raymond realty", "anant raj", "embassy reit", "mindspace reit",
            "real estate", "realty", "realty stocks", "property market",
            "property prices", "housing demand", "housing sales", "home sales",
            "residential sales", "residential launches", "new launches",
            "affordable housing", "luxury housing", "premium housing",
            "commercial real estate", "office leasing", "office space",
            "mall", "rera", "stamp duty", "circle rates", "home loan",
            "home loans", "mortgage", "builder", "developers", "land parcel",
            "redevelopment", "township",
        ],
    ),
    "Infrastructure": SectorDef(
        name="Infrastructure",
        etf="NIFTY INFRA",
        keywords=[
            "l&t", "larsen", "larsen & toubro", "adani ports", "gmr",
            "gmr airports", "irb", "irb infra", "kalpataru", "kec international",
            "ncc", "afcons", "rvnl", "rail vikas", "ircon", "nbcc", "hcc",
            "dilip buildcon", "pnc infratech", "knr constructions",
            "ultratech", "acc", "ambuja", "shree cement", "dalmia bharat",
            "jk cement", "ramco cements", "cement prices", "cement demand",
            "nhai", "highways", "highway construction", "road projects",
            "expressway", "metro rail", "railways", "railway capex",
            "dedicated freight corridor", "port", "ports", "airport", "airports",
            "infrastructure", "infra spending", "capex", "capital expenditure",
            "order book", "order inflows", "epc", "construction", "tunnel",
            "bharatmala", "smart cities", "national infrastructure pipeline",
        ],
    ),
    "PSU Banks": SectorDef(
        name="PSU Banks",
        etf="NIFTY PSU BANK",
        keywords=[
            "sbi", "state bank", "state bank of india", "pnb", "punjab national",
            "punjab national bank", "bank of baroda", "bob", "canara bank",
            "union bank", "union bank of india", "indian bank", "bank of india",
            "central bank of india", "uco bank", "indian overseas bank", "iob",
            "bank of maharashtra", "punjab & sind bank",
            "psu bank", "psu banks", "psu bank stocks", "public sector bank",
            "public sector banks", "public sector lender", "public sector lenders",
            "state-owned bank", "state-owned banks", "state-run bank",
            "state-run banks", "nationalised bank", "nationalised banks",
            "recapitalisation", "recapitalization", "bank merger",
            "divestment", "disinvestment", "stake dilution",
            "government stake", "write-off", "loan recovery",
            "bad bank", "narcl",
        ],
    ),
    "Media": SectorDef(
        name="Media",
        etf="NIFTY MEDIA",
        keywords=[
            "zee", "zee entertainment", "zeel", "sun tv", "pvr inox", "pvr",
            "nazara", "nazara technologies", "network18", "tv18", "tips music",
            "saregama", "dish tv", "hathway", "den networks", "balaji telefilms",
            "prime focus", "jiocinema", "jiostar", "disney star", "sony india",
            "netflix india", "hotstar",
            "multiplex", "multiplexes", "box office", "box-office", "bollywood",
            "film industry", "movie release", "theatrical release", "footfalls",
            "occupancy", "ott", "ott platform", "ott platforms", "streaming",
            "subscriber growth", "broadcasting", "broadcaster", "broadcasters",
            "television", "trai tariff", "advertising revenue", "ad revenue",
            "ad spends", "media stocks", "media sector", "content pipeline",
            "gaming", "esports",
        ],
    ),
}

_LEXICON: dict[str, float] = {
    "oversubscribed": 1.8,
    "pledged": -1.5,
    "delisting": -1.0,
    "demerger": 0.5,
    "divestment": 0.8,
    "recapitalisation": 1.2,
}

_PHRASES: dict[str, float] = {
    "upper circuit": 2.0,
    "lower circuit": -2.0,
    "fii selling": -1.5,
    "fii buying": 1.5,
    "fii inflows": 1.5,
    "fii outflows": -1.5,
    "dii buying": 1.2,
    "rate cut": 1.5,
    "rate hike": -1.2,
    "rate pause": 0.8,
    "repo rate cut": 1.5,
    "monsoon deficit": -1.5,
    "normal monsoon": 1.2,
    "above normal monsoon": 1.5,
    "gst cut": 1.5,
    "record high": 1.8,
    "all-time high": 1.8,
    "block deal": 0.3,
    "stake sale": -0.5,
    "open offer": 0.8,
}

# Tradeable NSE sector ETFs (verified with volume 2026-09-08, resolved
# against Kite's instrument master 2026-09-09).
# Media has no listed sector ETF and so gets no entry.
#
# These tickers are also the momentum source (see market_data), so each one
# must be the fund actually bought - a sibling ETF tracking the same index
# still has its own premium, tracking error and liquidity.
_TRADE_ETFS: dict[str, str] = {
    "Banks": "BANKBEES",
    "Financial Services": "BFSI",
    "IT": "ITETF",
    "Pharma": "PHARMABEES",
    "Auto": "AUTOBEES",
    "FMCG": "FMCGIETF",
    "Metal": "METALIETF",
    "Energy": "OILIETF",
    "Infrastructure": "INFRAIETF",
    "PSU Banks": "PSUBANK",
    "Realty": "MOREALTY",
}

IN_PROFILE = MarketProfile(
    key="IN",
    label="India (NSE)",
    currency="INR",
    feeds=_FEEDS,
    sectors=_SECTORS,
    lexicon=_LEXICON,
    phrases=_PHRASES,
    trade_etfs=_TRADE_ETFS,
)
