"""United States market profile: SPDR sector ETFs + US/world finance feeds.

Every feed URL below was live-verified on 2026-06-11: HTTP 200 with a
browser-like User-Agent and a body containing an RSS/Atom root element.
"""
from __future__ import annotations

from models import SectorDef
from profiles import MarketProfile

_FEEDS: list[dict] = [
    {
        "name": "Google News Business",
        "url": "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en",
        "category": "business",
    },
    {
        "name": "Google News World",
        "url": "https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-US&gl=US&ceid=US:en",
        "category": "world",
    },
    {
        "name": "BBC Business",
        "url": "http://feeds.bbci.co.uk/news/business/rss.xml",
        "category": "business",
    },
    {
        "name": "BBC World",
        "url": "http://feeds.bbci.co.uk/news/world/rss.xml",
        "category": "world",
    },
    {
        "name": "CNBC Top News",
        "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "category": "business",
    },
    {
        "name": "CNBC World Markets",
        "url": "https://www.cnbc.com/id/15839135/device/rss/rss.html",
        "category": "markets",
    },
    {
        "name": "Yahoo Finance",
        "url": "https://finance.yahoo.com/news/rssindex",
        "category": "markets",
    },
    {
        "name": "MarketWatch Top Stories",
        "url": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
        "category": "markets",
    },
    {
        "name": "The Guardian Business",
        "url": "https://www.theguardian.com/uk/business/rss",
        "category": "business",
    },
    {
        "name": "Google News: Stocks & Fed",
        "url": "https://news.google.com/rss/search?q=stock+market+OR+federal+reserve&hl=en-US&gl=US&ceid=US:en",
        "category": "markets",
    },
    {
        "name": "Google News: Oil & Commodities",
        "url": "https://news.google.com/rss/search?q=oil+prices+OR+OPEC+OR+commodities&hl=en-US&gl=US&ceid=US:en",
        "category": "markets",
    },
]

_SECTORS: dict[str, SectorDef] = {
    "Technology": SectorDef(
        name="Technology",
        etf="XLK",
        keywords=[
            "nvidia", "semiconductor", "semiconductors", "chip", "chips", "chipmaker",
            "apple", "iphone", "microsoft", "software", "cloud computing",
            "artificial intelligence", "ai chips", "generative ai", "machine learning",
            "tsmc", "intel", "amd", "qualcomm", "broadcom", "micron",
            "texas instruments", "asml", "arm holdings", "data center", "data centers",
            "cybersecurity", "oracle", "salesforce", "adobe", "ibm", "sap", "palantir",
            "dell", "cisco", "saas", "gpu", "gpus", "quantum computing", "big tech",
            "silicon valley", "openai", "smartphone", "smartphones", "tech stocks",
        ],
    ),
    "Financials": SectorDef(
        name="Financials",
        etf="XLF",
        keywords=[
            "bank", "banks", "banking", "jpmorgan", "goldman sachs", "goldman",
            "morgan stanley", "citigroup", "wells fargo", "bank of america",
            "berkshire hathaway", "blackrock", "charles schwab", "visa", "mastercard",
            "american express", "paypal", "interest rate", "interest rates",
            "rate cut", "rate cuts", "rate hike", "rate hikes", "federal reserve",
            "fed", "fomc", "central bank", "central banks", "bond yields",
            "treasury yields", "yield curve", "lending", "loans", "mortgage rates",
            "insurance", "insurer", "insurers", "credit card", "hedge fund",
            "private equity", "asset management", "fintech", "wall street",
            "deposits", "fdic", "stress test",
        ],
    ),
    "Energy": SectorDef(
        name="Energy",
        etf="XLE",
        keywords=[
            "oil", "opec", "crude", "brent", "wti", "shale", "exxon", "exxon mobil",
            "chevron", "conocophillips", "occidental", "halliburton", "schlumberger",
            "saudi aramco", "lng", "natural gas", "refinery", "refineries", "refining",
            "oilfield", "drilling", "rig count", "petroleum", "gasoline", "diesel",
            "oil prices", "oil production", "oil output", "barrel", "barrels",
            "pipeline", "pipelines", "fracking", "offshore drilling", "oil and gas",
            "fossil fuel", "fossil fuels", "energy prices", "coal",
        ],
    ),
    "Healthcare": SectorDef(
        name="Healthcare",
        etf="XLV",
        keywords=[
            "pfizer", "moderna", "johnson & johnson", "merck", "eli lilly", "novartis",
            "astrazeneca", "abbvie", "amgen", "bristol myers", "unitedhealth", "cvs",
            "cigna", "humana", "drug", "drugs", "drugmaker", "drugmakers",
            "pharmaceutical", "pharmaceuticals", "pharma", "biotech", "biotechnology",
            "vaccine", "vaccines", "fda", "clinical trial", "clinical trials",
            "obesity drug", "weight-loss drug", "ozempic", "wegovy", "oncology",
            "cancer treatment", "cancer drug", "medicare", "medicaid",
            "health insurance", "hospital", "hospitals", "medical device",
            "medical devices", "gene therapy", "prescription", "telehealth",
            "disease outbreak",
        ],
    ),
    "Industrials": SectorDef(
        name="Industrials",
        etf="XLI",
        keywords=[
            "boeing", "airbus", "lockheed", "lockheed martin", "raytheon", "rtx",
            "northrop", "general electric", "ge aerospace", "caterpillar", "deere",
            "honeywell", "union pacific", "csx", "railroad", "railroads", "airline",
            "airlines", "delta air lines", "united airlines", "american airlines",
            "fedex", "ups", "freight", "logistics", "trucking", "cargo",
            "defense contractor", "defense contractors", "defense spending",
            "military spending", "aerospace", "manufacturing", "factory orders",
            "industrial production", "machinery", "construction equipment",
            "infrastructure", "infrastructure spending", "uber", "waste management",
            "jet engine", "supply chain",
        ],
    ),
    "Consumer Discretionary": SectorDef(
        name="Consumer Discretionary",
        etf="XLY",
        keywords=[
            "amazon", "tesla", "home depot", "lowe's", "mcdonald's", "nike",
            "starbucks", "booking holdings", "marriott", "hilton", "airbnb",
            "doordash", "chipotle", "ford", "general motors", "toyota", "stellantis",
            "automaker", "automakers", "auto sales", "electric vehicle",
            "electric vehicles", "ev sales", "retail", "retail sales", "retailer",
            "retailers", "e-commerce", "online shopping", "consumer spending",
            "consumer confidence", "holiday shopping", "black friday", "restaurant",
            "restaurants", "hotel", "hotels", "travel demand", "theme park",
            "luxury goods", "lvmh", "apparel", "footwear", "casino", "casinos",
            "cruise line", "cruise lines", "dealership",
        ],
    ),
    "Consumer Staples": SectorDef(
        name="Consumer Staples",
        etf="XLP",
        keywords=[
            "walmart", "costco", "procter & gamble", "coca-cola", "coca cola",
            "pepsico", "pepsi", "colgate", "kimberly-clark", "general mills",
            "kraft heinz", "mondelez", "kellogg", "tyson foods", "hershey",
            "unilever", "nestle", "kroger", "dollar general", "dollar tree",
            "grocery", "groceries", "grocer", "supermarket", "supermarkets",
            "food prices", "food maker", "packaged food", "beverage", "beverages",
            "soda", "snack foods", "household products", "tobacco", "philip morris",
            "altria", "cigarettes", "toothpaste", "detergent", "diapers",
            "consumer staples",
        ],
    ),
    "Utilities": SectorDef(
        name="Utilities",
        etf="XLU",
        keywords=[
            "utility", "utilities", "electricity", "electric grid", "power grid",
            "grid operator", "power plant", "power plants", "nextera", "duke energy",
            "southern company", "dominion energy", "american electric power",
            "exelon", "constellation energy", "electricity prices",
            "electricity demand", "power demand", "power prices", "nuclear power",
            "nuclear plant", "nuclear reactor", "nuclear reactors",
            "renewable energy", "renewables", "solar power", "wind power",
            "solar farm", "wind farm", "hydroelectric", "geothermal", "blackout",
            "blackouts", "power outage", "power outages", "transmission lines",
            "electrification", "megawatt", "gigawatt", "water utility",
        ],
    ),
    "Materials": SectorDef(
        name="Materials",
        etf="XLB",
        keywords=[
            "copper", "gold", "gold prices", "gold miner", "silver", "lithium",
            "nickel", "cobalt", "zinc", "iron ore", "steel", "steelmaker",
            "steelmakers", "aluminum", "aluminium", "mining", "miner", "miners",
            "rio tinto", "bhp", "glencore", "freeport", "newmont", "dow inc",
            "dow chemical", "dupont", "chemicals", "chemical maker",
            "chemical makers", "fertilizer", "fertilizers", "potash", "cement",
            "lumber", "timber", "commodity prices", "commodities", "rare earth",
            "rare earths", "smelter", "precious metals", "base metals", "linde",
            "air products", "sherwin-williams", "packaging",
        ],
    ),
    "Real Estate": SectorDef(
        name="Real Estate",
        etf="XLRE",
        keywords=[
            "real estate", "reit", "reits", "property market", "property prices",
            "property values", "property developer", "property developers",
            "home prices", "housing", "housing market", "housing starts",
            "home sales", "homebuilder", "homebuilders", "housing affordability",
            "commercial real estate", "office space", "office vacancies", "rent",
            "rents", "rental", "rentals", "landlord", "landlords", "mortgage",
            "mortgages", "mortgage rates", "foreclosure", "foreclosures",
            "vacancy rate", "leasing", "tenants", "prologis", "american tower",
            "equinix", "simon property", "public storage", "realty income",
            "zillow", "redfin", "apartment", "apartments", "condo", "evergrande",
            "country garden",
        ],
    ),
    "Communication Services": SectorDef(
        name="Communication Services",
        etf="XLC",
        keywords=[
            "google", "alphabet", "meta", "facebook", "instagram", "whatsapp",
            "netflix", "disney", "walt disney", "youtube", "tiktok", "streaming",
            "telecom", "telecoms", "verizon", "at&t", "t-mobile", "comcast",
            "warner bros", "paramount", "spotify", "social media", "advertising",
            "ad revenue", "ad spending", "media company", "media companies",
            "broadband", "5g", "wireless carrier", "wireless carriers",
            "video game", "video games", "esports", "electronic arts", "take-two",
            "roblox", "snapchat", "pinterest", "reddit", "box office", "hollywood",
            "subscribers", "charter communications",
        ],
    ),
}

# Every US sector trades directly via its SPDR ETF.
_TRADE_ETFS: dict[str, str] = {
    name: sector.etf for name, sector in _SECTORS.items()
}

US_PROFILE = MarketProfile(
    key="US",
    label="United States",
    currency="USD",
    feeds=_FEEDS,
    sectors=_SECTORS,
    lexicon={},
    phrases={},
    trade_etfs=_TRADE_ETFS,
)
