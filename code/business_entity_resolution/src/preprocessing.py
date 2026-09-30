"""
Data Preprocessing & Normalization Module
Handles multilingual text normalization, state extraction, address standardization,
and street number extraction across US, India, and France.
"""

import re
import pandas as pd
from unidecode import unidecode
from tqdm.auto import tqdm

# ==============================================================================
# 1. State Dictionaries (US, India, France)
# ==============================================================================

US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
    "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "DC": "District of Columbia", "PR": "Puerto Rico"
}

US_STATE_NAMES = {v.lower(): v for v in US_STATES.values()}
US_STATE_CODES = {k.lower(): v for k, v in US_STATES.items()}

INDIA_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat",
    "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "West Bengal", "Delhi", "Jammu and Kashmir", "Ladakh", "Puducherry",
    "Chandigarh", "Andaman and Nicobar Islands", "Dadra and Nagar Haveli and Daman and Diu"
]
INDIA_STATE_NAMES = {s.lower(): s for s in INDIA_STATES}

FRANCE_DEP_TO_REGION = {
    # Hauts-de-France (Nord, Pas-de-Calais, Aisne, Somme, Oise + key cities)
    "nord": "Hauts-de-France", "pas-de-calais": "Hauts-de-France", "aisne": "Hauts-de-France",
    "somme": "Hauts-de-France", "oise": "Hauts-de-France",
    "lille": "Hauts-de-France", "tourcoing": "Hauts-de-France", "dunkerque": "Hauts-de-France",
    "calais": "Hauts-de-France", "roubaix": "Hauts-de-France",

    # Nouvelle-Aquitaine (Gironde, Dordogne, Landes, etc. + key cities)
    "gironde": "Nouvelle-Aquitaine", "dordogne": "Nouvelle-Aquitaine", "landes": "Nouvelle-Aquitaine",
    "lot-et-garonne": "Nouvelle-Aquitaine", "pyrenees-atlantiques": "Nouvelle-Aquitaine",
    "charente": "Nouvelle-Aquitaine", "charente-maritime": "Nouvelle-Aquitaine",
    "bordeaux": "Nouvelle-Aquitaine", "pessac": "Nouvelle-Aquitaine", "merignac": "Nouvelle-Aquitaine",
    "la teste-de-buch": "Nouvelle-Aquitaine", "lege-cap-ferret": "Nouvelle-Aquitaine",

    # Pays de la Loire (Loire-Atlantique, Maine-et-Loire, etc. + key cities)
    "loire-atlantique": "Pays de la Loire", "maine-et-loire": "Pays de la Loire",
    "mayenne": "Pays de la Loire", "sarthe": "Pays de la Loire", "vendee": "Pays de la Loire",
    "nantes": "Pays de la Loire", "saint-nazaire": "Pays de la Loire", "pornic": "Pays de la Loire",
    "la baule-escoublac": "Pays de la Loire", "saint-herblain": "Pays de la Loire",

    # Ile-de-France (Paris + suburban departments)
    "paris": "Ile-de-France", "seine-et-marne": "Ile-de-France", "yvelines": "Ile-de-France",
    "essonne": "Ile-de-France", "hauts-de-seine": "Ile-de-France", "seine-saint-denis": "Ile-de-France",
    "val-de-marne": "Ile-de-France", "val-d-oise": "Ile-de-France"
}

FRANCE_REGIONS_LIST = [
    "Auvergne-Rhone-Alpes", "Bourgogne-Franche-Comte", "Bretagne", "Centre-Val de Loire",
    "Corse", "Grand Est", "Hauts-de-France", "Ile-de-France", "Normandie", "Nouvelle-Aquitaine",
    "Occitanie", "Pays de la Loire", "Provence-Alpes-Cote d'Azur"
]
FRANCE_REGIONS_LOOKUP = {r.lower(): r for r in FRANCE_REGIONS_LIST}

# ==============================================================================
# 2. Address Expansions
# ==============================================================================

ADDR_EXPANSIONS = [
    (r"\b(rd|rd\.)\b", "road"),
    (r"\b(st|st\.)\b", "street"),
    (r"\b(ave|ave\.|av|av\.)\b", "avenue"),
    (r"\b(blvd|blvd\.)\b", "boulevard"),
    (r"\b(dr|dr\.)\b", "drive"),
    (r"\b(ln|ln\.)\b", "lane"),
    (r"\b(ct|ct\.)\b", "court"),
    (r"\b(pl|pl\.)\b", "place"),
    (r"\b(sq|sq\.)\b", "square"),
    (r"\b(hwy|hwy\.)\b", "highway"),
    (r"\b(pkwy|pkwy\.)\b", "parkway"),
    (r"\b(ste|ste\.|suite)\b", "suite"),
    (r"\b(apt|apt\.|apartment)\b", "apartment"),
    (r"\b(fl|fl\.|floor)\b", "floor"),
    (r"\b(bldg|bldg\.|building)\b", "building"),
    (r"\b(dept|dept\.|department)\b", "dept"),
    (r"\b(opp|opp\.|opposite)\b", "opposite"),
    (r"\b(nr|nr\.)\b", "near"),
    (r"\b(ext|ext\.)\b", "extension"),
    (r"\b(dist|dist\.)\b", "district"),
    (r"\b(sec|sec\.)\b", "sector"),
    (r"\b(col|col\.)\b", "colony"),
    (r"\b(nagar)\b", "nagar"),
    (r"\b(marg)\b", "marg"),
]

FRANCE_ADDR_EXPANSIONS = [
    (r"\b(r|r\.)\b", "rue"),
    (r"\b(av|av\.|ave)\b", "avenue"),
    (r"\b(bd|bd\.|bvd)\b", "boulevard"),
    (r"\b(all|all\.)\b", "allee"),
    (r"\b(ch|ch\.|che)\b", "chemin"),
    (r"\b(rte|rte\.)\b", "route"),
    (r"\b(imp|imp\.)\b", "impasse"),
    (r"\b(pl|pl\.)\b", "place"),
    (r"\b(sq|sq\.)\b", "square"),
    (r"\b(crs|crs\.)\b", "cours"),
    (r"\b(pass|pass\.)\b", "passage"),
    (r"\b(rpt|rpt\.)\b", "rond point"),
    (r"\b(zi|z\.i\.)\b", "zone industrielle"),
    (r"\b(za|z\.a\.)\b", "zone d activite"),
    (r"\b(zac|z\.a\.c\.)\b", "zone d amenagement concerte"),
    (r"\b(st|st\.)\b", "saint"),
    (r"\b(ste|ste\.)\b", "sainte"),
    (r"\b(bat|bat\.)\b", "batiment"),
    (r"\b(res|res\.)\b", "residence"),
    (r"\b(imm|imm\.)\b", "immeuble"),
    (r"\b(etg|etg\.)\b", "etage"),
    (r"\b(app|apt|appt)\b", "appartement"),
]

# ==============================================================================
# 3. Normalization Functions
# ==============================================================================

def clean_universal(text):
    if not isinstance(text, str):
        return ""
    text = unidecode(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def normalize_name(name):
    clean = clean_universal(name)
    clean = re.sub(r"\b(corp|corporation|inc|incorporated|llc|ltd|limited|pvt|private|co|company|gmbh|sa|sas|sarl)\b", "", clean)
    clean = re.sub(r"\b(the|and|of|in|at|on|for|by|de|la|le|du|des|et)\b", "", clean)
    return re.sub(r"\s+", " ", clean).strip()

def normalize_address(addr, country="US"):
    clean = clean_universal(addr)
    for pattern, repl in ADDR_EXPANSIONS:
        clean = re.sub(pattern, repl, clean)
    if str(country).strip() == "France":
        for pattern, repl in FRANCE_ADDR_EXPANSIONS:
            clean = re.sub(pattern, repl, clean)
    return re.sub(r"\s+", " ", clean).strip()

def extract_state_info(addr, country):
    if not isinstance(addr, str) or not addr.strip():
        return (None, 0)
    country = str(country).strip()

    if country == "US":
        parts = [p.strip() for p in addr.split(",") if p.strip()]
        for p in reversed(parts[-3:]):
            tokens = p.split()
            for t in tokens:
                t_clean = re.sub(r"[^a-zA-Z]", "", t).lower()
                if t_clean in US_STATE_CODES:
                    return (US_STATE_CODES[t_clean], 1)
        addr_lower = addr.lower()
        for s_lower, s_name in US_STATE_NAMES.items():
            if re.search(r"\b" + re.escape(s_lower) + r"\b", addr_lower):
                return (s_name, 1)
        return (None, 0)

    elif country == "India":
        addr_lower = addr.lower()
        parts = [p.strip() for p in addr_lower.split(",") if p.strip()]
        for p in reversed(parts[-2:]):
            cleaned_p = re.sub(r"[^a-z\s]", "", p).strip()
            if cleaned_p in INDIA_STATE_NAMES:
                return (INDIA_STATE_NAMES[cleaned_p], 1)
        for s_lower, s_name in INDIA_STATE_NAMES.items():
            if re.search(r"\b" + re.escape(s_lower) + r"\b", addr_lower):
                return (s_name, 1)
        return (None, 0)

    elif country == "France":
        addr_lower = unidecode(addr).lower()
        parts = [p.strip() for p in addr_lower.split(",") if p.strip()]
        for r_lower, r_full in FRANCE_REGIONS_LOOKUP.items():
            if r_lower in addr_lower:
                return (r_full, 1)
        check_tokens = []
        if len(parts) >= 2:
            check_tokens.extend([parts[-1], parts[-2]])
        elif len(parts) == 1:
            check_tokens.append(parts[0])
        for tok in check_tokens:
            cleaned_tok = re.sub(r"[^a-z\s-]", "", tok).strip()
            if cleaned_tok in FRANCE_DEP_TO_REGION:
                return (FRANCE_DEP_TO_REGION[cleaned_tok], 1)
        for k, v in FRANCE_DEP_TO_REGION.items():
            if len(k) > 3 and re.search(r"\b" + re.escape(k) + r"\b", addr_lower):
                return (v, 1)
        return (None, 0)

    return (None, 0)

def extract_house_number(addr):
    if not isinstance(addr, str) or not addr.strip():
        return (None, 0)
    first_part = addr.split(",")[0].strip()
    match = re.search(r"\b(\d{1,6}[a-zA-Z]?)\b", first_part)
    if match:
        num = match.group(1).lower()
        if len(num) >= 5 and re.match(r"^\d{5,6}$", num):
            return (None, 0)
        return (num, 1)
    return (None, 0)

def preprocess_dataset(df, desc="Dataset"):
    """Applies end-to-end normalization to a dataset with progress tracking."""
    print(f"--- Preprocessing {desc} ({len(df):,} records) ---")
    
    tqdm.pandas(desc=f"[{desc}] States")
    state_tuples = [extract_state_info(a, c) for a, c in zip(df["business_address"], df["country"])]
    df["state"] = [t[0] for t in state_tuples]
    df["has_state"] = [t[1] for t in state_tuples]

    tqdm.pandas(desc=f"[{desc}] Names")
    df["clean_name"] = df["business_name"].fillna("").astype(str).progress_apply(normalize_name)

    tqdm.pandas(desc=f"[{desc}] Addresses")
    df["clean_address"] = [normalize_address(a, c) for a, c in zip(df["business_address"], df["country"])]

    tqdm.pandas(desc=f"[{desc}] House Numbers")
    hn_tuples = [extract_house_number(a) for a in df["business_address"]]
    df["house_number"] = [t[0] for t in hn_tuples]
    df["has_house_number"] = [t[1] for t in hn_tuples]

    return df
