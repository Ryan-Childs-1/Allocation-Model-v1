from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


# One-based Excel columns from the known Sportsman's allocation layout.
# These are used as fallbacks only. Header names are preferred.
DEFAULT_EXCEL_COLUMNS: Dict[str, int] = {
    "line_id": 1,
    "department_id": 3,
    "class_id": 5,
    "site": 11,
    "item": 15,
    "upc": 18,
    "last_sold_date": 28,
    "last_receipt": 29,
    "l30": 37,       # AK
    "lw": 38,        # AL
    "d60": 39,       # AM
    "d30": 40,       # AN
    "ttm": 41,       # AO
    "supply": 43,    # AQ
    "qoh": 45,
    "dc_avail": 50,  # AX
    "retail": 55,
    "cost": 56,
    "gm_pct": 57,
    "atg_retail": 58,
    "square_footage": 59,
    "orgs": 60,
    "dc_flm": 61,
    "mil": 62,
    "flm": 63,       # BK
    "proj_demand": 65, # BM
    "alloc_rec": 66, # BN
    "flag": 70,      # BR
    "final_alloc": 71, # BS
    "left_dc": 72,   # BT
    "final_supply": 73, # BU
    "demand_check": 78, # BZ
    "helper": 79,    # CA
}

HEADER_ALIASES: Dict[str, List[str]] = {
    "line_id": ["line id", "lineid"],
    "department_id": ["department id", "dept id", "department"],
    "class_id": ["class id", "class"],
    "site": ["site", "store", "store id", "location"],
    "item": ["item", "item id", "sku"],
    "upc": ["upc"],
    "last_sold_date": ["last sold date", "last sold"],
    "last_receipt": ["last receipt", "last receipt date"],
    "l30": ["l30"],
    "lw": ["lw"],
    "d60": ["d60", "demand 60", "demand60", "60 day demand", "60-day demand"],
    "d30": ["d30", "demand 30", "demand30", "30 day demand", "30-day demand"],
    "ttm": ["ttm"],
    "supply": ["supply", "current supply"],
    "qoh": ["qoh", "qty on hand", "quantity on hand"],
    "dc_avail": ["dc avail", "dc available", "dc availability", "dc avl", "dc_avail"],
    "retail": ["retail"],
    "cost": ["cost"],
    "gm_pct": ["gm pct", "gm%", "gross margin pct", "gm percent"],
    "atg_retail": ["atg retail"],
    "square_footage": ["square footage", "sq ft", "sqft"],
    "orgs": ["orgs", "org"],
    "dc_flm": ["dc flm", "dcflm"],
    "mil": ["mil"],
    "flm": ["flm", "allocation multiple", "alloc multiple"],
    "proj_demand": ["proj. demand", "proj demand", "projected demand"],
    "alloc_rec": ["alloc. rec.", "alloc rec", "allocation rec", "allocation recommendation", "alloc recommendation"],
    "flag": ["flag", "review flag", "allocation flag"],
    "final_alloc": ["final alloc.", "final alloc", "final allocation"],
    "left_dc": ["left dc", "left in dc", "leftdc"],
    "final_supply": ["final supply"],
    "demand_check": ["demand check"],
    "helper": ["helper"],
}

NUMERIC_FIELDS = [
    "line_id", "department_id", "class_id", "l30", "lw", "d60", "d30", "ttm", "supply", "qoh",
    "dc_avail", "retail", "cost", "gm_pct", "atg_retail", "square_footage", "orgs", "dc_flm",
    "mil", "flm", "proj_demand", "alloc_rec", "left_dc", "final_supply", "demand_check", "helper",
]
CATEGORICAL_FIELDS = ["site", "item", "upc", "flag"]


def normalize_header(value) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace("\r", " ").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("_", " ")
    text = text.rstrip(":")
    return text


def excel_col_to_letters(n: int) -> str:
    out = ""
    while n:
        n, r = divmod(n - 1, 26)
        out = chr(65 + r) + out
    return out


def unique_columns(columns: Iterable) -> List[str]:
    counts = {}
    result = []
    for raw in columns:
        base = str(raw).strip() if str(raw).strip() else "Unnamed"
        c = counts.get(base, 0)
        if c == 0:
            result.append(base)
        else:
            result.append(f"{base}__{c+1}")
        counts[base] = c + 1
    return result


def detect_header_row(raw_df, max_scan_rows: int = 30) -> int:
    """Return zero-based row index most likely to contain headers."""
    best_i = 0
    best_score = -1
    aliases = {a for vals in HEADER_ALIASES.values() for a in vals}
    for i in range(min(max_scan_rows, len(raw_df))):
        vals = [normalize_header(v) for v in raw_df.iloc[i].tolist()]
        score = sum(1 for v in vals if v in aliases)
        # Bonus for known anchor headers.
        score += 3 * int("final alloc" in vals or "final alloc." in vals)
        score += 2 * int("flag" in vals)
        score += 2 * int("item" in vals)
        if score > best_score:
            best_score = score
            best_i = i
    return best_i


def build_column_map(df) -> Dict[str, Optional[str]]:
    """Map canonical fields to actual dataframe columns.

    Header aliases are preferred. If a header is missing or duplicated, fallback to known one-based Excel positions.
    """
    normalized_to_cols: Dict[str, List[str]] = {}
    for c in df.columns:
        normalized_to_cols.setdefault(normalize_header(c), []).append(c)

    result: Dict[str, Optional[str]] = {}
    for field, aliases in HEADER_ALIASES.items():
        found = None
        for alias in aliases:
            cols = normalized_to_cols.get(normalize_header(alias), [])
            if cols:
                found = cols[0]
                break
        if found is None:
            pos = DEFAULT_EXCEL_COLUMNS.get(field)
            if pos is not None and 1 <= pos <= len(df.columns):
                found = df.columns[pos - 1]
        result[field] = found
    return result


@dataclass
class ColumnDiagnostics:
    rows: int
    columns: int
    header_map: Dict[str, Optional[str]]

    def as_rows(self):
        return [{"field": k, "detected_column": v} for k, v in self.header_map.items()]
