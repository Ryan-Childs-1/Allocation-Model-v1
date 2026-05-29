from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from pyxlsb import open_workbook

from schema import detect_header_row, unique_columns


def save_upload(uploaded_file, suffix: str = "") -> Path:
    suffix = suffix or Path(uploaded_file.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def _read_xlsb_raw(path: str | Path, sheet_name: str = "3.3 Working Table", max_raw_rows: int | None = None) -> pd.DataFrame:
    """Fast .xlsb reader that preserves true Excel column positions.

    pyxlsb exposes cell.c as zero-based Excel column indexes. Earlier versions of
    this app converted rows through sparse dictionaries, which was slow on large
    allocation sheets. This implementation builds expandable row lists directly,
    so hosted Streamlit sessions can parse multiple workbooks much faster while
    still preserving hidden/blank column positions.
    """
    rows = []
    max_cols = 0
    with open_workbook(str(path)) as wb:
        sheet_names = wb.sheets
        if sheet_name not in sheet_names:
            candidates = [s for s in sheet_names if "working" in s.lower() or "table" in s.lower()]
            sheet_name = candidates[0] if candidates else sheet_names[0]
        with wb.get_sheet(sheet_name) as sh:
            consecutive_blank = 0
            for r_i, row in enumerate(sh.rows()):
                if max_raw_rows is not None and r_i >= max_raw_rows:
                    break
                values = []
                nonblank = False
                for cell in row:
                    col_pos = int(cell.c) + 1
                    if col_pos > len(values):
                        values.extend([None] * (col_pos - len(values)))
                    v = cell.v
                    values[col_pos - 1] = v
                    if v not in (None, ""):
                        nonblank = True
                    if col_pos > max_cols:
                        max_cols = col_pos
                rows.append(values)
                if nonblank:
                    consecutive_blank = 0
                else:
                    consecutive_blank += 1
                    # Sportsman's allocation exports often contain hundreds of thousands of blank formatted rows.
                    # Stop once we have already captured the header/data area and then hit a long blank tail.
                    if r_i > 60 and consecutive_blank >= 500:
                        rows = rows[:-consecutive_blank]
                        break
    if not rows:
        return pd.DataFrame()
    for r in rows:
        if len(r) < max_cols:
            r.extend([None] * (max_cols - len(r)))
    return pd.DataFrame(rows)


def _promote_header(raw: pd.DataFrame) -> pd.DataFrame:
    header_i = detect_header_row(raw)
    headers = unique_columns(raw.iloc[header_i].tolist())
    df = raw.iloc[header_i + 1:].copy().reset_index(drop=True)
    df.columns = headers[: len(df.columns)]
    # Remove fully blank rows and trailing fully blank columns, but preserve original row order.
    df = df.dropna(how="all")
    # Do NOT drop blank columns. Hidden/blank Excel columns are meaningful for fixed-position mapping.
    df = df.reset_index(drop=True)
    df["__row_order"] = range(len(df))
    df["__excel_row"] = df["__row_order"] + header_i + 2
    return df


def read_allocation_file(path: str | Path, sheet_name: str = "3.3 Working Table", max_rows: Optional[int] = None) -> pd.DataFrame:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".xlsb":
        # Read enough rows to detect the header plus requested data rows.
        limit = None if max_rows is None else int(max_rows) + 60
        raw = _read_xlsb_raw(path, sheet_name=sheet_name, max_raw_rows=limit)
        df = _promote_header(raw)
    elif ext in {".xlsx", ".xlsm"}:
        raw = pd.read_excel(path, sheet_name=sheet_name, header=None, engine="openpyxl")
        df = _promote_header(raw)
    elif ext == ".csv":
        df = pd.read_csv(path)
        df.columns = unique_columns(df.columns)
        if "__row_order" not in df.columns:
            df["__row_order"] = range(len(df))
        if "__excel_row" not in df.columns:
            df["__excel_row"] = df["__row_order"] + 2
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    if max_rows is not None and max_rows > 0:
        df = df.head(int(max_rows)).copy()
    return df


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def dataframe_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()
