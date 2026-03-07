from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any


import numpy as np
import pandas as pd
from dotenv import load_dotenv
import wrds
import nasdaqdatalink 

def configure_api_from_env(env_candidates: list[Path]) -> None:
    """Load API credentials from .env and configure Nasdaq Data Link.

    This function only loads and validates credentials and does not fetch data.
    """
    env_path = next((p for p in env_candidates if p.exists()), None)
    if env_path is None:
        checked = ", ".join(str(p) for p in env_candidates)
        raise FileNotFoundError(f"Could not find .env file. Checked: {checked}")

    load_dotenv(env_path)
    api_key = os.getenv("NASDAQ_API_KEY")
    if not api_key:
        raise ValueError(f"NASDAQ_API_KEY missing in {env_path}")

    nasdaqdatalink.ApiConfig.api_key = api_key


def normalize_ticker_for_prices(ticker: str) -> str:
    if pd.isna(ticker):
        return ticker
    return str(ticker).replace(".", "_")


def _to_api_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    api_filters: dict[str, Any] = {}
    if not filters:
        return api_filters

    for col, cond in filters.items():
        if isinstance(cond, dict) and "between" in cond:
            lo, hi = cond["between"]
            api_filters[col] = {"gte": str(lo), "lte": str(hi)}
        elif isinstance(cond, dict) and "in" in cond:
            api_filters[col] = list(cond["in"])
        else:
            api_filters[col] = cond

    return api_filters


def _apply_filters_in_memory(df: pd.DataFrame, filters: dict[str, Any] | None) -> pd.DataFrame:
    if not filters:
        return df

    out = df
    for col, cond in filters.items():
        if col not in out.columns:
            continue

        if isinstance(cond, dict) and "between" in cond:
            lo, hi = cond["between"]
            vals = out[col].astype(str)
            out = out[(vals >= str(lo)) & (vals <= str(hi))]
        elif isinstance(cond, dict) and "in" in cond:
            out = out[out[col].isin(list(cond["in"]))]
        else:
            out = out[out[col] == cond]

    return out


def fetch_zacks_table(
    table_code: str,
    columns: list[str],
    filters: dict[str, Any] | None,
    paginate: bool = True,
) -> pd.DataFrame:
    """Fetch a Zacks table with API filters and known filter exceptions."""
    unsupported_filter_cols_by_table = {
        "ZACKS/MT": {"sp500_member_flag"},
    }

    unsupported = unsupported_filter_cols_by_table.get(table_code, set())
    api_side_filters: dict[str, Any] = {}
    post_filters: dict[str, Any] = {}

    if filters:
        for key, val in filters.items():
            if key in unsupported:
                post_filters[key] = val
            else:
                api_side_filters[key] = val

    api_filters = _to_api_filters(api_side_filters)

    df = nasdaqdatalink.get_table(
        table_code,
        qopts={"columns": columns},
        paginate=paginate,
        **api_filters,
    )

    if df.empty:
        return pd.DataFrame(columns=columns)

    if post_filters:
        df = _apply_filters_in_memory(df, post_filters)

    existing = [c for c in columns if c in df.columns]
    if existing:
        df = df[existing].copy()

    return df.reset_index(drop=True)


def load_prices_csv_required(
    csv_path: Path,
    tickers: list[str],
    start: str,
    end: str,
    usecols: list[str],
) -> pd.DataFrame:
    """Load course-provided PRICES.csv and filter in chunks.

    The file is required and must be placed in the project data directory.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing required file: {csv_path}. Place course-provided PRICES.csv at data/PRICES.csv"
        )

    ticker_set = set(str(t) for t in tickers)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    parts: list[pd.DataFrame] = []
    reader = pd.read_csv(csv_path, usecols=usecols, chunksize=800_000, low_memory=False)

    for chunk in reader:
        sub = chunk[chunk["ticker"].astype(str).isin(ticker_set)].copy()
        if sub.empty:
            continue

        sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
        sub = sub.dropna(subset=["date"])
        if sub.empty:
            continue

        sub = sub[(sub["date"] >= start_ts) & (sub["date"] <= end_ts)]
        if not sub.empty:
            parts.append(sub)

    if not parts:
        return pd.DataFrame(columns=usecols)

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    return out


def build_static_top10_universe(
    mt_df: pd.DataFrame,
    mktv_df: pd.DataFrame,
    rank_date: str = "2012-12-31",
) -> pd.DataFrame:
    rank_ts = pd.Timestamp(rank_date)

    mt = mt_df.copy()
    mt["ticker"] = mt["ticker"].astype(str)
    mt = mt[mt["sp500_member_flag"] == "Y"][["ticker"]].drop_duplicates()

    mktv = mktv_df.copy()
    mktv["ticker"] = mktv["ticker"].astype(str)
    mktv["per_end_date"] = pd.to_datetime(mktv["per_end_date"], errors="coerce")
    mktv["mkt_val"] = pd.to_numeric(mktv["mkt_val"], errors="coerce")

    if "per_type" in mktv.columns:
        mktv = mktv[mktv["per_type"] == "Q"].copy()

    mktv = mktv[mktv["per_end_date"] == rank_ts].copy()
    mktv = mktv.merge(mt, on="ticker", how="inner")
    mktv = mktv.dropna(subset=["mkt_val"])
    mktv = mktv.sort_values(["mkt_val", "ticker"], ascending=[False, True])
    mktv = mktv.drop_duplicates(subset=["ticker"], keep="first")

    top10 = mktv.head(10).copy()
    top10["ticker_price"] = top10["ticker"].map(normalize_ticker_for_prices)
    top10["rank_date"] = rank_ts

    return top10[["ticker", "ticker_price", "mkt_val", "rank_date"]].reset_index(drop=True)


def prepare_fundamentals_with_availability(
    fr_df: pd.DataFrame,
    fc_df: pd.DataFrame,
    lag_days: int = 45,
) -> pd.DataFrame:
    fr = fr_df.copy()
    fc = fc_df.copy()

    fr["ticker"] = fr["ticker"].astype(str)
    fc["ticker"] = fc["ticker"].astype(str)

    fr["per_end_date"] = pd.to_datetime(fr["per_end_date"], errors="coerce")
    fc["per_end_date"] = pd.to_datetime(fc["per_end_date"], errors="coerce")

    if "per_type" in fr.columns:
        fr = fr[fr["per_type"] == "Q"].copy()
    if "per_type" in fc.columns:
        fc = fc[fc["per_type"] == "Q"].copy()

    keep_fr = [
        col
        for col in [
            "ticker",
            "per_end_date",
            "per_type",
            "tot_debt_tot_equity",
            "ret_equity",
            "profit_margin",
            "book_val_per_share",
        ]
        if col in fr.columns
    ]
    keep_fc = [
        col for col in ["ticker", "per_end_date", "per_type", "diluted_net_eps"] if col in fc.columns
    ]

    fr = fr[keep_fr].copy()
    fc = fc[keep_fc].copy()

    join_keys = ["ticker", "per_end_date"]
    if "per_type" in fr.columns and "per_type" in fc.columns:
        join_keys.append("per_type")

    fundamentals = fr.merge(fc, on=join_keys, how="outer")
    fundamentals["feature_available_date"] = fundamentals["per_end_date"] + pd.Timedelta(days=lag_days)
    fundamentals["ticker_price"] = fundamentals["ticker"].map(normalize_ticker_for_prices)

    numeric_cols = [
        "tot_debt_tot_equity",
        "ret_equity",
        "profit_margin",
        "book_val_per_share",
        "diluted_net_eps",
    ]
    for col in numeric_cols:
        if col in fundamentals.columns:
            fundamentals[col] = pd.to_numeric(fundamentals[col], errors="coerce")

    return fundamentals.sort_values(["ticker_price", "feature_available_date", "per_end_date"]).reset_index(drop=True)


def asof_join_point_in_time(
    prices_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    on_date_col: str,
    by_ticker_col: str,
) -> pd.DataFrame:
    if by_ticker_col not in prices_df.columns:
        raise KeyError(f"Missing ticker column in prices_df: {by_ticker_col}")
    if on_date_col not in prices_df.columns:
        raise KeyError(f"Missing date column in prices_df: {on_date_col}")
    if by_ticker_col not in fundamentals_df.columns:
        raise KeyError(f"Missing ticker column in fundamentals_df: {by_ticker_col}")
    if "feature_available_date" not in fundamentals_df.columns:
        raise KeyError("Missing feature_available_date in fundamentals_df")

    merged_parts: list[pd.DataFrame] = []

    fundamental_cols = [
        col
        for col in [
            "per_end_date",
            "feature_available_date",
            "tot_debt_tot_equity",
            "ret_equity",
            "profit_margin",
            "book_val_per_share",
            "diluted_net_eps",
        ]
        if col in fundamentals_df.columns
    ]

    ffill_cols = [
        col
        for col in [
            "tot_debt_tot_equity",
            "ret_equity",
            "profit_margin",
            "book_val_per_share",
            "diluted_net_eps",
        ]
        if col in fundamental_cols
    ]

    for ticker, px in prices_df.groupby(by_ticker_col, sort=False):
        px = px.sort_values(on_date_col).copy()
        f_ticker = fundamentals_df[fundamentals_df[by_ticker_col] == ticker].copy()
        f_ticker = f_ticker.sort_values("feature_available_date")

        if f_ticker.empty:
            for col in fundamental_cols:
                if col not in px.columns:
                    px[col] = np.nan
            merged_parts.append(px)
            continue

        merged = pd.merge_asof(
            px,
            f_ticker[fundamental_cols],
            left_on=on_date_col,
            right_on="feature_available_date",
            direction="backward",
            allow_exact_matches=True,
        )

        if ffill_cols:
            merged[ffill_cols] = merged[ffill_cols].ffill()

        merged_parts.append(merged)

    out = pd.concat(merged_parts, ignore_index=True)
    out = out.sort_values([by_ticker_col, on_date_col]).reset_index(drop=True)
    return out


def validate_point_in_time_panel(panel_df: pd.DataFrame) -> None:
    required = {"date", "feature_available_date"}
    missing = required - set(panel_df.columns)
    if missing:
        raise KeyError(f"Missing required columns for PIT validation: {sorted(missing)}")

    ok = panel_df["feature_available_date"].isna() | (panel_df["feature_available_date"] <= panel_df["date"])
    if not bool(ok.all()):
        raise AssertionError("Found lookahead leakage: feature_available_date > date")



def load_wrds_credentials(env_candidates: list[Path] | None = None) -> tuple[str, str]:
    """Load WRDS_USERNAME and WRDS_PASSWORD from the first .env file found."""
    if env_candidates is None:
        env_candidates = [
            Path(__file__).resolve().parents[1] / ".env",  # team_t/.env
            Path.cwd() / ".env",
        ]

    env_path = next((p for p in env_candidates if p.exists()), None)
    if env_path:
        load_dotenv(env_path, override=True)

    username = os.getenv("WRDS_USERNAME")
    password = os.getenv("WRDS_PASSWORD")

    if not username:
        raise ValueError(
            "WRDS_USERNAME missing. Add it to your .env file:\n"
            "  WRDS_USERNAME=your_wrds_username\n"
            "  WRDS_PASSWORD=your_wrds_password"
        )
    return username, password


def connect_wrds(env_candidates: list[Path] | None = None) -> wrds.Connection:
    """Return an open WRDS connection using credentials from .env."""
    username, password = load_wrds_credentials(env_candidates)
    return wrds.Connection(wrds_username=username, wrds_password=password or "")


def load_universe_tickers(
    universe_path: Path,
    fallback_universe_path: Path | None = None,
) -> tuple[list[str], Path]:
    """Load a unique, uppercase ticker list from the primary universe or fallback."""
    candidates = [universe_path]
    if fallback_universe_path is not None:
        candidates.append(fallback_universe_path)

    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        checked = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"No universe file found. Checked: {checked}")

    universe_df = pd.read_csv(source)
    if "ticker" not in universe_df.columns:
        raise KeyError(f"Universe file missing required 'ticker' column: {source}")

    tickers = (
        universe_df["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"": np.nan})
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    if not tickers:
        raise ValueError(f"No valid tickers found in universe file: {source}")

    return tickers, source


def _sql_literal_list(values: list[str]) -> str:
    escaped = [value.replace("'", "''") for value in values]
    return ", ".join(f"'{value}'" for value in escaped)


def _chunked(values: list[int], chunk_size: int = 250) -> list[list[int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [values[idx : idx + chunk_size] for idx in range(0, len(values), chunk_size)]


def _normalize_ticker_key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def _build_universe_key_map(tickers: list[str]) -> dict[str, str]:
    key_map: dict[str, str] = {}
    for ticker in tickers:
        base = str(ticker).upper().strip()
        variants = {
            base,
            base.replace(".", ""),
            base.replace(".", "_"),
            base.replace("/", ""),
            base.replace("-", ""),
            base.replace(" ", ""),
        }
        for variant in variants:
            key = _normalize_ticker_key(variant)
            if key and key not in key_map:
                key_map[key] = base
    return key_map


def _list_schema_tables(db: wrds.Connection, schema: str) -> set[str]:
    table_query = f"""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = '{schema}'
    """
    table_df = db.raw_sql(table_query)
    if table_df.empty:
        return set()
    return set(table_df["table_name"].astype(str).str.lower().tolist())


def _get_table_columns(db: wrds.Connection, schema: str, table: str) -> list[str]:
    col_query = f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = '{schema}'
          AND table_name = '{table}'
        ORDER BY ordinal_position
    """
    col_df = db.raw_sql(col_query)
    if col_df.empty:
        return []
    return col_df["column_name"].astype(str).str.lower().tolist()


def _resolve_optionm_schema(db: wrds.Connection) -> str:
    for schema in ["optionm", "optionm_all"]:
        tables = _list_schema_tables(db, schema)
        if any(table.startswith("opprcd") for table in tables):
            return schema
    raise ValueError("No OptionMetrics schema found (checked: optionm, optionm_all).")


def _resolve_opprcd_tables(db: wrds.Connection, schema: str) -> list[str]:
    tables = sorted(table for table in _list_schema_tables(db, schema) if table.startswith("opprcd"))
    if not tables:
        raise ValueError(f"No OPPRCD tables found in schema '{schema}'.")
    if "opprcd" in tables:
        return ["opprcd"]
    yearly = [table for table in tables if re.fullmatch(r"opprcd\d{4}", table)]
    if yearly:
        return sorted(yearly)
    return sorted(tables)


def _resolve_mapping_table(db: wrds.Connection, schema: str) -> str:
    tables = _list_schema_tables(db, schema)
    for candidate in ["secnmd", "securd"]:
        if candidate in tables:
            return candidate
    raise ValueError(f"No mapping table found in schema '{schema}' (expected secnmd or securd).")


def _resolve_implied_vol_col(opprcd_cols: set[str]) -> str | None:
    for candidate in ["impl_volatility", "impl_vol", "iv"]:
        if candidate in opprcd_cols:
            return candidate
    return None


def _resolve_latest_end_date(
    db: wrds.Connection,
    schema: str,
    opprcd_tables: list[str],
    start_date: str,
) -> str:
    max_dates: list[pd.Timestamp] = []
    for table in opprcd_tables:
        max_query = f"""
            SELECT MAX(date) AS max_date
            FROM {schema}.{table}
            WHERE date >= '{start_date}'
        """
        max_df = db.raw_sql(max_query)
        if max_df.empty:
            continue
        max_val = pd.to_datetime(max_df.loc[0, "max_date"], errors="coerce")
        if pd.notna(max_val):
            max_dates.append(max_val)

    if not max_dates:
        raise ValueError("Could not resolve latest available OPPRCD date from WRDS.")

    latest = max(max_dates).date().isoformat()
    return latest


def _fetch_secid_mapping(
    db: wrds.Connection,
    schema: str,
    mapping_table: str,
    universe_tickers: list[str],
) -> pd.DataFrame:
    mapping_cols = _get_table_columns(db, schema, mapping_table)
    if "secid" not in mapping_cols:
        raise KeyError(f"Mapping table {schema}.{mapping_table} missing required 'secid' column.")

    ticker_col = next((col for col in ["ticker", "tic", "symbol"] if col in mapping_cols), None)
    if ticker_col is None:
        raise KeyError(
            f"Could not find ticker column in {schema}.{mapping_table}. "
            f"Checked candidates: ticker, tic, symbol."
        )

    date_col = next(
        (
            col
            for col in ["effect_date", "namedt", "sdate", "start_date", "date"]
            if col in mapping_cols
        ),
        None,
    )

    universe_key_map = _build_universe_key_map(universe_tickers)
    if not universe_key_map:
        raise ValueError("Universe ticker list is empty after normalization.")

    key_sql = _sql_literal_list(sorted(universe_key_map))
    key_expr = f"UPPER(REGEXP_REPLACE(COALESCE({ticker_col}, ''), '[^A-Za-z0-9]', '', 'g'))"
    select_date = f", {date_col} AS map_date" if date_col else ""

    mapping_query = f"""
        SELECT
            secid,
            {ticker_col} AS raw_ticker
            {select_date}
        FROM {schema}.{mapping_table}
        WHERE {ticker_col} IS NOT NULL
          AND {key_expr} IN ({key_sql})
    """
    mapping_df = db.raw_sql(mapping_query)
    if mapping_df.empty:
        return pd.DataFrame(columns=["secid", "ticker"])

    mapping_df["ticker_key"] = mapping_df["raw_ticker"].map(_normalize_ticker_key)
    mapping_df["ticker"] = mapping_df["ticker_key"].map(universe_key_map)
    mapping_df = mapping_df.dropna(subset=["ticker"])
    mapping_df["secid"] = pd.to_numeric(mapping_df["secid"], errors="coerce")
    mapping_df = mapping_df.dropna(subset=["secid"])
    mapping_df["secid"] = mapping_df["secid"].astype(int)

    sort_cols = ["ticker", "secid"]
    if "map_date" in mapping_df.columns:
        mapping_df["map_date"] = pd.to_datetime(mapping_df["map_date"], errors="coerce")
        sort_cols.append("map_date")
    mapping_df = mapping_df.sort_values(sort_cols)
    mapping_df = mapping_df.drop_duplicates(["ticker", "secid"], keep="last")
    return mapping_df[["secid", "ticker"]].reset_index(drop=True)


def _fetch_opprcd_filtered(
    db: wrds.Connection,
    schema: str,
    table: str,
    secids: list[int],
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, list[str]]:
    cols = _get_table_columns(db, schema, table)
    col_set = set(cols)

    required_cols = ["date", "secid", "cp_flag", "strike_price", "exdate", "best_bid", "best_offer", "delta"]
    missing_required = [col for col in required_cols if col not in col_set]
    if missing_required:
        raise KeyError(f"Required columns missing from {schema}.{table}: {missing_required}")

    implied_vol_col = _resolve_implied_vol_col(col_set)
    optional_candidates = ["optionid", "gamma", "vega", "theta", "volume", "open_interest"]
    present_optional = [col for col in optional_candidates if col in col_set]
    missing_optional = [col for col in optional_candidates if col not in col_set]

    select_parts = [
        "o.date",
        "o.secid",
        "o.cp_flag",
        "o.strike_price",
        "o.exdate",
        "o.best_bid",
        "o.best_offer",
        "0.5 * (o.best_bid + o.best_offer) AS mid_price",
        "o.delta",
        "(o.exdate - o.date) AS dte",
    ]
    if implied_vol_col is not None:
        select_parts.append(f"o.{implied_vol_col} AS implied_vol")
    else:
        select_parts.append("NULL::double precision AS implied_vol")

    for col in present_optional:
        select_parts.append(f"o.{col}")

    where_parts = [
        f"o.date BETWEEN '{start_date}' AND '{end_date}'",
        "UPPER(TRIM(o.cp_flag)) = 'C'",
        "(o.exdate - o.date) BETWEEN 30 AND 60",
        "o.best_bid > 0",
        "o.best_offer > 0",
    ]
    if "open_interest" in col_set:
        where_parts.append("o.open_interest > 0")

    table_frames: list[pd.DataFrame] = []
    for secid_chunk in _chunked(secids, chunk_size=250):
        secid_sql = ", ".join(str(secid) for secid in secid_chunk)
        sql = f"""
            SELECT
                {", ".join(select_parts)}
            FROM {schema}.{table} o
            WHERE o.secid IN ({secid_sql})
              AND {" AND ".join(where_parts)}
        """
        chunk_df = db.raw_sql(sql)
        if not chunk_df.empty:
            table_frames.append(chunk_df)

    if not table_frames:
        return pd.DataFrame(columns=[*required_cols, "mid_price", "implied_vol", "dte", *present_optional]), missing_optional

    out = pd.concat(table_frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["exdate"] = pd.to_datetime(out["exdate"], errors="coerce")
    numeric_cols = [
        "strike_price",
        "best_bid",
        "best_offer",
        "mid_price",
        "implied_vol",
        "delta",
        "dte",
        "optionid",
        "gamma",
        "vega",
        "theta",
        "volume",
        "open_interest",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out, missing_optional


def _fetch_underlying_secprc(
    db: wrds.Connection,
    schema: str,
    secids: list[int],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    schema_candidates = [schema] + [candidate for candidate in ["optionm", "optionm_all"] if candidate != schema]
    selected_schema: str | None = None
    selected_table: str | None = None
    close_col: str | None = None

    for schema_candidate in schema_candidates:
        tables = _list_schema_tables(db, schema_candidate)
        table_candidate = None
        if "secprc" in tables:
            table_candidate = "secprc"
        elif "secprd" in tables:
            table_candidate = "secprd"

        if table_candidate is None:
            continue

        cols = set(_get_table_columns(db, schema_candidate, table_candidate))
        if {"secid", "date"}.issubset(cols):
            if "close" in cols:
                selected_schema = schema_candidate
                selected_table = table_candidate
                close_col = "close"
                break
            if "close_price" in cols:
                selected_schema = schema_candidate
                selected_table = table_candidate
                close_col = "close_price"
                break

    if selected_schema is None or selected_table is None or close_col is None:
        raise ValueError(
            "Underlying price table not found with required columns. "
            "Checked optionm/optionm_all for secprc and secprd."
        )

    frames: list[pd.DataFrame] = []
    for secid_chunk in _chunked(secids, chunk_size=250):
        secid_sql = ", ".join(str(secid) for secid in secid_chunk)
        sql = f"""
            SELECT
                secid,
                date,
                {close_col} AS underlying_price
            FROM {selected_schema}.{selected_table}
            WHERE secid IN ({secid_sql})
              AND date BETWEEN '{start_date}' AND '{end_date}'
        """
        chunk_df = db.raw_sql(sql)
        if not chunk_df.empty:
            frames.append(chunk_df)

    if not frames:
        return pd.DataFrame(columns=["secid", "date", "underlying_price"])

    out = pd.concat(frames, ignore_index=True)
    out["secid"] = pd.to_numeric(out["secid"], errors="coerce").astype("Int64")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["underlying_price"] = pd.to_numeric(out["underlying_price"], errors="coerce")
    out = out.dropna(subset=["secid", "date"]).copy()
    out["secid"] = out["secid"].astype(int)
    out = out.sort_values(["secid", "date"]).drop_duplicates(["secid", "date"], keep="last")
    return out.reset_index(drop=True)


def pull_optionmetrics_calls_atm_dataset(
    db: wrds.Connection,
    universe_path: Path,
    fallback_universe_path: Path,
    output_path: Path,
    start_date: str = "2006-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    """Pull OptionMetrics calls, filter to near-ATM contracts, and write parquet output."""
    universe_tickers, used_universe_path = load_universe_tickers(universe_path, fallback_universe_path)
    print(
        f"[universe] source={used_universe_path} tickers={len(universe_tickers)} "
        f"sample={universe_tickers[:5]}"
    )

    schema = _resolve_optionm_schema(db)
    opprcd_tables = _resolve_opprcd_tables(db, schema)
    mapping_table = _resolve_mapping_table(db, schema)
    print(f"[wrds] schema={schema} opprcd_tables={opprcd_tables} mapping_table={mapping_table}")

    if end_date is None:
        end_date = _resolve_latest_end_date(db, schema, opprcd_tables, start_date)
        print(f"[date range] start={start_date} end={end_date} (auto latest)")
    else:
        print(f"[date range] start={start_date} end={end_date}")

    secid_map = _fetch_secid_mapping(db, schema, mapping_table, universe_tickers)
    if secid_map.empty:
        raise ValueError("No secids mapped for universe tickers in OptionMetrics mapping tables.")

    mapped_tickers = secid_map["ticker"].nunique()
    print(f"[mapping] secids={len(secid_map):,} mapped_tickers={mapped_tickers:,}")
    secids = sorted(secid_map["secid"].astype(int).unique().tolist())

    option_frames: list[pd.DataFrame] = []
    missing_optional_union: set[str] = set()
    for table in opprcd_tables:
        table_df, missing_optional = _fetch_opprcd_filtered(
            db=db,
            schema=schema,
            table=table,
            secids=secids,
            start_date=start_date,
            end_date=end_date,
        )
        missing_optional_union.update(missing_optional)
        print(f"[opprcd] table={table} rows_after_sql_filters={len(table_df):,}")
        if not table_df.empty:
            option_frames.append(table_df)

    if not option_frames:
        raise ValueError("No OPPRCD rows returned after SQL filters.")

    options = pd.concat(option_frames, ignore_index=True)
    options = options.merge(secid_map, on="secid", how="left")
    options["ticker"] = options["ticker"].astype(str).str.upper()
    options = options.dropna(subset=["ticker", "date", "exdate"])
    print(f"[counts] rows_after_opprcd_merge={len(options):,}")

    # OptionMetrics OPPRCD strike_price is scaled by 1000.
    options["strike_price"] = pd.to_numeric(options["strike_price"], errors="coerce") / 1000.0
    options["best_bid"] = pd.to_numeric(options["best_bid"], errors="coerce")
    options["best_offer"] = pd.to_numeric(options["best_offer"], errors="coerce")
    options["mid_price"] = 0.5 * (options["best_bid"] + options["best_offer"])
    options["dte"] = (options["exdate"] - options["date"]).dt.days

    underlying_df = _fetch_underlying_secprc(
        db=db,
        schema=schema,
        secids=secids,
        start_date=start_date,
        end_date=end_date,
    )
    print(f"[secprc] rows={len(underlying_df):,}")

    before_underlying = len(options)
    options = options.merge(underlying_df, on=["secid", "date"], how="left")
    options["underlying_price"] = pd.to_numeric(options["underlying_price"], errors="coerce")
    options = options[options["underlying_price"] > 0].copy()
    print(
        f"[counts] before_underlying_filter={before_underlying:,} "
        f"after_underlying_filter={len(options):,}"
    )

    before_moneyness = len(options)
    options["moneyness"] = options["underlying_price"] / options["strike_price"]
    options = options[options["moneyness"].between(0.95, 1.05, inclusive="both")].copy()
    print(f"[counts] before_moneyness_filter={before_moneyness:,} after_moneyness_filter={len(options):,}")

    for col in ["optionid", "gamma", "vega", "theta", "volume", "open_interest"]:
        if col not in options.columns:
            options[col] = np.nan

    options["open_interest"] = pd.to_numeric(options["open_interest"], errors="coerce")
    options["optionid"] = pd.to_numeric(options["optionid"], errors="coerce")
    options["_atm_gap"] = (options["moneyness"] - 1.0).abs()

    before_single_contract = len(options)
    options = options.sort_values(
        ["ticker", "date", "_atm_gap", "open_interest", "exdate", "optionid"],
        ascending=[True, True, True, False, True, True],
        na_position="last",
    )
    options = options.drop_duplicates(["ticker", "date"], keep="first").copy()
    print(
        f"[counts] before_single_contract={before_single_contract:,} "
        f"after_single_contract={len(options):,}"
    )

    options = options.sort_values(["ticker", "date"]).reset_index(drop=True)
    options["option_mid_return"] = np.log(
        options["mid_price"] / options.groupby("ticker")["mid_price"].shift(1)
    )

    if "implied_vol" not in options.columns:
        options["implied_vol"] = np.nan

    for col in ["gamma", "vega", "theta", "volume", "open_interest", "optionid"]:
        if col not in options.columns:
            options[col] = np.nan

    final_cols = [
        "date",
        "ticker",
        "secid",
        "optionid",
        "cp_flag",
        "strike_price",
        "exdate",
        "dte",
        "best_bid",
        "best_offer",
        "mid_price",
        "implied_vol",
        "delta",
        "gamma",
        "vega",
        "theta",
        "volume",
        "open_interest",
        "underlying_price",
        "moneyness",
        "option_mid_return",
    ]
    for col in final_cols:
        if col not in options.columns:
            options[col] = np.nan

    final_df = options[final_cols].sort_values(["ticker", "date"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_path, index=False)

    missing_optional_sorted = sorted(missing_optional_union)
    print(
        f"[opprcd optional cols missing] {missing_optional_sorted if missing_optional_sorted else 'none'}"
    )
    print(f"[output] rows={len(final_df):,} path={output_path}")
    return final_df


# ---------------------------------------------------------------------------
# Compustat earnings announcement dates (rdq)
# ---------------------------------------------------------------------------

def fetch_earnings_announcement_dates(
    tickers: list,
    start_date: str = "2006-01-01",
    end_date: str = "2013-12-31",
    output_path=None,
    env_candidates=None,
) -> "pd.DataFrame":
    """Fetch actual earnings announcement dates (rdq) from WRDS Compustat.

    Uses comp.fundq (quarterly fundamentals) joined to comp.security for
    ticker→gvkey mapping. `rdq` is the actual earnings report date —
    far closer to the earnings call than the SEC 10-Q filing date.

    Returns DataFrame: ticker, gvkey, rdq, datadate, fqtr, fyearq.
    """
    if env_candidates is None:
        env_candidates = [Path(".env"), Path("../.env")]

    db = connect_wrds(env_candidates)

    # Normalise BRK_B → BRK/B for Compustat
    comp_tickers = [t.replace("_", "/") for t in tickers]
    ticker_list  = ", ".join(f"'{t}'" for t in comp_tickers)

    # Step 1: resolve gvkeys
    gvkey_df = db.raw_sql(f"""
        SELECT DISTINCT s.gvkey, s.tic AS ticker
        FROM   comp.security s
        WHERE  s.tic IN ({ticker_list})
          AND  s.excntry = 'USA'
    """)
    print(f"[compustat] gvkeys resolved: {len(gvkey_df)} rows")
    if gvkey_df.empty:
        db.close()
        raise ValueError(f"[compustat] No gvkeys found for: {comp_tickers}")

    gvkey_list = ", ".join(f"'{g}'" for g in gvkey_df["gvkey"].unique())

    # Step 2: pull rdq
    fundq_df = db.raw_sql(f"""
        SELECT f.gvkey, f.datadate, f.rdq, f.fqtr, f.fyearq
        FROM   comp.fundq f
        WHERE  f.gvkey  IN ({gvkey_list})
          AND  f.rdq    IS NOT NULL
          AND  f.rdq    >= '{start_date}'
          AND  f.rdq    <= '{end_date}'
          AND  f.indfmt  = 'INDL'
          AND  f.datafmt = 'STD'
          AND  f.popsrc  = 'D'
          AND  f.consol  = 'C'
        ORDER BY f.gvkey, f.rdq
    """)
    db.close()
    print(f"[compustat] fundq rows: {len(fundq_df)}")

    result = fundq_df.merge(gvkey_df, on="gvkey", how="left")
    result["rdq"]      = pd.to_datetime(result["rdq"])
    result["datadate"] = pd.to_datetime(result["datadate"])
    result["ticker"]   = result["ticker"].str.replace("/", "_", regex=False)
    result = result[result["ticker"].isin(tickers)].copy()
    result = result.sort_values(["ticker", "rdq"]).reset_index(drop=True)

    print(f"[compustat] rows={len(result)} | "
          f"{result['rdq'].min().date()} → {result['rdq'].max().date()}")
    print(f"[compustat] tickers: {sorted(result['ticker'].unique())}")

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(out, index=False)
        print(f"[compustat] saved → {out}")

    return result
