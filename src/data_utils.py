from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import nasdaqdatalink
import numpy as np
import pandas as pd
from dotenv import load_dotenv


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

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
import wrds


def load_wrds_credentials(env_candidates: list[Path] | None = None) -> tuple[str, str]:
    """Load WRDS_USERNAME and WRDS_PASSWORD from the first .env file found."""
    if env_candidates is None:
        env_candidates = [
            Path(__file__).resolve().parents[1] / ".env",  # team_t/.env
            Path.cwd() / ".env",
        ]

    env_path = next((p for p in env_candidates if p.exists()), None)
    if env_path:
        load_dotenv(env_path)

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
    return wrds.Connection(wrds_username=username, wrds_password=password)