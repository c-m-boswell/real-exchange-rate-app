"""Real Exchange Rate Explorer.

A Streamlit app that compares the nominal and real exchange rates between any
two IMF economies (or the Euro area) using monthly IMF data from the SDMX 3.0 API.

Economics (Country 1 = c1, Country 2 = c2):
    E1(t)  = c1 currency per USD          (IMF key {ISO3}.XDC_USD.PA_RT.M)
    E2(t)  = c2 currency per USD
    S(t)   = E2(t) / E1(t)                = c2 currency per 1 unit of c1 currency
    RER(t) = S(t) * P1(t) / P2(t)         (P = all-items CPI index)
    RER*   = mean RER over valid months in the selected sample
    Real Index(t) = 100 * RER(t) / RER*   (averages exactly 100 over valid months)

Data endpoints were verified against the live API (October 2026):
    {BASE}/data/dataflow/IMF.STA/CPI/+/{KEY}   SDMX-CSV 2.0, key COUNTRY.INDEX_TYPE.COICOP_1999.TYPE_OF_TRANSFORMATION.FREQUENCY
    {BASE}/data/dataflow/IMF.STA/ER/+/{KEY}    SDMX-CSV 2.0, key COUNTRY.INDICATOR.TYPE_OF_TRANSFORMATION.FREQUENCY
    {BASE}/structure/codelist/IMF/CL_COUNTRY   SDMX-JSON 2.0 country names
"""

from __future__ import annotations

import io
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# IMF API configuration (confirmed against the live API)
# ---------------------------------------------------------------------------

BASE_URL = "https://api.imf.org/external/sdmx/3.0"
AGENCY = "IMF.STA"
FLOW_VERSION = "+"  # SDMX 3.0 shorthand for "latest stable version"
CPI_FLOW = "CPI"
ER_FLOW = "ER"
# All-items monthly price index. The Euro area (G163) publishes only HICP, not
# "CPI", so we request both and prefer CPI wherever an economy has it.
CPI_KEY = "*.CPI+HICP._T.IX.M"
ER_KEY = "*.XDC_USD.PA_RT.M"  # period-average, local currency per USD, monthly
START_PERIOD = "1976-01"
COUNTRY_CODELIST_URL = f"{BASE_URL}/structure/codelist/IMF/CL_COUNTRY"

CSV_ACCEPT = "application/vnd.sdmx.data+csv;version=2.0.0"
JSON_STRUCTURE_ACCEPT = "application/vnd.sdmx.structure+json;version=2.0.0"
USER_AGENT = "RealExchangeRateExplorer/1.0 (Streamlit app; +https://streamlit.io)"
TIMEOUT_S = 30
MAX_ATTEMPTS = 4
RETRY_STATUS = {429, 500, 502, 503, 504}
PROBE_WORKERS = 8
DAY_S = 24 * 3600
LOADING_MSG = "Loading IMF data… this happens once per day"

US = "USA"
EURO_AREA = "G163"
NAME_OVERRIDES = {EURO_AREA: "Euro area"}

# ISO 4217 codes for axis labels. IMF ER metadata carries no currency codes, so
# this is a small hand-curated map; unlisted economies fall back to
# "<Country> currency". Euro-area members map to their *legacy* currencies,
# because the IMF's national USD rates for them stop at euro adoption.
CURRENCY_CODES = {
    "USA": "USD", "G163": "EUR", "JPN": "JPY", "GBR": "GBP", "CHE": "CHF",
    "CAN": "CAD", "AUS": "AUD", "CHN": "CNY", "HKG": "HKD", "KOR": "KRW",
    "IND": "INR", "IDN": "IDR", "MYS": "MYR", "PHL": "PHP", "SGP": "SGD",
    "THA": "THB", "VNM": "VND", "PAK": "PKR", "BGD": "BDT", "LKA": "LKR",
    "MEX": "MXN", "BRA": "BRL", "ARG": "ARS", "CHL": "CLP", "COL": "COP",
    "PER": "PEN", "URY": "UYU", "ZAF": "ZAR", "NGA": "NGN", "KEN": "KES",
    "EGY": "EGP", "MAR": "MAD", "ISR": "ILS", "SAU": "SAR", "ARE": "AED",
    "TUR": "TRY", "RUS": "RUB", "UKR": "UAH", "NOR": "NOK", "SWE": "SEK",
    "DNK": "DKK", "ISL": "ISK", "POL": "PLN", "CZE": "CZK", "HUN": "HUF",
    "ROU": "RON",
}
EURO_LEGACY_CURRENCIES = {
    "AUT": "ATS", "BEL": "BEF", "DEU": "DEM", "ESP": "ESP", "FIN": "FIM",
    "FRA": "FRF", "IRL": "IEP", "ITA": "ITL", "LUX": "LUF", "NLD": "NLG",
    "PRT": "PTE", "GRC": "GRD", "SVN": "SIT", "CYP": "CYP", "MLT": "MTL",
    "SVK": "SKK", "EST": "EEK", "LVA": "LVL", "LTU": "LTL", "HRV": "HRK",
    "BGR": "BGN",
}

# Chart colors per Streamlit theme. Two accents only: blue for the Real Index,
# orange for nominal S(t); everything else is neutral. Keep in sync with
# .streamlit/config.toml.
PALETTES = {
    "light": {"real": "#22427F", "nominal": "#C24A23", "ink": "#1C1C1C", "muted": "#6F6D68",
              "hairline": "#E7E4DE", "surface": "#FBFAF7"},
    "dark": {"real": "#8FA8DD", "nominal": "#EE7A55", "ink": "#E9E7E2", "muted": "#9C9A94",
             "hairline": "#2E3238", "surface": "#14161A"},
}
FONT_STACK = "Jost, 'Helvetica Neue', Arial, sans-serif"

TIDY_COLUMNS = ["iso3", "indicator", "month", "value"]
EXPORT_COLUMNS = [
    "month", "country1_iso3", "country2_iso3", "E1_lcu_per_usd", "E2_lcu_per_usd",
    "S_c2_per_c1", "P1_cpi", "P2_cpi", "RER", "RER_star", "real_index", "valid",
]

View = Literal["Real", "Nominal", "Both"]


class IMFDataError(RuntimeError):
    """The IMF API could not be reached or returned an unusable response."""


# ---------------------------------------------------------------------------
# HTTP + parsing (pure functions, no Streamlit)
# ---------------------------------------------------------------------------


def http_get(url: str, accept: str, params: dict[str, str] | None = None) -> requests.Response | None:
    """GET with timeout and exponential backoff.

    Returns None when the API says "no data" (HTTP 204/404 or an empty body).
    Raises IMFDataError when the service stays unavailable.
    """
    headers = {"User-Agent": USER_AGENT, "Accept": accept}
    last_error = "unknown error"
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=TIMEOUT_S)
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = type(exc).__name__
        else:
            if resp.status_code in (204, 404):
                return None
            if resp.ok:
                return resp if resp.content.strip() else None
            if resp.status_code not in RETRY_STATUS:
                raise IMFDataError(f"IMF API returned HTTP {resp.status_code}")
            last_error = f"HTTP {resp.status_code}"
        if attempt < MAX_ATTEMPTS - 1:
            time.sleep(min(2**attempt, 8))
    raise IMFDataError(f"IMF API unavailable after {MAX_ATTEMPTS} attempts ({last_error})")


def parse_months(periods: pd.Series) -> pd.Series:
    """Turn IMF period strings ('1976-M01', '1976-01', '1976M1', '1976-01-31') into
    month-start Timestamps. Anything else (e.g. quarters) becomes NaT."""
    parts = periods.astype(str).str.strip().str.extract(r"^(\d{4})\s*-?\s*M?(\d{1,2})(?!\d)")
    stamps = parts[0] + "-" + parts[1].str.zfill(2)
    return pd.to_datetime(stamps, format="%Y-%m", errors="coerce")


def empty_tidy() -> pd.DataFrame:
    return pd.DataFrame({
        "iso3": pd.Series(dtype=str),
        "indicator": pd.Series(dtype=str),
        "month": pd.Series(dtype="datetime64[ns]"),
        "value": pd.Series(dtype=float),
    })


def parse_sdmx_csv(text: str) -> pd.DataFrame:
    """Parse an SDMX-CSV 2.0 data message into tidy [iso3, indicator, month, value].

    `indicator` is the second series-key dimension (INDEX_TYPE for CPI, INDICATOR
    for ER). Blank, non-numeric and non-positive observations are dropped.
    """
    raw = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    if raw.empty or not {"COUNTRY", "TIME_PERIOD", "OBS_VALUE"} <= set(raw.columns):
        return empty_tidy()
    cols = list(raw.columns)
    key_dims = cols[cols.index("COUNTRY"):cols.index("TIME_PERIOD")]
    tidy = pd.DataFrame({
        "iso3": raw["COUNTRY"].str.strip().str.upper(),
        "indicator": raw[key_dims[1]].str.strip() if len(key_dims) > 1 else "",
        "month": parse_months(raw["TIME_PERIOD"]),
        "value": pd.to_numeric(raw["OBS_VALUE"].str.strip(), errors="coerce"),
    })
    keep = tidy["month"].notna() & (tidy["value"] > 0) & (tidy["month"] >= pd.Timestamp(START_PERIOD))
    tidy = tidy.loc[keep].drop_duplicates(["iso3", "indicator", "month"], keep="last")
    return tidy.sort_values(["iso3", "indicator", "month"]).reset_index(drop=True)


def fetch_series(dataflow: str, key: str) -> pd.DataFrame:
    """Fetch one dataflow/key (wildcards allowed) from 1976-01 onward as tidy data."""
    url = f"{BASE_URL}/data/dataflow/{AGENCY}/{dataflow}/{FLOW_VERSION}/{key}"
    resp = http_get(url, CSV_ACCEPT, params={"c[TIME_PERIOD]": f"ge:{START_PERIOD}"})
    if resp is None:
        return empty_tidy()
    resp.encoding = "utf-8"
    return parse_sdmx_csv(resp.text)


def probe_countries(dataflow: str, key: str, codes: list[str]) -> pd.DataFrame:
    """Fallback for a failed wildcard request: query each country in parallel."""
    suffix = key.split(".", 1)[1]

    def probe(iso3: str) -> pd.DataFrame | None:
        try:
            return fetch_series(dataflow, f"{iso3}.{suffix}")
        except IMFDataError:
            return None

    with ThreadPoolExecutor(max_workers=PROBE_WORKERS) as pool:
        results = list(pool.map(probe, codes))
    if all(r is None for r in results):
        raise IMFDataError(f"could not download the {dataflow} dataset")
    frames = [r for r in results if r is not None and not r.empty]
    return pd.concat(frames, ignore_index=True) if frames else empty_tidy()


def parse_codelist(payload: dict) -> dict[str, str]:
    """Extract {code: English name} from an SDMX-JSON 2.0 structure message."""
    names: dict[str, str] = {}
    for codelist in payload.get("data", {}).get("codelists", []):
        for code in codelist.get("codes", []):
            name = code.get("name") or code.get("names", {}).get("en")
            if code.get("id") and name:
                names[code["id"]] = name
    names.update(NAME_OVERRIDES)
    return names


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------


@st.cache_data(ttl=DAY_S, show_spinner=LOADING_MSG)
def load_country_names() -> dict[str, str]:
    """IMF COUNTRY codelist as {ISO3: name}."""
    resp = http_get(COUNTRY_CODELIST_URL, JSON_STRUCTURE_ACCEPT)
    if resp is None:
        raise IMFDataError("IMF country codelist is empty")
    try:
        return parse_codelist(resp.json())
    except ValueError as exc:
        raise IMFDataError("IMF country codelist was not valid JSON") from exc


@st.cache_data(ttl=DAY_S, show_spinner=False)
def fetch_dataflow_bulk(dataflow: str, key: str) -> pd.DataFrame:
    """One wildcard request for every country; per-country probing if it fails."""
    try:
        data = fetch_series(dataflow, key)
        if not data.empty:
            return data
    except IMFDataError:
        if not key.startswith("*."):
            raise
    return probe_countries(dataflow, key, sorted(load_country_names()))


def select_price_index(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep one all-items index per economy: CPI where it exists, else HICP
    (needed for the Euro area, G163). Returns [iso3, measure, month, value]."""
    has_cpi = set(raw.loc[raw["indicator"] == "CPI", "iso3"])
    keep = (raw["indicator"] == "CPI") | ((raw["indicator"] == "HICP") & ~raw["iso3"].isin(has_cpi))
    return raw.loc[keep].rename(columns={"indicator": "measure"}).reset_index(drop=True)


@st.cache_data(ttl=DAY_S, show_spinner=LOADING_MSG)
def load_prices() -> pd.DataFrame:
    """Monthly all-items price index for every economy."""
    return select_price_index(fetch_dataflow_bulk(CPI_FLOW, CPI_KEY))


@st.cache_data(ttl=DAY_S, show_spinner=LOADING_MSG)
def load_rates() -> pd.DataFrame:
    """Monthly period-average local-currency-per-USD rates for every economy."""
    return fetch_dataflow_bulk(ER_FLOW, ER_KEY)


# ---------------------------------------------------------------------------
# Calculations (pure functions)
# ---------------------------------------------------------------------------


def available_countries(prices: pd.DataFrame, rates: pd.DataFrame) -> list[str]:
    """ISO3 codes with both a monthly price index and a monthly USD rate.
    The US always 'has' a USD rate (1 USD per USD)."""
    with_prices = set(prices["iso3"])
    with_rates = set(rates["iso3"]) | {US}
    return sorted(with_prices & with_rates)


def get_series(df: pd.DataFrame, iso3: str) -> pd.Series:
    """Monthly series for one economy, indexed by month-start Timestamp."""
    rows = df.loc[df["iso3"] == iso3, ["month", "value"]]
    s = rows.set_index("month")["value"].astype(float).sort_index()
    return s[~s.index.duplicated(keep="last")]


def get_usd_rate(rates: pd.DataFrame, prices: pd.DataFrame, iso3: str) -> pd.Series:
    """E(t) = local currency per USD. For the US, if the IMF has no USD-per-USD
    series, E(t) = 1 in every month where US CPI exists."""
    e = get_series(rates, iso3)
    if e.empty and iso3 == US:
        p = get_series(prices, US)
        e = pd.Series(1.0, index=p.index)
    return e


def build_pair_frame(prices: pd.DataFrame, rates: pd.DataFrame, c1: str, c2: str) -> pd.DataFrame:
    """Align E1, E2, P1, P2 on a full monthly index and compute S, RER and `valid`."""
    parts = {
        "E1": get_usd_rate(rates, prices, c1),
        "E2": get_usd_rate(rates, prices, c2),
        "P1": get_series(prices, c1),
        "P2": get_series(prices, c2),
    }
    present = [s for s in parts.values() if not s.empty]
    if present:
        idx = pd.date_range(min(s.index.min() for s in present), max(s.index.max() for s in present),
                            freq="MS", name="month")
    else:
        idx = pd.DatetimeIndex([], name="month")
    frame = pd.DataFrame({k: v.reindex(idx) for k, v in parts.items()}, index=idx)

    # S = E2 / E1 = (c2 per USD) / (c1 per USD) = c2 currency per 1 unit of c1 currency.
    frame["S"] = frame["E2"] / frame["E1"]
    # A month is valid only if every input is present and positive.
    frame["valid"] = frame[["E1", "E2", "P1", "P2"]].gt(0).all(axis=1)
    # RER = S * P1 / P2: the price of c1's consumption basket in units of c2's basket.
    # A rise is a real appreciation of c1 (c1 goods dearer relative to c2 goods).
    frame["RER"] = (frame["S"] * frame["P1"] / frame["P2"]).where(frame["valid"])
    return frame


def valid_range(frame: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """First and last month with a defined RER for the pair."""
    valid_months = frame.index[frame["valid"]]
    if valid_months.empty:
        return None
    return valid_months.min(), valid_months.max()


def compute_sample(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, float]:
    """Slice [start, end] inclusive, compute RER* over valid months, add real_index.

    RER* is NaN when the sample has no valid months (no division by zero)."""
    sample = frame.loc[start:end].copy()
    valid_rer = sample.loc[sample["valid"], "RER"]
    rer_star = float(valid_rer.mean()) if not valid_rer.empty else float("nan")
    sample["RER_star"] = rer_star
    sample["real_index"] = 100.0 * sample["RER"] / rer_star if rer_star > 0 else np.nan
    return sample, rer_star


def export_frame(sample: pd.DataFrame, c1: str, c2: str) -> pd.DataFrame:
    """CSV/table layout: one row per month in the sample, including invalid months."""
    out = pd.DataFrame({
        "month": sample.index.strftime("%Y-%m"),
        "country1_iso3": c1,
        "country2_iso3": c2,
        "E1_lcu_per_usd": sample["E1"].to_numpy(),
        "E2_lcu_per_usd": sample["E2"].to_numpy(),
        "S_c2_per_c1": sample["S"].to_numpy(),
        "P1_cpi": sample["P1"].to_numpy(),
        "P2_cpi": sample["P2"].to_numpy(),
        "RER": sample["RER"].to_numpy(),
        "RER_star": sample["RER_star"].to_numpy(),
        "real_index": sample["real_index"].to_numpy(),
        "valid": sample["valid"].to_numpy(dtype=bool),
    })
    return out[EXPORT_COLUMNS]


def series_coverage(df: pd.DataFrame, iso3: str) -> tuple[str, str, int]:
    """(first month, last month, observation count) for one economy's series."""
    s = df.loc[df["iso3"] == iso3, "month"]
    if s.empty:
        return "—", "—", 0
    return s.min().strftime("%Y-%m"), s.max().strftime("%Y-%m"), int(s.size)


# ---------------------------------------------------------------------------
# Labels and formatting
# ---------------------------------------------------------------------------


def country_label(iso3: str, names: dict[str, str]) -> str:
    return f"{names.get(iso3, iso3)} ({iso3})"


def currency_unit(iso3: str, names: dict[str, str]) -> str:
    """ISO currency code if known, else '<Country> currency'."""
    code = CURRENCY_CODES.get(iso3) or EURO_LEGACY_CURRENCIES.get(iso3)
    return code or f"{names.get(iso3, iso3)} currency"


def fmt_sig(x: float, sig: int = 4) -> str:
    """Format with ~`sig` significant digits; scientific only for tiny values."""
    if x is None or not np.isfinite(x):
        return "n/a"
    if x == 0:
        return "0"
    decimals = max(0, sig - 1 - int(np.floor(np.log10(abs(x)))))
    return f"{x:.{sig}g}" if decimals > 10 else f"{x:,.{decimals}f}"


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------


def make_chart(df: pd.DataFrame, view: View, labels: dict[str, str], log_nominal: bool = False,
               palette: dict[str, str] | None = None) -> go.Figure:
    """Real Index and/or nominal S(t). Invalid months are dropped, so lines connect
    across gaps without any interpolated values entering the calculations."""
    p = palette or PALETTES["light"]
    real = df["real_index"].dropna()
    nominal = df["S"].dropna()
    s_axis = labels["s_axis"]

    real_trace = go.Scatter(
        x=real.index, y=real.to_numpy(), name="Real Index", mode="lines",
        line=dict(color=p["real"], width=1.75), connectgaps=True,
        hovertemplate="Real Index: %{y:.1f}<extra></extra>",
    )
    ref_trace = go.Scatter(
        x=[df.index.min(), df.index.max()], y=[100, 100], name="Sample average (=100)", mode="lines",
        line=dict(color=p["muted"], width=1, dash="dot"), hoverinfo="skip",
    )
    nominal_trace = go.Scatter(
        x=nominal.index, y=nominal.to_numpy(), name=f"S(t): {s_axis}", mode="lines",
        line=dict(color=p["nominal"], width=1.75), connectgaps=True,
        hovertemplate=f"S(t): %{{y:,.5~g}} {s_axis}<extra></extra>",
    )

    axis_style = dict(showline=True, linewidth=1, linecolor=p["muted"], ticks="outside", ticklen=4,
                      tickwidth=1, tickcolor=p["muted"], tickfont=dict(color=p["muted"]),
                      showgrid=False, zeroline=False)
    real_axis = {**axis_style, "showgrid": True, "gridcolor": p["hairline"],
                 "title": dict(text="Real Index (sample average = 100)", font=dict(color=p["real"]))}
    nominal_axis = {**axis_style, "showgrid": True, "gridcolor": p["hairline"],
                    "title": dict(text=s_axis, font=dict(color=p["nominal"])),
                    "type": "log" if log_nominal else "linear"}

    if view == "Both":
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(real_trace, secondary_y=False)
        fig.add_trace(ref_trace, secondary_y=False)
        fig.add_trace(nominal_trace, secondary_y=True)
        fig.update_yaxes(secondary_y=False, **real_axis)
        # Own round ticks on the right axis (not synced to the left one), no second grid.
        fig.update_yaxes(secondary_y=True, **{**nominal_axis, "showgrid": False, "tickmode": "auto"})
    elif view == "Real":
        fig = go.Figure([real_trace, ref_trace])
        fig.update_yaxes(**real_axis)
    else:
        fig = go.Figure([nominal_trace])
        fig.update_yaxes(**nominal_axis)

    # Zoom by dragging; quick ranges via the selector; double-click resets.
    fig.update_xaxes(
        **axis_style, hoverformat="%Y-%m",
        rangeselector=dict(
            buttons=[
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(count=10, label="10Y", step="year", stepmode="backward"),
                dict(count=20, label="20Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor="rgba(0,0,0,0)", activecolor=p["hairline"], bordercolor=p["hairline"], borderwidth=1,
            font=dict(color=p["muted"], size=12),
        ),
    )
    fig.update_layout(
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_STACK, color=p["ink"], size=13),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=p["surface"], bordercolor=p["hairline"], font=dict(color=p["ink"])),
        height=520, margin=dict(l=70, r=70, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(color=p["muted"])),
    )
    return fig


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

# Fonts and colors come from .streamlit/config.toml; this only sets type weight,
# hairline rules and a comfortable reading width.
CSS = """
<style>
.block-container { max-width: 1240px; }
.app-name { font-size: 0.72rem; font-weight: 500; letter-spacing: 0.22em; text-transform: uppercase;
            opacity: 0.6; }
[data-testid="stMetric"] { border-top: 1px solid rgba(128, 128, 128, 0.35); padding-top: 0.7rem; }
[data-testid="stMetricLabel"] p { font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase;
                                  opacity: 0.75; }
[data-testid="stMetricValue"] { font-weight: 300; }
</style>
"""


def theme_type() -> str:
    """'light' or 'dark', following the viewer's active Streamlit theme."""
    kind = st.context.theme.type
    return kind if kind in PALETTES else "light"


def render_app_name() -> None:
    st.markdown('<div class="app-name">Real Exchange Rate Explorer</div>', unsafe_allow_html=True)


def render_intro() -> None:
    render_app_name()
    st.title("Compare real and nominal exchange rates")
    st.markdown(
        "Pick any two economies with monthly IMF consumer-price and US-dollar exchange-rate data "
        "(including the **Euro area** and the **United States**) to see how their nominal and real "
        "exchange rates have moved since 1976.\n\n"
        "1. Choose **Country 1** and **Country 2** in the sidebar.\n"
        "2. Adjust the **start** and **end month**. The Real Index re-centres on that sample.\n"
        "3. Switch the chart between **Real**, **Nominal** or **Both**.\n"
        "4. Download the monthly data and calculations as **CSV**."
    )
    st.caption(
        "S(t) = Country 2 currency per 1 unit of Country 1 currency · "
        "RER(t) = S(t) × P1(t) / P2(t) · Real Index(t) = 100 × RER(t) / RER*, "
        "where RER* is the average RER over the selected months."
    )


def render_metrics(sample: pd.DataFrame, rer_star: float, s_unit: str) -> None:
    valid = sample.loc[sample["valid"]]
    latest_month = valid.index.max()
    latest_ri = float(valid.loc[latest_month, "real_index"])
    ri = valid["real_index"]
    min_month, max_month = ri.idxmin(), ri.idxmax()
    s_known = sample["S"].dropna()
    month = lambda ts: ts.strftime("%Y-%m")  # noqa: E731

    cols = st.columns(4)
    cols[0].metric("RER* (sample-average RER)", fmt_sig(rer_star, 5),
                   help="Mean of S(t)·P1(t)/P2(t) over valid months in the sample. Its level depends on "
                        "the CPI base years, so only changes in RER are meaningful.")
    cols[1].metric("Real Index min – max", f"{ri.min():.1f} – {ri.max():.1f}",
                   help=f"Minimum {ri.min():.1f} in {month(min_month)}; "
                        f"maximum {ri.max():.1f} in {month(max_month)}.")
    cols[2].metric(f"Latest Real Index · {month(latest_month)}", f"{latest_ri:.1f}")
    if s_known.empty:
        cols[3].metric("Latest S(t)", "n/a")
    else:
        cols[3].metric(f"Latest S(t) · {month(s_known.index.max())}", fmt_sig(float(s_known.iloc[-1]), 5),
                       help=f"{s_unit}, period average")


def render_data_notes(prices: pd.DataFrame, rates: pd.DataFrame, c1: str, c2: str,
                      names: dict[str, str], sample: pd.DataFrame) -> None:
    rows = []
    for role, iso3 in (("Country 1", c1), ("Country 2", c2)):
        measure = prices.loc[prices["iso3"] == iso3, "measure"]
        measure = measure.iloc[0] if not measure.empty else "CPI"
        for label, df, flow, key in (
            ("Exchange rate (LCU per USD)", rates, ER_FLOW, f"{iso3}.XDC_USD.PA_RT.M"),
            (f"Price index ({measure})", prices, CPI_FLOW, f"{iso3}.{measure}._T.IX.M"),
        ):
            first, last, n = series_coverage(df, iso3)
            if iso3 == US and flow == ER_FLOW and n == 0:
                first, last, n = series_coverage(prices, US)
                key += " (set to 1.0)"
            rows.append({"role": role, "country": country_label(iso3, names), "series": label,
                         "IMF dataflow / key": f"{AGENCY}/{flow} · {key}",
                         "first month": first, "last month": last, "observations": n})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    excluded = int((~sample["valid"]).sum())
    notes = [f"**{excluded}** month(s) in the selected period are excluded from RER* because at least one "
             "input is missing or non-positive. They appear in the CSV with `valid = False`; chart lines "
             "simply connect across them (no interpolation)."]
    for iso3 in (c1, c2):
        if iso3 == US:
            notes.append("United States: the USD rate is 1 USD per USD by definition.")
        if iso3 == EURO_AREA:
            notes.append("Euro area (G163): prices use the HICP (the IMF publishes no separate 'CPI' for "
                         "the aggregate); exchange-rate data start in 1999 with the euro.")
        if iso3 in EURO_LEGACY_CURRENCIES:
            notes.append(f"{names.get(iso3, iso3)}: the IMF's national USD rate is quoted in the legacy "
                         f"currency ({EURO_LEGACY_CURRENCIES[iso3]}) and ends at euro adoption. For later "
                         "months choose Euro area (G163).")
    notes.append("IMF data can contain gaps, revisions, CPI rebasings (harmless here, because the Real Index "
                 "is normalised), currency redenominations (historical rates are often restated in today's "
                 "currency units) and changes in exchange-rate regime.")
    st.markdown("\n".join(f"- {n}" for n in notes))


def main() -> None:
    st.set_page_config(page_title="Real Exchange Rate Explorer", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    try:
        prices = load_prices()
        rates = load_rates()
    except IMFDataError as exc:
        render_app_name()
        st.error(f"Could not load data from the IMF API: {exc}. The service may be busy or down. "
                 "Please press **Retry** in a minute.")
        if st.button("Retry"):
            st.cache_data.clear()
            st.rerun()
        st.stop()
    try:
        names = load_country_names()
    except IMFDataError:
        names = dict(NAME_OVERRIDES)  # labels fall back to ISO3 codes

    countries = available_countries(prices, rates)
    if not countries:
        st.error("The IMF API returned no economies with both monthly CPI and exchange-rate data. Please retry later.")
        st.stop()
    labels = {iso3: country_label(iso3, names) for iso3 in countries}
    options = sorted(countries, key=lambda iso3: labels[iso3].casefold())

    with st.sidebar:
        st.header("Controls")
        c1 = st.selectbox("Country 1", options, index=None, format_func=labels.get,
                          placeholder="Choose a country…", key="country1")
        c2 = st.selectbox("Country 2", options, index=None, format_func=labels.get,
                          placeholder="Choose a country…", key="country2")
        period_box = st.container()
        view: View = st.radio("Chart view", ["Real", "Nominal", "Both"], index=2, horizontal=True)
        log_nominal = st.checkbox("Log scale for S(t)", value=False)
        st.caption(f"{len(countries)} economies available · source: IMF SDMX 3.0 API")

    if c1 is None or c2 is None:
        render_intro()
        return
    if c1 == c2:
        st.warning("Please choose two different countries.")
        return

    frame = build_pair_frame(prices, rates, c1, c2)
    s_unit = f"{currency_unit(c2, names)} per 1 {currency_unit(c1, names)}"
    render_app_name()
    st.title(f"{labels[c1]} vs {labels[c2]}")
    st.caption(f"S(t) = {s_unit} — Country 2 currency per 1 unit of Country 1 currency. "
               "RER(t) = S(t) × P1(t) / P2(t); Real Index(t) = 100 × RER(t) / RER*. "
               "A rising Real Index is a real appreciation of Country 1's currency against Country 2's.")

    bounds = valid_range(frame)
    if bounds is None:
        st.warning(f"{labels[c1]} and {labels[c2]} have no month in which both exchange rates and both "
                   "price indices are available. Try a different pair.")
        with st.expander("Data notes", expanded=True):
            render_data_notes(prices, rates, c1, c2, names, frame)
        return

    months = pd.date_range(*bounds, freq="MS")
    month_labels = [m.strftime("%Y-%m") for m in months]
    with period_box:
        start_label = st.selectbox("Start month", month_labels, index=0, key=f"start_{c1}_{c2}")
        end_label = st.selectbox("End month", month_labels, index=len(month_labels) - 1, key=f"end_{c1}_{c2}")
    start, end = pd.Timestamp(start_label), pd.Timestamp(end_label)
    if start > end:
        st.error("The start month must be on or before the end month.")
        return

    sample, rer_star = compute_sample(frame, start, end)
    if not np.isfinite(rer_star):
        st.warning("There are no valid months (all four series present) in the selected period, so the "
                   "real exchange rate cannot be averaged. Widen the period.")
        return

    render_metrics(sample, rer_star, s_unit)

    fig = make_chart(sample, view, {"s_axis": s_unit}, log_nominal=log_nominal, palette=PALETTES[theme_type()])
    st.plotly_chart(fig, width="stretch", theme=None,
                    config={"displaylogo": False, "toImageButtonOptions": {"filename": f"RER_{c1}_{c2}"}})

    table = export_frame(sample, c1, c2)
    with st.expander("View data table"):
        st.dataframe(table, hide_index=True, width="stretch")
    st.download_button(
        "Download CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"RER_{c1}_{c2}_{start_label}_{end_label}.csv",
        mime="text/csv",
    )
    with st.expander("Data notes"):
        render_data_notes(prices, rates, c1, c2, names, sample)


if __name__ == "__main__":
    main()
