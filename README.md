# Real Exchange Rate Explorer

A Streamlit app for comparing the **nominal** and **real** exchange rate between any two economies in the IMF's monthly statistics, including the **Euro area** and the **United States**. Data are monthly, from January 1976 onward where available, and are pulled live from the IMF SDMX 3.0 API. No API key is needed.

## Interface

- **Sidebar.** Country 1 and Country 2 pickers (both start empty), a *Real / Nominal / Both* chart switch (default *Both*), and an optional log scale for S(t).
- **Main area.** Before both countries are chosen, it shows only a short introduction. After that it shows:
  - the pair title, e.g. *United States (USA) vs Japan (JPN)*, and the definition of S(t) in currency terms (*JPY per 1 USD*)
  - the **sample period**: start year, start month, end year and end month pickers, limited to the months where the pair has data, plus *5Y / 10Y / 20Y / All* quick ranges counted back from the last valid month. RER\* and the Real Index are recalculated whenever the period changes.
  - four summary metrics: RER\* (labelled with its period), latest Real Index, the Real Index min–max range (hover for the months), and latest S(t)
  - a Plotly chart of the selected period: the Real Index on the left axis with a dotted reference line at 100 (the period average), and S(t) on the right axis. Hover shows monthly values. Zoom is disabled so that the chart always shows exactly the months RER\* is averaged over; change the period with the pickers instead.
  - an expandable data table, a CSV download button, and data-coverage notes

## Economic definitions

| Symbol | Definition |
|---|---|
| **E1(t), E2(t)** | Country 1 / Country 2 currency per USD (IMF period-average rate) |
| **S(t)** | **E2(t) / E1(t)**: units of Country 2 currency per 1 unit of Country 1 currency (the nominal exchange rate) |
| **P1(t), P2(t)** | All-items consumer price index of Country 1 / Country 2 |
| **RER(t)** | **S(t) × P1(t) / P2(t)**: the real exchange rate |
| **RER\*** | Mean of RER(t) over the *valid* months in the selected sample |
| **Real Index(t)** | **100 × RER(t) / RER\***. It averages exactly 100 over the valid months of the sample |

S = E2/E1 has this direction because (C2 per USD) ÷ (C1 per USD) = C2 per C1. For example, Country 1 = USA and Country 2 = JPN gives S ≈ 145 JPY per USD. Swapping the countries gives the reciprocal.

**Interpretation.** RER(t) is the price of Country 1's consumption basket measured in Country 2's baskets.

- **A rising Real Index** is a *real appreciation* of Country 1's currency: Country 1 goods have become more expensive relative to Country 2 goods, through the exchange rate, relative inflation, or both.
- **A falling Real Index** is a real depreciation of Country 1's currency.
- A value of 120 means Country 1's currency is 20% stronger in real terms than its average over the chosen sample.

The *level* of RER depends on arbitrary CPI base years, so only its movements, and the Real Index built from it, are meaningful. RER\*, and with it the Real Index, is recomputed whenever the countries or the sample period change.

## Data source

The data come from the IMF SDMX 3.0 API at `https://api.imf.org/external/sdmx/3.0`. Each item below was verified against the live API.

| Item | Request |
|---|---|
| Consumer prices | `GET /data/dataflow/IMF.STA/CPI/+/*.CPI+HICP._T.IX.M?c[TIME_PERIOD]=ge:1976-01` |
| Exchange rates | `GET /data/dataflow/IMF.STA/ER/+/*.XDC_USD.PA_RT.M?c[TIME_PERIOD]=ge:1976-01` |
| Country names | `GET /structure/codelist/IMF/CL_COUNTRY` |

- **Series keys.**
  - Prices: `{ISO3}.CPI._T.IX.M`, the all-items CPI index, monthly. Dimensions: COUNTRY.INDEX_TYPE.COICOP_1999.TYPE_OF_TRANSFORMATION.FREQUENCY.
  - Exchange rates: `{ISO3}.XDC_USD.PA_RT.M`, the period-average rate in local currency per USD, monthly.
- **Formats.** Data are requested as SDMX-CSV 2.0 (`Accept: application/vnd.sdmx.data+csv;version=2.0.0`) and the codelist as SDMX-JSON 2.0. Both are parsed with `requests` and `pandas`. The `+` in the path selects the latest stable dataflow version.
- **Euro area = `G163`.** The IMF publishes no series with index type `CPI` for the Euro area aggregate, only the **HICP** (`G163.HICP._T.IX.M`). The app therefore requests both index types and uses CPI wherever an economy has it, falling back to HICP only when it doesn't. In practice the fallback applies only to G163. The Euro area exchange rate (`G163.XDC_USD.PA_RT.M`, EUR per USD) starts in January 1999.
- **United States.** The USD-per-USD rate is 1 by definition. The IMF does publish `USA.XDC_USD.PA_RT.M` (equal to 1). If that series is ever missing, the app sets E(t) = 1.0 in every month where US CPI exists.
- **Bulk download and caching.** Two wildcard requests (about 10 MB each) fetch every country at once. They are cached for 24 hours with `st.cache_data`, so choosing countries triggers no further API calls. If a bulk request fails, the app falls back to probing countries in parallel with a bounded thread pool. Requests send a descriptive `User-Agent`, time out after 30 s, and retry with exponential backoff on HTTP 429/5xx and connection errors.
- **Country list.** It shows only economies with at least one valid monthly CPI/HICP observation *and* a monthly USD rate, plus the USA rule above. That is about 190 economies, labelled like `Japan (JPN)` and `Euro area (G163)`.
- **Time periods.** The IMF returns periods like `1976-M01`. The parser also accepts `1976-01`, `1976M1` and `1976-01-31`.

## Missing-data policy

- An observation is **valid** only if it is numeric and > 0. Blanks, non-numeric and non-positive values count as missing.
- A **month is valid** only if E1, E2, P1 and P2 are all present and positive. RER(t) is defined only for valid months.
- **RER\*** averages valid months only. If the selected period has no valid months, the app shows a warning instead of dividing by zero.
- **Charts** drop invalid months, so lines connect straight across gaps. **No value is ever interpolated** into the calculations or the CSV.
- The **CSV** has one row per month from the start month to the end month inclusive, including invalid months, with `valid = False`. Columns:
  `month, country1_iso3, country2_iso3, E1_lcu_per_usd, E2_lcu_per_usd, S_c2_per_c1, P1_cpi, P2_cpi, RER, RER_star, real_index, valid`.
  The file name looks like `RER_JPN_USA_1976-01_2026-08.csv`.

## Run locally (Python 3.12)

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Run the last command from the repository root so Streamlit picks up `.streamlit/config.toml`.

The first page load downloads the IMF data, which takes a few seconds to about 20 s. A spinner reads "Loading IMF data… this happens once per day". After that, everything is served from the cache.

## Deploy on Streamlit Community Cloud

Push `app.py`, `requirements.txt`, `README.md` and `.streamlit/config.toml` (the app theme) to a GitHub repository. Then go to [share.streamlit.io](https://share.streamlit.io), click **Create app**, pick the repo and branch, set the entry point to **`app.py`**, and choose Python 3.12 under *Advanced settings*. No secrets are needed.

## Known limitations

- **Coverage gaps.** Not every economy reports monthly CPI. New Zealand and Taiwan, for example, have no monthly CPI in the IMF CPI dataset, so they are not listed. Series start and end at different dates, and single months can be missing; for example, US CPI has no October 2025 observation. The *Data notes* expander shows each series' first and last month for the chosen pair.
- **Euro-area members.** The IMF's national USD rates for euro members (e.g. Germany, France, Croatia) are quoted in the **legacy currency** (DEM, FRF, HRK, …) and **end at euro adoption**. A pair like Germany vs USA therefore stops in December 1998. Use **Euro area (G163)** for later periods. The Euro area itself has data only from 1999 (exchange rate) and December 1999 (HICP).
- **Revisions.** IMF data are revised, so results can change slightly from day to day after the 24-hour cache refreshes.
- **CPI rebasing.** Different base years across countries or over time are harmless here because the Real Index is normalised to its sample mean. Only the arbitrary level of RER\* depends on the bases.
- **Redenominations and regime changes.** Historical rates are usually restated in today's currency units, which gives tiny early values for Brazil, Argentina and Türkiye. Pegs, dual rates and hyperinflations can make some series hard to interpret.
- **Currency labels.** The IMF metadata carries no currency codes, so axis labels use a small hand-curated ISO 4217 map for major economies. Other economies are labelled "*Country* currency".
- **First-load latency.** The first visit after a cache refresh, or after the app wakes on Community Cloud, downloads about 20 MB from the IMF.
