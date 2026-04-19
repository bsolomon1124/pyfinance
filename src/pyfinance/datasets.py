"""Financial dataset web scrubbing.

Descriptions
------------
load_13f
    Parse SEC 13F XML file to pandas DataFrame.
load_industries
    Load industry portfolio returns from Ken French's website.
load_rates
    Load database of interest rates (CP, corporate, government).
load_rf
    Build a risk-free rate return series using 3-month US T-bill yields.

Note
----
Several loaders that previously lived here (`load_factors`,
`load_shiller`, `load_retaildata`) were removed in pyfinance 2.0
because their upstream sources had drifted past reasonable repair
(dead bit.ly redirects, restructured CBOE URLs, deprecated census.gov
endpoints, schema changes in Shiller's `ie_data.xls`). See CHANGELOG.
"""

from __future__ import annotations

import itertools
from typing import Any, Literal

import pandas as pd
import requests
import xmltodict
from pandas.tseries import offsets

RateFreq = Literal["D", "W", "M"]
RfFreq = Literal["D", "W", "M", "Q", "A"]

__all__ = [
    "load_13f",
    "load_industries",
    "load_rates",
    "load_rf",
]


class _LazyPandasDataReader:
    """Defer `pandas_datareader` import until it is actually used.

    Why: pandas-datareader 0.10.0 (the latest release) still imports
    the stdlib `distutils` module, which was removed in Python 3.12.
    Importing lazily lets the rest of `pyfinance.datasets` load on
    Python 3.12+ and only surfaces the error to callers that actually
    reach into pandas-datareader functionality.
    """

    def __getattr__(self, name: str) -> Any:
        import pandas_datareader as _pdr

        return getattr(_pdr, name)


pdr = _LazyPandasDataReader()

# Default start date for web-retrieved time series.
DSTART = "1950-01"


def load_13f(url: str, user_agent: str) -> pd.DataFrame:
    """Load and parse an SEC.gov-hosted 13F XML file to a pandas DataFrame.

    Provide the URL to the raw .xml form13fInfoTable. (See example below.)

    SEC EDGAR requires every HTTP client to identify itself via a
    descriptive ``User-Agent`` header including contact information;
    requests without one are denied at the gateway. See the policy at
    https://www.sec.gov/os/accessing-edgar-data.

    Parameters
    ----------
    url : str
        Link to the .xml ``form13fInfoTable`` file.
    user_agent : str
        A descriptive User-Agent, including a contact email, per SEC
        EDGAR's fair-access policy. For example:
        ``"Jane Doe (jane@example.com)"``.

    Returns
    -------
    df : pd.DataFrame
        Holdings snapshot.

    Example
    -------
    >>> from pyfinance import datasets
    >>> url = (
    ...     "https://www.sec.gov/Archives/edgar/data/1040273/"
    ...     "000108514617001787/form13fInfoTable.xml"
    ... )
    >>> df = datasets.load_13f(url, user_agent="Jane Doe (jane@example.com)")
    """

    if not user_agent or not isinstance(user_agent, str):
        raise ValueError(
            "SEC EDGAR requires a descriptive `user_agent` string with "
            "contact information. See https://www.sec.gov/os/accessing-edgar-data."
        )
    headers = {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    data = xmltodict.parse(resp.text)["informationTable"]["infoTable"]
    df = pd.DataFrame(data).drop(["shrsOrPrnAmt", "investmentDiscretion"], axis=1)
    df["votingAuthority"] = df["votingAuthority"].apply(lambda d: d["Sole"])
    df.loc[:, "value"] = pd.to_numeric(df["value"], errors="coerce")
    df.loc[:, "votingAuthority"] = pd.to_numeric(df["votingAuthority"], errors="coerce")
    return df


def load_industries() -> dict[int, pd.DataFrame]:
    """Load industry portfolio returns from Ken French's website.

    Returns
    -------
    industries : dictionary of Pandas DataFrames
        Each key is a portfolio group.

    Example
    -------
    >>> from pyfinance import datasets
    >>> ind = datasets.load_industries()

    # Monthly returns to 5 industry portfolios
    >>> ind[5].head()
                Cnsmr  Manuf  HiTec  Hlth   Other
    Date
    1950-01-31   1.26   1.47   3.21   1.06   3.19
    1950-02-28   1.91   1.29   2.06   1.92   1.02
    1950-03-31   0.28   1.93   3.46  -2.90  -0.68
    1950-04-30   3.22   5.21   3.58   5.52   1.50
    1950-05-31   3.81   6.18   1.07   3.96   1.36
    """

    n = [5, 10, 12, 17, 30, 38, 48]
    port = (f"{i}_Industry_Portfolios" for i in n)
    rets = []
    for p in port:
        ret = pdr.get_data_famafrench(p, start=DSTART)[0]
        rets.append(ret.to_timestamp(how="end", copy=False))
    return dict(zip(n, rets, strict=False))


def load_rates(freq: RateFreq = "D") -> pd.DataFrame:
    """Load interest rates from https://fred.stlouisfed.org/.

    Parameters
    ----------
    freq : str {'D', 'W', 'M'}, default 'D'
        Frequency of time series; daily, weekly, or monthly.

    Notes
    -----
    FRED occasionally discontinues individual series. If a caller
    receives ``pandas_datareader`` errors about an unknown symbol, the
    most likely cause is an upstream discontinuation; drop the
    offending code from the corresponding frequency list below.

    Original source
    ---------------
    Board of Governors of the Federal Reserve System
    H.15 Selected Interest Rates
    https://www.federalreserve.gov/releases/h15/
    """

    months = [1, 3, 6]
    years = [1, 2, 3, 5, 7, 10, 20, 30]

    # Nested dictionaries of symbols from fred.stlouisfed.org
    nom = {
        "D": [f"DGS{m}MO" for m in months] + [f"DGS{y}" for y in years],
        "W": [f"WGS{m}MO" for m in months] + [f"WGS{y}YR" for y in years],
        "M": [f"GS{m}M" for m in months] + [f"GS{y}" for y in years],
    }

    tips = {
        "D": [f"DFII{y}" for y in years[3:7]],
        "W": [f"WFII{y}" for y in years[3:7]],
        "M": [f"FII{y}" for y in years[3:7]],
    }

    fcp = {
        "D": ["DCPF1M", "DCPF2M", "DCPF3M"],
        "W": ["WCPF1M", "WCPF2M", "WCPF3M"],
        "M": ["CPF1M", "CPF2M", "CPF3M"],
    }

    nfcp = {
        "D": ["DCPN1M", "DCPN2M", "DCPN3M"],
        "W": ["WCPN1M", "WCPN2M", "WCPN3M"],
        "M": ["CPN1M", "CPN2M", "CPN3M"],
    }

    short = {
        "D": ["DFF", "DPRIME", "DPCREDIT"],
        "W": ["FF", "WPRIME", "WPCREDIT"],
        "M": ["FEDFUNDS", "MPRIME", "MPCREDIT"],
    }

    rates = list(
        itertools.chain.from_iterable([d[freq] for d in [nom, tips, fcp, nfcp, short]])
    )
    rates = pdr.DataReader(rates, "fred", start=DSTART)

    l1 = (
        ["Nominal"] * 11
        + ["TIPS"] * 4
        + ["Fncl CP"] * 3
        + ["Non-Fncl CP"] * 3
        + ["Short Rates"] * 3
    )

    l2 = (
        [f"{m}m" for m in months]
        + [f"{y}y" for y in years]
        + [f"{y}y" for y in years[3:7]]
        + 2 * [f"{m}m" for m in range(1, 4)]
        + ["Fed Funds", "Prime Rate", "Primary Credit"]
    )

    rates.columns = pd.MultiIndex.from_arrays([l1, l2])

    return rates


def load_rf(freq: RfFreq = "M") -> pd.Series:
    """Build a risk-free rate return series using 3-month US T-bill yields.

    The 3-Month Treasury Bill: Secondary Market Rate from the Federal Reserve
    (a yield) is convert to a total return.  See 'Methodology' for details.

    The time series should closely mimic returns of the BofA Merrill Lynch US
    Treasury Bill (3M) (Local Total Return) index.

    Parameters
    ----------
    freq : str, sequence, or set
        If a single-character string, return a single-column DataFrame with
        index frequency corresponding to `freq`.  If a sequence or set, return
        a dict of DataFrames with the keys corresponding to `freq`(s)

    Methodology
    -----------
    The Federal Reserve publishes a daily chart of Selected Interest Rates
    (release H.15; www.federalreserve.gov/releases/h15/).  As with a yield
    curve, some yields are interpolated from recent issues because Treasury
    auctions do not occur daily.

    While the de-annualized ex-ante yield itself is a fairly good tracker of
    the day's total return, it is not perfect and can exhibit non-neglible
    error in periods of volatile short rates.  The purpose of this function
    is to convert yields to total returns for 3-month T-bills.  It is a
    straightforward process given that these are discount (zero-coupon)
    securities.  It consists of buying a 3-month bond at the beginning of each
    month, then amortizing that bond throughout the month to back into the
    price of a <3-month tenor bond.

    The source data (pulled from fred.stlouisfed.org) is quoted on a discount
    basis.  (See footnote 4 from release H.15.)  This is converted to a
    bond-equivlanet yield (BEY) and then translated to a hypothetical daily
    total return.

    The process largely follows Morningstar's published Return Calculation of
    U.S. Treasury Constant Maturity Indices, and is as follows:
    - At the beginning of each month a bill is purchased at the prior month-end
      price, and daily returns in the month reflect the change in daily
      valuation of this bill
    - If t is not a business day, its yield is the yield of the prior
      business day.
    - At each day during the month, the price of a 3-month bill purchased on
      the final calendar day of the previous month is computed.
    - Month-end pricing is unique.  At each month-end date, there are
      effectively two bonds and two prices.  The first is the bond
      hypothetically purchased on the final day of the prior month with 2m
      remaining to maturity, and the second is a new-issue bond purchased that
      day with 3m to maturity.  The former is used as the numerator to compute
      that day's total return, while the latter is used as the denominator
      to compute the next day's (1st day of next month) total return.

    Description of the BofA Merrill Lynch US 3-Month Treasury Bill Index:
    The BofA Merrill Lynch US 3-Month Treasury Bill Index is comprised of a
    single issue purchased at the beginning of the month and held for a full
    month. At the end of the month that issue is sold and rolled into a newly
    selected issue. The     issue selected at each month-end rebalancing is the
    outstanding Treasury Bill that matures closest to, but not beyond, three
    months from the rebalancing date. To qualify for selection, an issue must
    have settled on or before the month-end rebalancing date.
        (Source: Bank of America Merrill Lynch)

    See also
    --------
    FRED: 3-Month Treasury Bill: Secondary Market Rate (DTB3)
      https://fred.stlouisfed.org/series/DTB3
    McGraw-Hill/Irwin, Interest Rates, 2008.
      https://people.ucsc.edu/~lbaum/econ80h/LS-Chap009.pdf
    Morningstar, Return Calculation of U.S. Treasury Constant Maturity Indices,
      September 2008.
    """

    freqs = "DWMQA"
    freq = freq.upper()
    if freq not in freqs:
        raise ValueError(
            "`freq` must be either a single element or subset"
            f" from {freqs}, case-insensitive"
        )
    # Pandas 2.2+ deprecated single-letter month/quarter/year codes.
    freq = {"M": "ME", "Q": "QE", "A": "YE"}.get(freq, freq)

    # Load daily 3-Month Treasury Bill: Secondary Market Rate.
    # Note that this is on discount basis and will be converted to BEY.
    # Periodicity is daily.
    rates = (
        pdr.DataReader("DTB3", "fred", DSTART)
        .mul(0.01)
        .asfreq("D", method="ffill")
        .ffill()
        .squeeze()
    )

    # Algebra doesn't 'work' on DateOffsets, don't simplify here!
    minus_one_month = offsets.MonthEnd(-1)
    plus_three_months = offsets.MonthEnd(3)
    trigger = rates.index.is_month_end
    dtm_old = rates.index + minus_one_month + plus_three_months - rates.index
    dtm_new = (
        rates.index.where(trigger, rates.index + minus_one_month)
        + plus_three_months
        - rates.index
    )

    # This does 2 things in one step:
    # (1) convert discount yield to BEY
    # (2) get the price at that BEY and days to maturity
    # The two equations are simplified
    # See https://people.ucsc.edu/~lbaum/econ80h/LS-Chap009.pdf
    p_old = (100 / 360) * (360 - rates * dtm_old.days)
    p_new = (100 / 360) * (360 - rates * dtm_new.days)

    res = p_old.pct_change().where(trigger, p_new.pct_change()).dropna()
    # TODO: For purpose of using in TSeries, we should drop upsampled
    #       periods where we don't have the full period constituents.
    return res.add(1.0).resample(freq).prod().sub(1.0)
