"""Financial dataset web scrubbing.

Descriptions
------------
load_13f
    Parse SEC 13F XML file to pandas DataFrame.
load_factors
    Load risk factor returns.
load_industries
    Load industry portfolio returns from Ken French's website.
load_rates
    Load database of interest rates (CP, corporate, government).
load_rf
    Build a risk-free rate return series using 3-month US T-bill yields.
load_retaildata
    Load and clean retail trade data from census.gov.
"""

__author__ = 'Brad Solomon <brad.solomon.1124@gmail.com>'
__all__ = ['load_factors', 'load_industries', 'load_rates', 'load_shiller',
           'load_rf', 'load_13f']

import itertools

import numpy as np
import pandas as pd
from pandas.tseries import offsets
import pandas_datareader as pdr
import requests
import xmltodict

# Default start date for web-retrieved time series.
DSTART = '1950-01'


def load_13f(url):
    """Load and parse an SEC.gov-hosted 13F XML file to Pandas DataFrame.

    Provide the URL to the raw .xml form13fInfoTable.  (See example below.)

    Parameters
    ----------
    url : str
        Link to .xml file.

    Returns
    -------
    df : pd.DataFrame
        Holdings snapshot.

    Example
    -------
    # Third Point LLC June 2017 13F
    >>> from pyfinance import datasets
    >>> url = 'https://www.sec.gov/Archives/edgar/data/1040273/000108514617001787/form13fInfoTable.xml'  # noqa
    >>> df = datasets.load_13f(url=url)

    .. _U.S. SEC: Accessing EDGAR Data:
       https://www.sec.gov/edgar/searchedgar/accessing-edgar-data.htm
    """

    resp = requests.get(url).text
    data = xmltodict.parse(resp)['informationTable']['infoTable']
    df = pd.DataFrame(data).drop(['shrsOrPrnAmt', 'investmentDiscretion'],
                                 axis=1)
    df['votingAuthority'] = df['votingAuthority'].apply(lambda d: d['Sole'])
    df.loc[:, 'value'] = pd.to_numeric(df['value'], errors='coerce')
    df.loc[:, 'votingAuthority'] = pd.to_numeric(df['votingAuthority'],
                                                 errors='coerce')
    return df


def load_factors():
    """Load risk factor returns.

    Factors
    -------
    Symbol      Description                                            Source
    ------      ----------                                             ------
    MKT                                                                French
    SMB         Size (small minus big)                                 French
    HML         Value (high minus low)                                 French
    RMW         Profitability (robust minus weak)                      French
    CMA         Investment (conservative minus aggressive)             French
    UMD         Momentum (up minus down)                               French
    STR         Short-term reversal                                    French
    LTR         Long-term reversal                                     French
    BETA        Beta                                                   French
    ACC         Accruals                                               French
    VAR         Variance                                               French
    IVAR        Residual variance                                      French
    EP          Earnings-to-price                                      French
    CP          Cash flow-to-price                                     French
    DP          Dividend-to-price                                      French
    BAB         Betting against beta                                   AQR
    QMJ         Quality minus junk                                     AQR
    HMLD        Value (high minus low) [modified version]              AQR
    LIQ         Liquidity                                              Pastor
    BDLB        Bond lookback straddle                                 Hsieh
    FXLB        Curency lookback straddle                              Hsieh
    CMLB        Commodity lookback straddle                            Hsieh
    IRLB        Interest rate lookback straddle                        Hsieh
    STLB        Stock lookback straddle                                Hsieh
    PUT         CBOE S&P 500 PutWrite Index                            CBOE
    BXM         CBOE S&P 500 BuyWrite Index®                           CBOE
    RXM         CBOE S&P 500 Risk Reversal Index                       CBOE

    Source Directory
    ----------------
    Source      Link
    ------      ----
    French      http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html  # noqa
    Pastor      http://faculty.chicagobooth.edu/lubos.pastor/research/liq_data_1962_2016.txt  # noqa
    AQR         https://www.aqr.com/library/data-sets
    Hsieh       https://faculty.fuqua.duke.edu/~dah7/HFData.htm
    Fed         https://fred.stlouisfed.org/
    CBOE        http://www.cboe.com/products/strategy-benchmark-indexes
    """

    # TODO: factors elegible for addition
    #   VIIX, VIIZ, XIV, ZIV, CRP (AQR)
    #   http://www.cboe.com/micro/buywrite/monthendpricehistory.xls ends 2016
    #   could use:
    #   http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/putdailyprice.csv

    # Warning: slow, kludgy data retrieval follows
    # ------------------------------------------------------------------------

    # `tgt` will become a list of DataFrames and eventually concatenated
    tgt = []

    # MKT, SMB, HML, RMW, CMA, RF, UMD, STR, LTR
    facs = [
        'F-F_Research_Data_5_Factors_2x3',
        'F-F_Momentum_Factor',
        'F-F_ST_Reversal_Factor',
        'F-F_LT_Reversal_Factor'
        ]

    for fac in facs:
        tgt.append(pdr.DataReader(fac, 'famafrench', DSTART)[0])

    # BETA, ACC, VAR, IVAR require some manipulation to compute returns
    # in the dual-sort method of Fama-French
    for i in ['BETA', 'AC', 'VAR', 'RESVAR']:
        ser = pdr.DataReader('25_Portfolios_ME_' + i + '_5x5', 'famafrench',
                             DSTART)[0]
        ser = ser.iloc[:, [0, 5, 10, 15, 20]].mean(axis=1)\
            - ser.iloc[:, [4, 9, 14, 19, 24]].mean(axis=1)
        ser = ser.rename(i)
        tgt.append(ser)

    # E/P, CF/P, D/P (univariate sorts, quintile spreads)
    for i in ['E-P', 'CF-P', 'D-P']:
        ser = pdr.DataReader('Portfolios_Formed_on_' + i, 'famafrench',
                             DSTART)[0]
        ser = ser.loc[:, 'Hi 20'] - ser.loc[:, 'Lo 20']
        ser = ser.rename(i)
        tgt.append(ser)

    tgt = [df.to_timestamp(how='end') for df in tgt]

    # BAB, QMJ, HMLD
    # TODO: performance is poor here, runtime is eaten up by these 3
    links = {'BAB': 'http://bit.ly/2hWyaG8',
             'QMJ': 'http://bit.ly/2hUBSgF',
             'HMLD': 'http://bit.ly/2hdVb7G'}
    for key, value in links.items():
        ser = pd.read_excel(value, header=18, index_col=0)['USA'] * 100
        ser = ser.rename(key)
        tgt.append(ser)

    # Lookback straddles
    link = 'http://faculty.fuqua.duke.edu/~dah7/DataLibrary/TF-Fac.xls'
    straddles = pd.read_excel(link, header=14, index_col=0)
    straddles.index = pd.DatetimeIndex(straddles.index.astype(str) + '01') \
        + offsets.MonthEnd(1)
    straddles = straddles * 100.
    tgt.append(straddles)

    # LIQ
    link = 'http://bit.ly/2pn2oBK'
    liq = pd.read_csv(link, skiprows=14, delim_whitespace=True, header=None,
                      usecols=[0, 3], index_col=0, names=['date', 'LIQ'])
    liq.index = pd.DatetimeIndex(liq.index.astype(str) + '01') \
        + offsets.MonthEnd(1)
    liq = liq.replace(-99, np.nan) * 100.
    tgt.append(liq)

    # USD, HY
    fred = pdr.DataReader(['DTWEXB', 'BAMLH0A0HYM2'], 'fred', DSTART)
    fred = (fred.asfreq('D', method='ffill')
            .fillna(method='ffill')
            .asfreq('M'))
    fred.loc[:, 'DTWEXB'] = fred['DTWEXB'].pct_change() * 100.
    fred.loc[:, 'BAMLH0A0HYM2'] = fred['BAMLH0A0HYM2'].diff()
    tgt.append(fred)

    # PUT, BXM, RXM (CBOE options strategy indices)
    link1 = 'http://www.cboe.com/micro/put/put_86-06.xls'
    link2 = 'http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/putdailyprice.csv'  # noqa

    put1 = pd.read_excel(link1, index_col=0, skiprows=6, header=None)\
        .rename_axis('DATE')
    put2 = pd.read_csv(link2, index_col=0, parse_dates=True, skiprows=7,
                       header=None).rename_axis('DATE')
    put = pd.concat((put1, put2))\
        .rename(columns={1: 'PUT'})\
        .iloc[:, 0]\
        .asfreq('D', method='ffill')\
        .fillna(method='ffill')\
        .asfreq('M')\
        .pct_change() * 100.
    tgt.append(put)

    link1 = 'http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/bxmarchive.csv'  # noqa
    link2 = 'http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/bxmcurrent.csv'  # noqa

    bxm1 = pd.read_csv(link1, index_col=0, parse_dates=True, skiprows=5,
                       header=None).rename_axis('DATE')
    bxm2 = pd.read_csv(link2, index_col=0, parse_dates=True, skiprows=4,
                       header=None).rename_axis('DATE')
    bxm = pd.concat((bxm1, bxm2))\
        .rename(columns={1: 'BXM'})\
        .iloc[:, 0]\
        .asfreq('D', method='ffill')\
        .fillna(method='ffill')\
        .asfreq('M')\
        .pct_change() * 100.
    tgt.append(bxm)

    link = 'http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/rxm_historical.csv'  # noqa
    rxm = pd.read_csv(link, index_col=0, parse_dates=True, skiprows=2,
                      header=None)\
        .rename(columns={1: 'RXM'})\
        .rename_axis('DATE')\
        .iloc[:, 0]\
        .asfreq('D', method='ffill')\
        .fillna(method='ffill')\
        .asfreq('M')\
        .pct_change() * 100.
    tgt.append(rxm)

    # Clean up data retrieved above
    # -----------------------------------------------------------------

    factors = pd.concat(tgt, axis=1).round(2)
    newnames = {
        'Mkt-RF': 'MKT',
        'Mom   ': 'UMD',
        'ST_Rev': 'STR',
        'LT_Rev': 'LTR',
        'RESVAR': 'IVAR',
        'AC': 'ACC',
        'PTFSBD': 'BDLB',
        'PTFSFX': 'FXLB',
        'PTFSCOM': 'CMLB',
        'PTFSIR': 'IRLB',
        'PTFSSTK': 'STLB',
        'DTWEXB': 'USD',
        'BAMLH0A0HYM2': 'HY'
        }
    factors.rename(columns=newnames, inplace=True)

    # Get last valid RF date; returns will be constrained to this date
    factors = factors[:factors['RF'].last_valid_index()]

    # Subtract RF for long-only factors
    subtract = ['HY', 'PUT', 'BXM', 'RXM']

    for i in subtract:
        factors.loc[:, i] = factors[i] - factors['RF']

    return factors


def load_industries():
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
    port = ('%s_Industry_Portfolios' % i for i in n)
    rets = []
    for p in port:
        ret = pdr.get_data_famafrench(p, start=DSTART)[0]
        rets.append(ret.to_timestamp(how='end', copy=False))
    industries = dict(zip(n, rets))
    return industries


def load_rates(freq='D'):
    """Load interest rates from https://fred.stlouisfed.org/.

    Parameters
    ----------
    reload : bool, default True
        If True, download the data from source rather than loading pickled data
    freq : str {'D', 'W', 'M'}, default 'D'
        Frequency of time series; daily, weekly, or monthly
    start : str or datetime, default '1963', optional
        Start date of time series
    dropna : bool, default True
        If True, drop NaN along rows in resulting DataFrame
    how : str, default 'any'
        Passed to dropna()

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
        'D': ['DGS%sMO' % m for m in months] + ['DGS%s' % y for y in years],
        'W': ['WGS%sMO' % m for m in months] + ['WGS%sYR' % y for y in years],
        'M': ['GS%sM' % m for m in months] + ['GS%s' % y for y in years]
        }

    tips = {
        'D': ['DFII%s' % y for y in years[3:7]],
        'W': ['WFII%s' % y for y in years[3:7]],
        'M': ['FII%s' % y for y in years[3:7]]
        }

    fcp = {
        'D': ['DCPF1M', 'DCPF2M', 'DCPF3M'],
        'W': ['WCPF1M', 'WCPF2M', 'WCPF3M'],
        'M': ['CPF1M', 'CPF2M', 'CPF3M']
        }

    nfcp = {
        'D': ['DCPN30', 'DCPN2M', 'DCPN3M'],
        'W': ['WCPN1M', 'WCPN2M', 'WCPN3M'],
        'M': ['CPN1M', 'CPN2M', 'CPN3M']
        }

    short = {
        'D': ['DFF', 'DPRIME', 'DPCREDIT'],
        'W': ['FF', 'WPRIME', 'WPCREDIT'],
        'M': ['FEDFUNDS', 'MPRIME', 'MPCREDIT']
        }

    rates = list(itertools.chain.from_iterable([d[freq] for d in
                 [nom, tips, fcp, nfcp, short]]))
    rates = pdr.DataReader(rates, 'fred', start=DSTART)

    l1 = ['Nominal'] * 11 + ['TIPS'] * 4 + ['Fncl CP'] * 3 \
        + ['Non-Fncl CP'] * 3 + ['Short Rates'] * 3

    l2 = ['%sm' % m for m in months] + ['%sy' % y for y in years] \
        + ['%sy' % y for y in years[3:7]] \
        + 2 * ['%sm' % m for m in range(1, 4)] \
        + ['Fed Funds', 'Prime Rate', 'Primary Credit']

    rates.columns = pd.MultiIndex.from_arrays([l1, l2])

    return rates


def load_rf(freq='M'):
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

    freqs = 'DWMQA'
    freq = freq.upper()
    if freq not in freqs:
        raise ValueError('`freq` must be either a single element or subset'
                         ' from %s, case-insensitive' % freqs)

    # Load daily 3-Month Treasury Bill: Secondary Market Rate.
    # Note that this is on discount basis and will be converted to BEY.
    # Periodicity is daily.
    rates = pdr.DataReader('DTB3', 'fred', DSTART)\
        .mul(0.01)\
        .asfreq('D', method='ffill')\
        .fillna(method='ffill')\
        .squeeze()

    # Algebra doesn't 'work' on DateOffsets, don't simplify here!
    minus_one_month = offsets.MonthEnd(-1)
    plus_three_months = offsets.MonthEnd(3)
    trigger = rates.index.is_month_end
    dtm_old = rates.index + minus_one_month + plus_three_months - rates.index
    dtm_new = rates.index.where(trigger, rates.index + minus_one_month) \
        + plus_three_months - rates.index

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
    return res.add(1.).resample(freq).prod().sub(1.)


def load_shiller():
    """Load market & macroeconomic data from Robert Shiller's website.

    Returns
    -------
    iedata : pd.DataFrame
        Time series of S&P 500 and interest rate variables.

    Example
    -------
    >>> from pyfinance import datasets
    >>> shiller = datasets.load_shiller()
    >>> shiller.iloc[:7, :5]
                sp50p  sp50d  sp50e      cpi  real_rate
    date
    1871-01-31   4.44   0.26    0.4  12.4641     5.3200
    1871-02-28   4.50   0.26    0.4  12.8446     5.3233
    1871-03-31   4.61   0.26    0.4  13.0350     5.3267
    1871-04-30   4.74   0.26    0.4  12.5592     5.3300
    1871-05-31   4.86   0.26    0.4  12.2738     5.3333
    1871-06-30   4.82   0.26    0.4  12.0835     5.3367
    1871-07-31   4.73   0.26    0.4  12.0835     5.3400

    .. _ONLINE DATA ROBERT SHILLER:
        http://www.econ.yale.edu/~shiller/data.htm
    """

    xls = 'http://www.econ.yale.edu/~shiller/data/ie_data.xls'
    cols = ['date', 'sp50p', 'sp50d', 'sp50e', 'cpi', 'frac', 'real_rate',
            'real_sp50p', 'real_sp50d', 'real_sp50e', 'cape']
    iedata = pd.read_excel(xls, sheet_name='Data', skiprows=7,
                           skip_footer=1, names=cols).drop('frac', axis=1)
    dt = iedata['date'].astype(str).str.replace('.', '') + '01'
    iedata['date'] = pd.to_datetime(dt, format="%Y%m%d") + offsets.MonthEnd()
    return iedata.set_index('date')


def load_retaildata():
    """Monthly retail trade data from census.gov."""
    # full = 'https://www.census.gov/retail/mrts/www/mrtssales92-present.xls'
    # indiv = 'https://www.census.gov/retail/marts/www/timeseries.html'

    db = {
        'Auto, other Motor Vehicle':
        'https://www.census.gov/retail/marts/www/adv441x0.txt',

        'Building Material and Garden Equipment and Supplies Dealers':
        'https://www.census.gov/retail/marts/www/adv44400.txt',

        'Clothing and Clothing Accessories Stores':
        'https://www.census.gov/retail/marts/www/adv44800.txt',

        'Dept. Stores (ex. leased depts)':
        'https://www.census.gov/retail/marts/www/adv45210.txt',

        'Electronics and Appliance Stores':
        'https://www.census.gov/retail/marts/www/adv44300.txt',

        'Food Services and Drinking Places':
        'https://www.census.gov/retail/marts/www/adv72200.txt',

        'Food and Beverage Stores':
        'https://www.census.gov/retail/marts/www/adv44500.txt',

        'Furniture and Home Furnishings Stores':
        'https://www.census.gov/retail/marts/www/adv44200.txt',

        'Gasoline Stations':
        'https://www.census.gov/retail/marts/www/adv44700.txt',

        'General Merchandise Stores':
        'https://www.census.gov/retail/marts/www/adv45200.txt',

        'Grocery Stores':
        'https://www.census.gov/retail/marts/www/adv44510.txt',

        'Health and Personal Care Stores':
        'https://www.census.gov/retail/marts/www/adv44600.txt',

        'Miscellaneous Store Retailers':
        'https://www.census.gov/retail/marts/www/adv45300.txt',

        'Motor Vehicle and Parts Dealers':
        'https://www.census.gov/retail/marts/www/adv44100.txt',

        'Nonstore Retailers':
        'https://www.census.gov/retail/marts/www/adv45400.txt',

        'Retail and Food Services, total':
        'https://www.census.gov/retail/marts/www/adv44x72.txt',

        'Retail, total':
        'https://www.census.gov/retail/marts/www/adv44000.txt',

        'Sporting Goods, Hobby, Book, and Music Stores':
        'https://www.census.gov/retail/marts/www/adv45100.txt',

        'Total (excl. Motor Vehicle)':
        'https://www.census.gov/retail/marts/www/adv44y72.txt',

        'Retail (excl. Motor Vehicle and Parts Dealers)':
        'https://www.census.gov/retail/marts/www/adv4400a.txt'
        }

    dct = {}
    for key, value in db.items():
        data = pd.read_csv(value, skiprows=5, skip_blank_lines=True,
                           header=None, sep='\s+', index_col=0)
        try:
            cut = data.index.get_loc('SEASONAL')
        except KeyError:
            cut = data.index.get_loc('NO')
        data = data.iloc[:cut]
        data = data.apply(lambda col: pd.to_numeric(col, downcast='float'))
        data = data.stack()
        year = data.index.get_level_values(0)
        month = data.index.get_level_values(1)
        idx = pd.to_datetime({'year': year, 'month': month, 'day': 1}) \
            + offsets.MonthEnd(1)
        data.index = idx
        data.name = key
        dct[key] = data

    sales = pd.DataFrame(dct)
    sales = sales.reindex(pd.date_range(sales.index[0],
                          sales.index[-1], freq='M'))
    # TODO: account for any skipped months; could specify a DateOffset to
    # `freq` param of `pandas.DataFrame.shift`
    yoy = sales.pct_change(periods=12)

    return sales, yoy
