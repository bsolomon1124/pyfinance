"""Financial dataset web scrubbing.

Descriptions
============
load_13f
    Parse SEC 13F XML file to pandas DataFrame.
load_activeshare
    Reconstruct active share database from activeshare.info.
load_factors
    Load risk factor returns.
load_industries
    Load industry portfolio returns from Ken French's website.
load_rates
    Load database of interest rates (CP, corporate, government).
load_rf
    Build a risk-free rate return series using 3-month US T-bill yields.
scrub_activeshare
    Scrub active share data from activeshare.info for a given ticker or id.
load_retaildata
    Load and clean retail trade data from census.gov.
"""

__author__ = 'Brad Solomon <brad.solomon.1124@gmail.com>'

__all__ = [
    'load_factors', 'load_industries', 'load_rates', 'load_shiller',
    'load_activeshare', 'scrub_activeshare', 'load_rf'
    ]

from collections import defaultdict
import itertools
import re
import time
try:
   import cPickle as pickle
except ImportError:
   import pickle

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from pandas import DataFrame, Series, read_csv, read_excel
from pandas.tseries import offsets
from pandas_datareader.data import DataReader as dr
import requests
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import xmltodict

from pyfinance import returns, utils

# Default start date for web-retrieved time series
DSTART = '1950-01'

def load_13f(url=None, CIK=None):
    """Parse SEC 13F XML file to pandas DataFrame.

    Use the raw .xml file rather than formatted table form as url.  Providing
    a CIK will be slower than direct URL.  CIK will find the most recent
    submission.

    Parameters
    ==========
    url : str, default None.  Use raw .xml file rather than formatted table
        Link to .xml file
    CIK : int, default None
        Firm CIK number.  Will find the most recent submission

    Returns
    =======
    df : DataFrame
        holdings snapshot

    Example
    =======
    url = 'https://www.sec.gov/Archives/edgar/data/1040273/000108514617001787/form13fInfoTable.xml'
    res = load_13f(url=url)

    Note
    ====
    URL structure: in 000110380417000040 (example):
        - 0001103804 is the CIK (000 added)
        - 17 is the year
        - 000040 is 'a sequential count of submitted filings from that CIK.
          The count  is usually, but not always, reset to 0 at the start of
          each calendar year.'

    See https://www.sec.gov/edgar/searchedgar/accessing-edgar-data.htm
    """

    if np.count_nonzero([url, CIK]) != 1:
        raise ValueError('One of url or CIK must be specified, but not both.')
    if CIK:
        url = _direc(CIK)

    data = requests.get(url).text
    df = DataFrame(xmltodict.parse(data)['informationTable']['infoTable'])
    df = pd.concat((df, df.votingAuthority.apply(pd.Series),
                    df.shrsOrPrnAmt.apply(pd.Series)), axis=1)
    del df['votingAuthority'], df['shrsOrPrnAmt']
    nums = ['value', 'Sole', 'sshPrnamt']
    df.loc[:, nums] = df.loc[:, nums].apply(pd.to_numeric)
    df.loc[:, 'weight'] = df.value / df.value.sum()

    return df


def _direc(CIK):
    direc = 'https://www.sec.gov/Archives/edgar/data/' + str(CIK)
    r = requests.get(direc)
    soup = BeautifulSoup(r.text, 'html.parser')
    folders = [link.get('href') for link in soup.find_all('a')]
    folders = list(filter(lambda k: '/Archives' in k, folders))

    for folder in folders:
        link = 'https://www.sec.gov' + folder
        r = requests.get(link)
        soup = BeautifulSoup(r.text, 'html.parser')
        link = [link.get('href') for link in soup.find_all('a')
                if 'Form13fInfoTable.xml' in link]
        if link:
            break

    return 'https://www.sec.gov' + link[0]


def load_activeshare():
    """Reconstruct active share database from activeshare.info."""

    # TODO: There are 3560 funds in database as of last run, but this will
    #     be error-prone as number of funds change.  Convert to while loop

    def _get_actsh(start=1, end=3560):
        info = defaultdict(list)
        data = []
        errors = []
        for fundid in range(start, end):
            try:
                fund = scrub_activeshare(fundid=fundid)
                info['names'].append(fund['Name'])
                info['tickers'].append(fund['Ticker'])
                info['minbenchmarks'].append(fund['Min Benchmark'])
                info['sdbenchmarks'].append(fund['SD. Benchmark'])
                data.append(fund['Download'])
            except:
                errors.append(fundid)
        info = pd.DataFrame.from_dict(info)
        data = pd.concat(data, axis=0)

        return info, data, errors

    info, data, _ = _get_actsh() # return errors if needed
    data.index = data.index.to_timestamp(freq='Q', how='end')
    data.replace('', np.nan, inplace=True)
    info.replace('', np.nan, inplace=True)

    data.fillna(value=np.nan, inplace=True)
    info.fillna(value=np.nan, inplace=True)

    merged = (data.reset_index()
              .merge(info, how='left',
                     left_on=['Fund Family: Fund Name', 'Tickers'],
                     right_on=['names', 'tickers'])
              .set_index('Date'))

    data.reset_index(inplace=True)
    merged.reset_index(inplace=True)

    sc = [
        'Russell 2000',
        'Russell 2000 Growth',
        'Russell 2000 Value',
        'S&P SmallCap 600',
        'S&P SmallCap 600 Growth',
        'S&P SmallCap 600 Pure Value',
        'FTSE RAFI US Mid Small 1500',
        'MSCI US Small Cap 1750',
    ]

    mc = [
        'S&P MidCap 400',
        'S&P MidCap 400 Growth',
        'S&P MidCap 400 Value',
        'S&P MidCap 400 Pure Growth',
        'S&P MidCap 400 Pure Value',
        'Russell Mid Cap',
        'Russell Mid Cap Value',
        'Russell Mid Cap Growth',
        'MSCI KLD 400 Social NR',
        ]

    lc = [
        'S&P 100',
        'S&P 500',
        'S&P 500 Growth',
        'S&P 500 Value',
        'S&P 500 Pure Growth',
        'S&P 500 Pure Value',
        'Russell 1000',
        'Russell 1000 Growth',
        'Russell 1000 Value',
        'Russell Top 200',
        'Russell Top 200 Growth',
        'Russell Top 200 Value',
        'DJ Industrial Average',
        'NASDAQ 100',
        'FTSE RAFI US 1000',
        'MSCI US Prime Market Value',
        'MSCI US Prime Market Growth',
        'Calvert Social',
        'FTSE4Good US Benchmark',
        'MSCI US Prime Market 750',
        'FTSE High Dividend Yield',
        'DJ US Select Dividend',
        ]

    ac = [
        'S&P 1500'
        'Russell 3000',
        'NASDAQ Composite',
        'DJ Wilshire 5000',
        'DJ Wilshire 4500',
        'DJ US Total',
        ]

    sc = dict(zip(sc, ['Small Cap'] * len(sc)))
    mc = dict(zip(mc, ['Mid Cap'] * len(mc)))
    lc = dict(zip(lc, ['Large Cap'] * len(lc)))
    ac = dict(zip(ac, ['All Cap'] * len(ac)))

    categories = {**sc, **mc, **lc, **ac}

    cat1 = merged['minbenchmarks'].map(categories)
    cat2 = merged['sdbenchmarks'].map(categories)
    merged['Ctgry'] = np.where(~cat1.isnull(), cat1, cat2) # nan fallback?

    grp1 = merged.groupby(['Date', 'Ctgry'])
    grp2 = merged.groupby(['Date', 'Ctgry'])
    rank1 = grp1['Minimum Active Share (Across Benchmarks)'].rank(pct=True)
    rank2 = grp2['Minimum Active Share (Across Benchmarks)'].rank(pct=True)
    cond1 = merged.minbenchmarks.isnull()
    cond2 = merged.sdbenchmarks.isnull()
    rank = np.where(cond1 & cond2, np.nan,
                    np.where(cond1, rank2, rank1))
    merged['Rank'] = rank

    return merged


@utils.pickle_option
def load_factors(pickle_from=None, pickle_to=None):
    """Load risk factor returns.

    Factors
    =======
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
    ================
    Source      Link
    ------      ----
    French      http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
    Pastor      http://faculty.chicagobooth.edu/lubos.pastor/research/liq_data_1962_2016.txt
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
    facs = ['F-F_Research_Data_5_Factors_2x3',
            'F-F_Momentum_Factor',
            'F-F_ST_Reversal_Factor',
            'F-F_LT_Reversal_Factor'
           ]

    for fac in facs:
        tgt.append(dr(fac, 'famafrench', DSTART)[0])

    # BETA, ACC, VAR, IVAR require some manipulation to compute returns
    # in the dual-sort method of Fama-French
    for i in ['BETA', 'AC', 'VAR', 'RESVAR']:
        ser = dr('25_Portfolios_ME_' + i + '_5x5', 'famafrench', DSTART)[0]
        ser = (ser.iloc[:, [0,5,10,15,20]].mean(axis = 1)
              - ser.iloc[:, [4,9,14,19,24]].mean(axis = 1))
        ser = ser.rename(i)
        tgt.append(ser)

    # E/P, CF/P, D/P (univariate sorts, quintile spreads)
    for i in ['E-P', 'CF-P', 'D-P']:
        ser = dr('Portfolios_Formed_on_' + i, 'famafrench', DSTART)[0]
        ser = ser.loc[:, 'Hi 20'] - ser.loc[:, 'Lo 20']
        ser = ser.rename(i)
        tgt.append(ser)

    tgt = [df.to_timestamp(how='end') for df in tgt]

    # BAB, QMJ, HMLD
    # TODO: performance is poor here, runtime is eaten up by these 3
    links = {'BAB' : 'http://bit.ly/2hWyaG8',
             'QMJ' : 'http://bit.ly/2hUBSgF',
             'HMLD' : 'http://bit.ly/2hdVb7G'}
    for key, value in links.items():
        ser = read_excel(value, header=18, index_col=0)['USA'] * 100
        ser = ser.rename(key)
        tgt.append(ser)

    # Lookback straddles
    link = 'http://faculty.fuqua.duke.edu/~dah7/DataLibrary/TF-Fac.xls'
    straddles = read_excel(link, header=14, index_col=0)
    straddles.index = (pd.DatetimeIndex(straddles.index.astype(str) + '01')
                    + offsets.MonthEnd(1))
    straddles = straddles * 100.
    tgt.append(straddles)

    # LIQ
    link = 'http://bit.ly/2pn2oBK'
    liq = read_csv(link, skiprows=14, delim_whitespace=True, header=None,
                   usecols=[0, 3], index_col=0, names=['date', 'LIQ'])
    liq.index = (pd.DatetimeIndex(liq.index.astype(str) + '01')
                    + offsets.MonthEnd(1))
    liq = liq.replace(-99, np.nan) * 100.
    tgt.append(liq)

    # USD, HY
    fred = dr(['DTWEXB', 'BAMLH0A0HYM2'], 'fred', DSTART) # daily default
    fred = (fred.asfreq('D', method='ffill')
            .fillna(method='ffill')
            .asfreq('M'))
    fred.loc[:, 'DTWEXB'] = fred['DTWEXB'].pct_change() * 100.
    fred.loc[:, 'BAMLH0A0HYM2'] = fred['BAMLH0A0HYM2'].diff()
    tgt.append(fred)

    # PUT, BXM, RXM (CBOE options strategy indices)
    link1 = 'http://www.cboe.com/micro/put/put_86-06.xls'
    link2 = 'http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/putdailyprice.csv'

    put1 = (read_excel(link1, index_col=0, skiprows=6, header=None)
               .rename_axis('DATE'))
    put2 = read_csv(link2, index_col=0, parse_dates=True, skiprows=7,
               header=None).rename_axis('DATE')
    put = (pd.concat((put1, put2))
           .rename(columns={1 : 'PUT'})
           .iloc[:, 0]
           .asfreq('D', method='ffill')
           .fillna(method='ffill')
           .asfreq('M')
           .pct_change() * 100.)
    tgt.append(put)

    link1 = 'http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/bxmarchive.csv'
    link2 = 'http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/bxmcurrent.csv'

    bxm1 = read_csv(link1, index_col=0, parse_dates=True, skiprows=5,
               header=None).rename_axis('DATE')
    bxm2 = read_csv(link2, index_col=0, parse_dates=True, skiprows=4,
               header=None).rename_axis('DATE')
    bxm = (pd.concat((bxm1, bxm2))
              .rename(columns={1 : 'BXM'})
              .iloc[:, 0]
              .asfreq('D', method='ffill')
              .fillna(method='ffill')
              .asfreq('M')
              .pct_change() * 100.)
    tgt.append(bxm)

    link = 'http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/rxm_historical.csv'
    rxm = (read_csv(link, index_col=0, parse_dates=True, skiprows=2,
              header=None)
              .rename(columns={1 : 'RXM'})
              .rename_axis('DATE')
              .iloc[:, 0]
              .asfreq('D', method='ffill')
              .fillna(method='ffill')
              .asfreq('M')
              .pct_change() * 100.)
    tgt.append(rxm)

    # Clean up data retrieved above
    # ------------------------------------------------------------------------

    factors = pd.concat(tgt, axis=1).round(2)
    newnames = {'Mkt-RF' : 'MKT',
               'Mom   ' : 'UMD',
               'ST_Rev' : 'STR',
               'LT_Rev' : 'LTR',
               'RESVAR' : 'IVAR',
               'AC' : 'ACC',
               'PTFSBD' : 'BDLB',
               'PTFSFX' : 'FXLB',
               'PTFSCOM' : 'CMLB',
               'PTFSIR' : 'IRLB',
               'PTFSSTK' : 'STLB',
               'DTWEXB' : 'USD',
               'BAMLH0A0HYM2' : 'HY'
              }
    factors.rename(columns=newnames, inplace=True)

    # Get last valid RF date; returns will be constrained to this date
    factors = factors[:factors['RF'].last_valid_index()]

    # Subtract RF for long-only factors
    subtract = ['HY', 'PUT', 'BXM', 'RXM']

    for i in subtract:
        factors.loc[:, i] = factors[i] - factors['RF']

    return factors


@utils.pickle_option
def load_industries(pickle_from=None, pickle_to=None):
    """Load industry portfolio returns from Ken French's website.

    Parameters
    ==========
    ports : int or list
        number of portfolios; choose from [5, 10, 12, 17, 30, 38, 48]
    start : str, default 1967
        start date in time series of returns
    end : date, default date.today()
        end date in time series of returns
    form : str, {'num', 'dec'}, default 'num'
        display format of factor returns
    round : int, default 3
        decimal places to round data

    Returns
    =======
    industries : dict of DataFrames
        each key is a portfolio group

    See also
    =====
    # from pandas_datareader.famafrench import get_available_datasets
    # get_available_datasets()
    """

    full = [5, 10, 12, 17, 30, 38, 48]
    rets = []
    for port in [str(port) + '_Industry_Portfolios' for port in full]:
        ret = dr(port, 'famafrench', start=DSTART)[0]
        rets.append(ret.to_timestamp(how='end'))
    industries = dict(zip(full, rets))

    return industries


@utils.pickle_option
def load_rates(freq='D', pickle_from=None, pickle_to=None):
    """Load interest rates from https://fred.stlouisfed.org/.

    Parameters
    ==========
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
    ===============
    Board of Governors of the Federal Reserve System
    H.15 Selected Interest Rates
    https://www.federalreserve.gov/releases/h15/
    """

    months = [1, 3, 6]
    years = [1, 2, 3, 5, 7, 10, 20, 30]

    # Nested dictionaries of symbols from fred.stlouisfed.org
    nom = {'D' :
           ['DGS%sMO' % m for m in months] + ['DGS%s' % y for y in years],
           'W' :
           ['WGS%sMO' % m for m in months] + ['WGS%sYR' % y for y in years],
           'M' :
           ['GS%sM' % m for m in months] + ['GS%s' % y for y in years]
          }

    tips = {'D' : ['DFII%s' % y for y in years[3:7]],
            'W' : ['WFII%s' % y for y in years[3:7]],
            'M' : ['FII%s' % y for y in years[3:7]]
           }

    fcp = {'D' : ['DCPF1M', 'DCPF2M', 'DCPF3M'],
           'W' : ['WCPF1M', 'WCPF2M', 'WCPF3M'],
           'M' : ['CPF1M', 'CPF2M', 'CPF3M']
          }

    nfcp = {'D' : ['DCPN30', 'DCPN2M', 'DCPN3M'],
            'W' : ['WCPN1M', 'WCPN2M', 'WCPN3M'],
            'M' : ['CPN1M', 'CPN2M', 'CPN3M']
           }

    short = {'D' : ['DFF', 'DPRIME', 'DPCREDIT'],
             'W' : ['FF', 'WPRIME', 'WPCREDIT'],
             'M' : ['FEDFUNDS', 'MPRIME', 'MPCREDIT']
            }

    rates = list(itertools.chain.from_iterable([d[freq] for d in
                 [nom, tips, fcp, nfcp, short]]))
    rates = dr(rates, 'fred', start=DSTART)

    l1 = ['Nominal'] * 11 + ['TIPS'] * 4 + ['Fncl CP'] * 3 \
       + ['Non-Fncl CP'] * 3 + ['Short Rates'] * 3

    l2 = ['%sm' % m for m in months] + ['%sy' % y for y in years] \
       + ['%sy'% y for y in years[3:7]] \
       + 2 * ['%sm' % m for m in range(1, 4)] \
       + ['Fed Funds', 'Prime Rate', 'Primary Credit']

    rates.columns = pd.MultiIndex.from_arrays([l1, l2])

    return rates


def load_rf(freq='M', pickle_from=None, pickle_to=None, ):
    """Build a risk-free rate return series using 3-month US T-bill yields.

    The 3-Month Treasury Bill: Secondary Market Rate from the Federal Reserve
    (a yield) is convert to a total return.  See 'Methodology' for details.

    The time series should closely mimic returns of the BofA Merrill Lynch US
    Treasury Bill (3M) (Local Total Return) index.

    Parameters
    ==========
    reload : bool, default False
        If False, use pickled data.  If True, reload from source
    freq : str, sequence, or set
        If a single-character string, return a single-column DataFrame with
        index frequency corresponding to `freq`.  If a sequence or set, return
        a dict of DataFrames with the keys corresponding to `freq`(s)

    Methodology
    ===========
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
    ========
    FRED: 3-Month Treasury Bill: Secondary Market Rate (DTB3)
      https://fred.stlouisfed.org/series/DTB3
    McGraw-Hill/Irwin, Interest Rates, 2008.
      https://people.ucsc.edu/~lbaum/econ80h/LS-Chap009.pdf
    Morningstar, Return Calculation of U.S. Treasury Constant Maturity Indices,
      September 2008.
    """

    # Validate `freq` param
    freqs = list('DWMQA')
    freq = freq.upper() if freq.islower() else freq
    if freq not in freqs:
        raise ValueError('`freq` must be either a single element or subset'
                         ' from %s, case-insensitive' % freqs)

    # Load daily 3-Month Treasury Bill: Secondary Market Rate
    # Note that this is on discount basis and will be converted to BEY
    # Periodicity is daily
    rates = dr('DTB3', 'fred', DSTART) * 0.01
    rates = (rates.asfreq('D', method='ffill').fillna(method='ffill')
                  .squeeze())

    # Algebra doesn't 'work' on DateOffsets, don't simplify here!
    trigger = rates.index.is_month_end
    dtm_old = rates.index + offsets.MonthEnd(-1) + offsets.MonthEnd(3) \
            - rates.index
    dtm_new = rates.index.where(trigger, rates.index +
                                offsets.MonthEnd(-1)) \
            + offsets.MonthEnd(3) - rates.index

    # This does 2 things in one step:
    # (1) convert discount yield to BEY
    # (2) get the price at that BEY and days to maturity
    # The two equations are simplified
    # See https://people.ucsc.edu/~lbaum/econ80h/LS-Chap009.pdf
    p_old = (100 / 360) * (360 - rates * dtm_old.days)
    p_new = (100 / 360) * (360 - rates * dtm_new.days)

    res = p_old.pct_change().where(trigger, p_new.pct_change())
    res = returns.prep(res, in_format='dec', name='RF', freq='D')

    if freq != 'D':
        res = returns.prep(res.rollup(out_freq=freq), in_format='dec', freq=freq)

    return res


@utils.pickle_option
def load_shiller(pickle_from=None, pickle_to=None):
    """Load data from Robert Shiller's website.

    Description: http://www.econ.yale.edu/~shiller/data.htm

    Examples
    ========
    shiller = load_shiller()
    shiller = shiller[shiller.index.month % 3 == 0]
    """

    link = 'http://www.econ.yale.edu/~shiller/data/ie_data.xls'
    iedata = (read_excel(link, sheetname='Data', skiprows = range(0,7))
        .loc[:, :'CAPE']
        .dropna(subset=['Date'])
        .drop('Fraction', axis=1))
    cols = [
        'date',
        'sp50p',
        'sp50d',
        'sp50e',
        'cpi',
        'real_rate',
        'real_sp50p',
        'real_sp50d',
        'real_sp50e',
        'cape'
        ]
    iedata.columns = cols
    iedata.loc[:, 'date'] = (pd.to_datetime((iedata.date.astype(str)
        .str.replace('.', '') + '01'), format="%Y%m%d")
        + offsets.MonthEnd(1))
    iedata.set_index('date', inplace=True)

    return iedata


def scrub_activeshare(ticker=None, fundid=None):
    """Scrub active share data from activeshare.info for a given ticker or id.

    Specify *one* of (`ticker`, `fundid`).  Specifying fundid will be exceptionally
    faster, because using `ticker` routes through Selenium and remotely
    navigates through the search bar with Selenium.

    Parameters
    ==========
    ticker : str
        Fund ticker.  Accepts non-primary share classes
    fundid : int
        Fund id number.  Unclear mapping to fund name; may be proprietary

    Returns
    =======
    result : dict
        Keys: ['Download', 'Ticker', 'Name', 'Benchmarks']

    Dependencies
    ============
    You will need phantomjs installed locally in your PYTHONPATH, see
      http://phantomjs.org/download.html
    """

    if ticker is not None and fundid is not None:
        raise ValueError('Specify one of {ticker, fundid}, but not both.')

    if ticker is None and fundid is None:
        raise ValueError('Must specify one of {ticker, fundid}.')

    home = 'https://activeshare.info/'
    export = '/api/activeshare/export?fundIds='
    key = 'Active Share with respect to the'

    if fundid is not None:
        exportlink = home[:-1] + export + str(fundid)
        df = read_excel(exportlink, index_col='Date') # dates not parsed
        df.index = pd.PeriodIndex(df.index, freq='Q')

        # Each fund goes by a hyphenated full name, i.e.
        # columbia-funds-series-trust-i-columbia-select-large-cap-growth-fund
        # The replace chain below takes fund name as specified in the page's
        # csv and attempts to convert to the hyphenated version using a number
        # of replacement conventions, for example:
        # ARK ETF Trust: ARK Web x.0 ETF --> ark-etf-trust-ark-web-x0-etf
        fundname = (df.iloc[0, 0].lower()
                    .replace(' - ', '-')
                    .replace('  ', ' ')
                    .replace(' ', '-')
                    .replace('--', '-')
                    .replace(':', '')
                    .replace('/', '-')
                    .replace('(', '')
                    .replace(')', '')
                    .replace(',', '')
                    .replace('\'', '')
                    .replace('u.s.', 'us')
                    .replace('.', '')
                    .replace(' :', ':')
                    .replace('&-', '')
                    .replace('&', '-')) # a work in progress...
        r = requests.get(home + 'fund/' + fundname)
        soup = BeautifulSoup(r.text, 'html.parser')

    else:
        # TODO: driver = webdriver.PhantomJS() takes ~65% of runtime (~3s)
        driver = webdriver.PhantomJS()
        driver.get(home)
        sbox = driver.find_element_by_id('searchBox')
        sbox.send_keys(ticker)
        time.sleep(1)
        sbox = driver.find_element_by_id('searchBox')
        sbox.send_keys(Keys.ENTER)

        # OLD:
        # if direct_download:
            # (driver
             # .find_element_by_xpath('//a[contains(@href, "%s")]' % export)
             # .click())

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        ref = (soup.find('a', href=lambda href:
               href and href.startswith(export))['href'])
        exportlink = home[:-1] + ref
        df = read_excel(exportlink, index_col='Date') # dates not parsed
        df.index = pd.PeriodIndex(df.index, freq='Q')

    strs = [l.get_text() for l in
            soup.find_all('p', text=re.compile(key), limit=2)]
    strs = [s.replace(';', '') for s in strs]

    end1 = 'benchmark'
    end2 = ' (the self-declared benchmark)'

    # <250: a hack for catching instances describing actual benchmark
    # rather than general footnotes that might contain the same snippet

    if len(strs[0]) < 250:
        try:
            minbench = strs[0].replace(key, '').replace(end1, '').strip()
        except:
            minbench = None
    else:
        minbench = None

    if len(strs[1]) < 250:
        try:
            sdbench = strs[1].replace(key, '').replace(end2, '').strip()
        except:
            sdbench = None
    else:
        sdbench = None

    result = {'Download' : df,
              'Ticker' : df.Tickers[0],
              'Name' : df['Fund Family: Fund Name'][0],
              'Min Benchmark' : minbench,
              'SD. Benchmark' : sdbench
             }

    return result


@utils.pickle_option
def load_retaildata(pickle_from=None, pickle_to=None):
    """Monthly retail trade data from census.gov."""
    # full = 'https://www.census.gov/retail/mrts/www/mrtssales92-present.xls'
    # indiv = 'https://www.census.gov/retail/marts/www/timeseries.html'

    db = {'Auto, other Motor Vehicle':
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
        data = read_csv(value, skiprows=5, skip_blank_lines=True, header=None,
                        sep='\s+', index_col=0)
        try:
            cut = data.index.get_loc('SEASONAL')
        except KeyError:
            cut = data.index.get_loc('NO')
        data = data.iloc[:cut]
        data = data.apply(lambda col: pd.to_numeric(col, downcast='float'))
        data = data.stack()
        year = data.index.get_level_values(0)
        month = data.index.get_level_values(1)
        idx = pd.to_datetime({'year' : year, 'month' : month, 'day' : 1}) \
            + offsets.MonthEnd(1)
        data.index = idx
        data.name = key
        dct[key] = data

    sales = DataFrame(dct)
    sales = sales.reindex(pd.date_range(sales.index[0],
                          sales.index[-1], freq='M'))
    # TODO: account for any skipped months; could specify a DateOffset to
    # `freq` param of `pandas.DataFrame.shift`
    yoy = sales.pct_change(periods=12)

    return sales, yoy