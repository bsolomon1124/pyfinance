"""Statistical analysis of time series security returns data.

Methods include 'benchmark-agnostic' statistics such as max drawdown, as
well as benchmark-relative statistics such as correlation; these stats
will always have a `benchmark` parameter.  Note that the methods currently
only support *single-benchmark* cases.  pandas' DataFrame/Series are inputs.

Using `prep`
============
`r` and `benchmark` should first be 'prepped' by calling `prep`.  This step
mimics some of the features of class instantiation or subclassing without the
additional baggage of actually using a class-based implementation.

The purposes of `prep` are to:
- Give a `.freq` attribute to an index without one (required for financial time
  series in most cases)
- Convert 'numeric' form (i.e. 5.43%->5.43)  to 'decimal' form (0.0543)
- Convert a Series to a single-column DataFrame

This ensures compatability with the functions defined herein.  For more, see
documentation for `prep`.

Author's note
=============
You might notice that many functions here take the same default parameters.
I made the deliberate decision not to use a class-based implementation here.
That approach would have consisted of using the core DataFrame or Series
of time series returns as a class attribute and building on top of that.

This gets to be a huge drag on memory with some functions here, which
would need to create sub-instances (class withina a class), each of which
would then need to carry along with it possibly unncessary attributes.  So,
the current setup is designed to mimic the feel of using a `Returns` class
but in a more flexible way that does not waste space.

Standard parameters
===================
r : Series or DataFrame
    The core time series of periodic returns.  May be a single security
    (column vector) or multiple securities
benchmark : Series or single-column DataFrame
    The time series of periodic returns for the benchmark.  Must be a
    single security
window : int or offset, default None
    Size of the moving window. This is the number of observations used for
    calculating the statistic. Each window will be a fixed size.
    If it is an offset then this will be the time period of each window.
    Each window will be a variable size based on the observations included
    in the time-period. This is only valid for datetimelike indexes.  If
    `window` is None (default), the statistic is calculated over the full
    sample
anlz : bool, default True
    If True, annualize the returned values by applying an annualization
    factor.  If False, the returned values are on a per-period basis.  The
    annualization factors are as follows (with `freq` as keys):
    {'D' : 252., 'W' : 52., 'M' : 12., 'Q' : 4., 'A' : 1.}
method : str
    One of ('geo', 'geometric', 'arith', 'arithmetic, 'cum', 'cumulative').
    Specify use of either arithmetic or geometric means and differences.
    'Cumulative' will not apply to all functions in this module
ddof : str or int, default 1
    The degrees of freedom {'sample' (1), 'population' or 'pop' (0)}
log : bool, default False
    If False, use geometric methodology; if True, use continuous
    compounding with natural logarithm
thresh : float or 'mean', default 0.0
    The cutoff filter on which to test and filter `benchmark` returns.
    If 'mean', the arithmetic mean of `benchmark` will be used.  In some
    statistics this is analogous to the Minimum Acceptable Return (MAR)
base : float, default 1.0
    Within return indices, the initial (anchoring) value
rf : float, default 0.0; or str, one of ('mean')
    The risk-free rate to utilize.  See the `load_rf` function within
    `wl.datasets`.
    - If a float is given, this is assumed to be the annualized risk-free
      rate and the `load_rf` function is not used
    - If a string is given, this is passed to the `source` keyword of
      `load_rf`.  Valid sources are ('fred', 'factset'/'fs').
reload : bool, default False
    Passed to `load_rf` is `rf` is a string (source).  Specifies whether to
    use pickled data or refresh

Examples
========
import wl.returns as rt

# Bring in some returns data
rets = read_csv('input/data.csv', index_col=0, parse_dates=True)
print(rets.head(3))
              ELGAX       EEM      SPY
2007-10-31  8.97119  11.15985  1.59068
2007-11-30 -3.77643  -7.08429 -4.18066
2007-12-31 -2.19780   0.35906 -0.69376

# The returns are in 'numeral' form (8.97% -> 8.97) and don't have a recognized
# index frequency.

print(rets.index.freq)
None

# Using `prep` handles all this in one step

bm = rt.prep(rets.SPY)
rets = rt.prep(rets[['ELGAX', 'EEM']])

# Now, all funcs in the module are callable on these two Dataframes
# Build a return index
print(rets.return_index(base=100, include_start=True))
                ELGAX        EEM
2007-09-30  100.00000  100.00000
2007-10-31  108.97119  111.15985
2007-11-30  104.85597  103.28496
2007-12-31  102.55144  103.65582
2008-01-31   94.32099   90.75058
              ...        ...
2017-02-28  197.06135   99.94250
2017-03-31  202.06024  102.49045
2017-04-30  207.19067  104.75543
2017-05-31  212.45266  107.87434
2017-06-30  216.89175  109.03349

# Many functions are fairly plain-vanilla and serve just to build some
# additional convenience around existing pandas/NumPy functions

print(rets.stdev(anlz=True, ddof=0))
ELGAX    0.20057
EEM      0.23271
dtype: float64

"""

# TODO: the main shortcoming of this module is that it only supports a
#     1-benchmark case.  (Using >1 securities with >1 benchmarks would require
#     taking a cartesian product of the two sets.)  It's doable (the result 
#     could be a DataFrame with MultiIndex columns), but the module wasn't 
#     designed as such to start.

__author__ = 'Brad Solomon <brad.solomon.1124@gmail.com>'

__all__ = [
    'return_relatives', 'return_index', 'cumulative_return',
    'cumulative_returns', 'drawdown_index', 'stdev', 'correl', 'cond_correl',
    'covar', 'jarque_bera', 'bias_ratio', 'sharpe_ratio', 'msquared',
    'rollup', 'max_drawdown', 'calmar_ratio', 'ulcer_index',
    'semi_stdev', 'sortino_ratio', 'rolling_returns', 'min_max_return',
    'pct_negative', 'pct_positive', 'pct_pos_neg', 'excess_returns',
    'excess_drawdown_index', 'tracking_error', 'batting_avg',
    'downmarket_filter', 'upmarket_filter', 'upside_capture',
    'downside_capture', 'capture_ratio', 'insert_start', 'beta',
    'alpha', 'tstat_alpha', 'tstat_beta', 'rsq', 'rsq_adj', 'prep',
    'factor_loadings', 'Portfolio', 'cross_corr'
    ]

from collections import OrderedDict
from functools import lru_cache
import itertools
import warnings

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from pandas.core.base import PandasObject
from pandas.tseries import offsets
import scipy.optimize as sco
import scipy.stats as scs

from pyfinance import datasets, general, ols, utils

_defaultdocs = {

    'r' :
    """
    r : Series or DataFrame
        The core time series of periodic returns.  May be a single security
        (column vector) or multiple securities""",

    'benchmark' :
    """
    benchmark : Series or single-column DataFrame
        The time series of periodic returns for the benchmark.  Must be a
        single security""",

    'window' :
    """
    window : int or offset, default None
        Size of the moving window. This is the number of observations used for
        calculating the statistic. Each window will be a fixed size.
        If it is an offset then this will be the time period of each window.
        Each window will be a variable size based on the observations included
        in the time-period. This is only valid for datetimelike indexes.  If
        `window` is None (default), the statistic is calculated over the full
        sample""",

    'anlz' :
    """
    anlz : bool, default True
        If True, annualize the returned values by applying an annualization
        factor.  If False, the returned values are on a per-period basis.  The
        annualization factors are as follows (with `freq` as keys):
        {'D' : 252., 'W' : 52., 'M' : 12., 'Q' : 4., 'A' : 1.}""",

    'method' :
    """
    method : str
        One of ('geo', 'geometric', 'arith', 'arithmetic, 'cum', 'cumulative').
        Specify use of either arithmetic or geometric means and differences.
        'Cumulative' will not apply to all functions in this module""",

    'ddof' :
    """
    ddof : str or int, default 1
        The degrees of freedom {'sample' (1), 'population' or 'pop' (0)}""",

    'log' :
    """
    log : bool, default False
        If False, use geometric methodology; if True, use continuous
        compounding with natural logarithm""",

    'thresh' :
    """
    thresh : float or 'mean', default 0.0
        The cutoff filter on which to test and filter `benchmark` returns.
        If 'mean', the arithmetic mean of `benchmark` will be used.  In some
        statistics this is analogous to the Minimum Acceptable Return (MAR)""",

    'base' :
    """
    base : float, default 1.0
        Within return indices, the initial (anchoring) value""",

    'rf' :
    """
    rf : float, default 0.0; or str, one of ('mean')
        The risk-free rate to utilize.  See the `load_rf` function within
        `wl.datasets`.
        - If a float is given, this is assumed to be the annualized risk-free
          rate and the `load_rf` function is not used
        - If a string is given, this is passed to the `source` keyword of
          `load_rf`.  Valid sources are ('fred', 'factset'/'fs').""",

    'reload' :
    """
    reload : bool, default False
        Passed to `load_rf` is `rf` is a string (source).  Specifies whether to
        use pickled data or refresh"""
    }


# `prep`: uses `.pipe` to chain together several convenience functions
# -----------------------------------------------------------------------------

def prep(r, freq=None, name=None, in_format='num'):
    """Prep raw returns to create compatability with returns functions.

    This convenience function allows, for instance, specification of an
    index frequency in one parameter for indices lacking a frequency.

    The original object `r` is not modified.  (Each chained method returns
    a copy.)

    Internal function calls
    =======================
    Several funcs are called using `.pipe`:
    - `add_freq` : give a `.freq` attribute to an index without one
    - `check_format`: safety check that input format is not misspecified
    - `num_to_dec`: if 'numeric' form (i.e. 5.43%->5.43), convert to 0.0543
    - `series_to_frame` : if Series, convert to DataFrame, else do nothing

    Parameters
    ==========
    r : Series or DataFrame
        The core time series of periodic returns.  May be a single security
        (column vector) or multiple securities
    freq : 'str', default None
        The frequency of `r`.  Precedence:
        - If not None, use the specified `freq` str
        - If None and the index of `r` has a nonnull `freq.freqstr` attribute,
          use this
        - If None and the `freq` of the index is none, attempt to infer a
          frequency with pd.infer_freq
        - Exception raised if the above 3 options are exhausted
    name : str, default None
        For cases where a Series is converted to a single-column DataFrame,
        `name` specifies the resulting column name; it is passed to
        `pd.Series.to_frame`
    in_format : str, one of ('numeral', 'numeric', 'num', 'decimal', 'dec')
        Converts percentage figures from numeral form to decimal form.  Given a
        percent p=5.42%, its:
        - numeral form is 5.43
        - decimal form is 0.0543
    """

    return (r.pipe(add_freq, freq=freq)
             .pipe(check_format, in_format=in_format)
             .pipe(num_to_dec, in_format=in_format)
             .pipe(series_to_frame, name=name)
           )


# Funcs to be called by `pipe` for 'prepping' return streams to uniform format
# -----------------------------------------------------------------------------

def add_freq(r, freq=None):
    """Add a frequency attribute to r.index, through inference or directly."""
    r = r.copy()
    if freq is None:
        if r.index.freq is None:
            freq = pd.infer_freq(r.index)
        else:
            return r
    r.index.freq = pd.tseries.frequencies.to_offset(freq)
    if r.index.freq is None:
        raise AttributeError('no discernible frequency found to `r`.  Specify'
                             ' a frequency string with `freq`.')
    return r


def series_to_frame(r, name=None):
    """If `r` is a Series, convert it to a 1-col DataFrame, otherwise pass."""
    if isinstance(r, Series):
        return r.to_frame(name=name)
    elif isinstance(r, DataFrame):
        return r.copy()
    else:
        raise TypeError('`r` must be one of (`Series`, `DataFrame`)')


def num_to_dec(r, in_format='num'):
    """Convert numeral format to decimal format for percentage values."""
    if in_format == 'num':
        return r * 0.01
    elif in_format == 'dec':
        return r.copy()
    else:
        raise ValueError("`in_format` must be one of ('num', 'dec')")


def check_format(r, in_format='num'):
    """Logic check on proper specification of in_format."""
    freq = r.index.freq.freqstr
    thresh = 1.0
    avg = r.values.mean() # use .values to collapse axis
    std = r.values.std()

    if in_format == 'num' and std * np.sqrt(utils.convertfreq(freq)) < thresh:
        warnings.warn('Returns show anlzd stdev. less than than %.0f%%,'
                      ' in_format may be misspecified.' % (thresh),
                      RuntimeWarning)
    if in_format == 'dec' and (1+avg) ** utils.convertfreq(freq) - 1 > thresh:
        warnings.warn('Returns annualize to greater than %.0f%%,'
                      ' in_format may be misspecified.' % (thresh * 100),
                      RuntimeWarning)
    return r.copy()


# Handful of elementary functions related to compounding.
#     NumPy functions are used wherever possible for compatability with both
#     numpy.ndarrays and pandas Series/DataFrames.
# ----------------------------------------------------------------------------

@utils.appender(_defaultdocs)
def return_relatives(r, log=False):
    """Return-relatives, the term (1.+r).  If `log` is True, then ln(1+r)."""
    res = np.add(r, 1.)
    if log:
        res = np.log(res)
        # or: numpy.log1p
    return res


def return_index(r, log=False, base=1.0, include_start=False):
    """An index of cumulative total returns, expressed as index values.

    Parameters
    ==========
    r : Series or DataFrame
        The core time series of periodic returns.  May be a single security
        (column vector) or multiple securities
    log : bool, default False
        If False, use geometric methodology; if True, use continuous
        compounding with natural logarithm
    base : float, default 1.0
        Within return indices, the initial (anchoring) value
    include_start : bool, default False
        If True, call `insert_start` and add a 0-position row equal to `base`
        to the top of the object
    """

    if log:
        res = np.add(np.cumsum(return_relatives(r, log=log)), 1.)
        res = np.multiply(res, base)
    else:
        res = np.multiply(np.cumprod(return_relatives(r, log=log)), base)
    if include_start:
        res = insert_start(res, base=base)
    return res


@utils.appender(_defaultdocs)
def cumulative_return(r, anlz=True, method='arithmetic', log=False):
    """Total return over the aggregated input periods; a scalar.

    Options
    =======

        anlz        method                  result
        ----        ------                  ------
        False       ('cum', 'cumulative')   Un-anlzd cumulative return
        False       ('geo', 'geometric')    Un-anlzd geometric mean return
        False       ('arith', 'arithmetic') Un-anlzd arithmetic mean return
        True        ('geo', 'geometric')    Anlzd return (using geometric mean)
        True        ('arith', 'arithmetic') Anlzd return (using arith. mean)
        True        ('cum', 'cumulative')   n/a
    """

    # TODO: this func needs cleaned up conceptually.  Does it really make sense
    #     to multiply geometric returns?  (not log-relative-geometric).  Some
    #     param combinations should be explicitly forbidden.  log=True should 
    #     probably necessitate method='arithmetic'.

    if log:
        res = np.add(np.sum(return_relatives(r, log=log)), 1.)
    else:
        res = np.prod(return_relatives(r, log=log))

    if anlz:
        freq = r.index.freq.freqstr
        yrs = 1. / ( r.count() / utils.convertfreq(freq) )
        if method in ['geo', 'geometric']:
            res = res ** yrs
        elif method in ['arith', 'arithmetic']:
            res = np.add(np.subtract(res, 1.) * yrs, 1.)
        else:
            raise ValueError("Annualization not possible in tandem with"
                             " cumulation.  Specify one of `anlz`=False or"
                             " `method` != 'cumulative'")

    elif method in ['geo', 'geometric']:
        per = 1. / r.count()
        res = res ** per
    elif method in ['arith', 'arithmetic']:
        per = 1. / r.count()
        res = res * per

    return np.subtract(res, 1.)


@utils.appender(_defaultdocs)
def cumulative_returns(r, log=False):
    """Time series: a `cumulative_return` at each index location."""
    return np.subtract(return_index(r, log=log), 1.)


@utils.appender(_defaultdocs)
def drawdown_index(r, log=False):
    """Generate an index of drawdowns from cumulative high water marks."""
    ri = return_index(r, log=log)
    cummax = np.maximum.accumulate
    return np.subtract(np.divide(ri, cummax(ri)), 1.)


# Variance, standard deviation, correlation, covariance, and other stats
#     derived primarily from moments of distribution.
# ----------------------------------------------------------------------------

@utils.appender(_defaultdocs)
def stdev(r, anlz=True, ddof=1):
    """Convenience function from `pd.std` with the option to annualize."""
    stdev = r.std(ddof=ddof)
    if anlz:
        freq = r.index.freq.freqstr
        stdev *= np.sqrt(utils.convertfreq(freq))
    return stdev


@utils.appender(_defaultdocs)
def correl(r, benchmark, window=None):
    """Like `pd.corrwith`, but ignore column names and allow rolling."""
    if window is not None:
        res = pd.concat((r, benchmark), axis=1).rolling(window=window).corr()
        res = res.loc[(slice(None), r.columns), benchmark.columns[0]].unstack()
    else:
        res = r.corrwith(benchmark.iloc[:, 0])
    return res


@utils.appender(_defaultdocs)
def cond_correl(r, benchmark, ddof=1):
    """Conditional correlation of `r` to `benchmark`.

    Similar to standard quantiling, the bucketing thresholds are `benchmark`'s
    standard deviations from its mean.  For instance, one correlation figure
    would be the corr(`r`, `benchmark`) | `benchmark` <= mu - 2 * sigma.
    """

    # TODO: rolling?  Could be either the average of rolling periods
    #     (preserving the underlying periodicity) or first rolled up then
    #     matrix calc'd.

    sigma = stdev(benchmark, anlz=False, ddof=ddof)[0]
    mu = benchmark.mean()[0] # TODO: geometric mean?

    breakpoints = mu + np.array([-2., -1., 1., 2.]) * sigma
    breakpoints = np.pad(breakpoints, 1, mode='constant',
                         constant_values=(-np.inf, np.inf))
    labels = ['[-inf,-2]', '(-2,-1]', '(-1,1]', '(1,2]', '(2,inf]']
    groups = pd.cut(benchmark.iloc[:, 0], bins=breakpoints, labels=labels)

    # TODO: kludgy
    full = r.join(benchmark)
    corrs = (full.groupby(groups.values)
                 .corr()
                 .loc[(slice(None), r.columns), benchmark.columns[0]]
                 .unstack()
                 .T)

    return corrs

@utils.appender(_defaultdocs)
def cross_corr(r, window=None):
    """Average pairwise cross-correlation.

    If window is None, returns a scalar.  Otherwise, returns a Series.
    """

    # TODO: Drop NaN ops (front and back side of function)
    if isinstance(r, np.ndarray):
        r = DataFrame(r)
    if window:
        r = r.rolling(window=window)
    corr = r.corr()
    def _cc(mat):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            return np.nanmean(mat.values[np.triu_indices_from(mat.values, 1)])
    if window:
        res = corr.groupby(corr.index.get_level_values(0)).apply(_cc)
    else:
        res = _cc(corr)

    return res

@utils.appender(_defaultdocs)
def covar(r, benchmark, window=None, ddof=1):
    """Covariance between each security in `r` & `benchmark`.

    Ignores column names.
    """

    if window is not None:
        res = pd.concat((r, benchmark), axis=1).rolling(window=window).cov()
        res = res.loc[(slice(None), r.columns), benchmark.columns[0]].unstack()
    else:
        res = correl(r, benchmark=benchmark) \
            * stdev(r, ddof=ddof) * stdev(benchmark, ddof=ddof)[0]
    return res


def jarque_bera(r, normal_kurtosis=0.0):
    """Perform the Jarque-Bera goodness of fit test on each return series.

    Tests whether data has skewness & kurtosis matching normal distribution.

    Warning: this test only works well for a large enough number of data
    samples (>2000) as the test statistic asymptotically has a Chi-squared
    distribution with 2 degrees of freedom.

    Parameters
    ==========
    r : Series or DataFrame
        The core time series of periodic returns.  May be a single security
        (column vector) or multiple securities
    normal_kurtosis : float or int, default 0.0
        If 0, uses Fisher’s definition of kurtosis (kurtosis of normal ==
        0.0), i.e. K = k - 3, where k is the classic definition of
        kurtosis, a scaled version of the fourth moment.  Otherwise,
        specify 3 to define K as just k.
    """

    if normal_kurtosis == 0.0:
        s = r.skew()
        k = r.kurt() - 3.
        n = r.notnull().count()
        jb = (n / 6.) * (s ** 2 + 0.25 * k ** 2)
    elif normal_kurtosis == 3.0:
        jb = np.apply_along_axis(scs.jarque_bera, 0, r).T[:, 0]
    else:
        raise ValueError('`normal_kurtosis` should be one of `0` or `3`')
    return Series(jb, index=r.columns, name='jarque_bera')


# Core return analysis functions.
# Grouped here (1) in order of dependency and then (2) alphabetically.
# ----------------------------------------------------------------------------

@utils.appender(_defaultdocs)
def bias_ratio(r, anlz=True, ddof=1):
    """Measures how far returns are from an unbiased distribution.

    Numerator: interval [0, +1 standard deviation of returns]
    Denominator: interval [-1 standard deviation of returns, 0)

    Note that this measure can give a very different 'reading' than skew.
    """

    std = stdev(r, anlz=anlz, ddof=ddof)
    num = r[(r >= 0) & (r <= std)].count()
    denom = 1 + r[(r < 0) & (r >= -1. * std)].count()
    ratio = num / denom
    return ratio


@utils.appender(_defaultdocs)
def sharpe_ratio(r, anlz=True, ddof=1, rf=None, method='geometric',
                 reload=False):
    """Sharpe ratio, a risk-adjusted measure of reward per unit of  risk.

    The higher the Sharpe Ratio, the better. The numerator is the difference
    between the portfolio's annualized return and the annualized return of a
    risk-free instrument. The denominator is the portfolio's annualized
    standard deviation.
    """

    # TODO: 
    # - Automatic de-annualization for periods <1 yr
    #       (but maybe you want to keep this manual).
    # - Rolling

    freq = r.index.freq.freqstr
    ri = cumulative_return(r, method=method, anlz=anlz)
    si = stdev(r, anlz=anlz, ddof=ddof)

    if rf is None:
        rf = datasets.load_rf(reload=reload, freq=freq).reindex(r.index)
        rf = prep(rf, freq=freq, in_format='dec')
        rf = cumulative_return(rf, method=method, anlz=anlz)[0]
    elif not isinstance(rf, (int, float)):
        if not anlz:
            # De-annualize the input `rf`, assumed to be anlzd as given
            yrs = ri.count() / utils.convertfreq(freq)
            rf = (1. + rf) ** yrs - 1.
    else:
        raise ValueError('`type(rf)` must be in (None, int, float)')

    # TODO: you may want to use .values to ignore indices here
    return (ri - rf) / si


@utils.appender(_defaultdocs)
def msquared(r, benchmark, anlz=True, ddof=1, method='geometric',
             rf=None, reload=False):
    """M-squared, a risk-adjusted performanc metric.

    Hypothetical return of the security if volatility is scaled to equal
    that of the benchmark.  Formula:

    msquared = [(r_i - r_f) / s_r] * s_p + rf
    where:
    - r_i -> security return
    - s_i -> standard deviation of security returns
    - s_m -> standard deviation of benchmark returns
    - r_f -> risk-free rate
    - all are annualized, by convention
    """

    freq = r.index.freq.freqstr
    ri = cumulative_return(r, method=method, anlz=anlz)
    si = stdev(r, anlz=anlz, ddof=ddof)
    sm = stdev(benchmark, anlz=anlz, ddof=ddof)

    if rf is None:
        rf = datasets.load_rf(reload=reload, freq=freq).reindex(r.index)
        rf = cumulative_return(rf, method=method, anlz=anlz)
    elif not isinstance(rf, (int, float)):
        if not anlz:
            # De-annualize the input `rf`, assumed to be anlzd as given
            yrs = ri.count() / utils.convertfreq(freq)
            rf = (1. + rf) ** yrs - 1.
    else:
        raise ValueError('`type(rf)` must be in (None, int, float)')

    return ( (ri - rf) / (sm / si) ) + rf


def rollup(r, log=False, out_freq='Q'):
    """Downsample returns by geometric linking.

    Parameters
    ==========
    r : Series or DataFrame
        The core time series of periodic returns.  May be a single security
        (column vector) or multiple securities
    log : bool, default False
        If False, use geometric methodology; if True, use continuous
        compounding with natural logarithm
    out_freq : str
        The desired frequency of the output/result.  Refer to pandas'
        Offset Aliases:
        pandas.pydata.org/pandas-docs/stable/timeseries/offset-aliases.
        Includes 'D', 'W', 'M', 'Q', 'A'
    """

    freq = r.index.freq.freqstr
    if __frqs__[out_freq] > utils.convertfreq(freq):
        raise ValueError('Cannot upsample a frequency from %s to %s'
                         % (freq, out_freq))
    else:
        return (r.groupby(pd.Grouper(freq=out_freq))
                 .apply(cumulative_return, method='cumulative',
                        anlz=False, log=log))


def max_drawdown(r, log=False, full_stats=True):
    """The maximum drawdown from high water mark, expressed as negative number.

    Basic computation: a return index, `ri`, is formed with `r`.
    The max drawdown is then the minimum of vectorized [ ri / cummax(ri) - 1 ].

    Parameters
    ==========
    r : Series or DataFrame
        The core time series of periodic returns.  May be a single security
        (column vector) or multiple securities
    log : bool, default False
        If False, use geometric methodology; if True, use continuous
        compounding with natural logarithm
    full_stats : bool, default True
        If True, return a DataFrame with additional info:
        'max_dd', 'start', 'end', 'recov', 'duration_days', 'recov_days'.
        Otherwise, return just the maximum drawdown for each security
    """

    dd = drawdown_index(r, log=log)
    mdd = dd.min()

    if full_stats:

        def dd_start(col, end):
            return col.loc[:end].sort_index(ascending=False).argmax()

        def dd_recov(col, end):
            recov = col.loc[end:].argmax()
            if col[recov] != 0:
                recov = np.nan
                return recov
            return recov - end

        # Note .apply won't work on `start` here because `end` is not a scalar
        end = dd.apply(lambda col: col.argmin())
        start = [dd_start(dd[i], j) for i, j in zip(dd.columns, end.values)]
        recov = [dd_recov(dd[i], j) for i, j in zip(dd.columns, end.values)]
        mdd = DataFrame({'max_dd' : mdd, 'end' : end, 'start' : start,
                         'recov_days' : recov},
                        index=r.columns)
        mdd.loc[:, 'duration_days'] = mdd.end - mdd.start
        mdd.loc[:, 'recov'] = mdd.end + mdd.recov_days
        mdd = mdd[['max_dd', 'start', 'end', 'recov', 'duration_days',
                 'recov_days']]

    return mdd


@utils.appender(_defaultdocs)
def calmar_ratio(r, anlz=True, log=False):
    """Return Calmar Ratio, a measure of return per unit of downside risk.

    calmar_ratio = compounded anlzd return / abs(max drawdown)
    """

    # TODO: clean up params
    return cumulative_return(r, anlz=anlz) \
               / np.abs(max_drawdown(r, log=log, full_stats=False))


@utils.appender(_defaultdocs)
def ulcer_index(r, log=False):
    """The Ulcer Index, a metric measuring drawdown risk & severity.

    Factors in the depth and duration of drawdowns from recent peaks.
    A large value indicates greater ex-post risk.

    See also
    ========
    The Investor's Guide to Fidelity Funds.  Peter G. Martin & Byron B. McCann,
      1989.
    """

    return np.sqrt(np.sum(np.square(drawdown_index(r, log=log)) / r.count()))


@utils.appender(_defaultdocs)
def semi_stdev(r, anlz=True, ddof=1, thresh=0.0):
    """Calculate semi-standard deviation.

    A characterization of the upside/downside risk of a distribution.  The
    semi-standard deviation is always lower than the total standard
    deviation of the distribution.  Also known as the downside deviation.

    Parameters
    ==========
    r : Series or DataFrame
        The core time series of periodic returns.  May be a single security
        (column vector) or multiple securities

    Notes
    =====
    Note that the methodology does not simply 'filter' for returns below
    the threshold (`thresh`) and then take the standard deviation of this
    subset.  Instead, it uses the formula:

    sqrt( sum([min(r_i - thresh, 0] **2 ) / (n - ddof) )

    where:
    - r_i is each period's return
    - thresh is the threshold or minimum acceptable return (MAR)
    - n is the *total* non-null periods (again, not the filtered subset)
    - ddof is the degrees of freedom (1 for sample, 0 for population)

    That is, deviation from the threshold squared is computed if the
    security return is less than the threshold; otherwise, it will be
    equal to zero.

    See also
    ========
    Managing Investment Portfolios: A Dynamic Process, Third Edition,
      John L. Maginn, CFA, Donald L. Tuttle, CFA, Jerald E. Pinto, CFA,
      and Dennis W. McLeavey, CFA, editors. © 2007 CFA Institute.
    """
    n = r.count()
    if thresh == 'mean':
        thresh = r.mean()
    else:
        # `thresh` is assumed to be an annualized rate; de-anlz it
        thresh = np.subtract(np.add(thresh, 1.) ** (np.divide(1., n)), 1.)
    sum_sqs = (np.minimum(np.subtract(r, np.asscalar(thresh)), 0.) ** 2).sum()
    normalize = np.subtract(n, ddof)
    res = np.sqrt(sum_sqs.div(normalize))
    if anlz:
        freq = r.index.freq.freqstr
        res *= np.sqrt(utils.convertfreq(freq))
    return res


@utils.appender(_defaultdocs, passed_to='semi_std')
def sortino_ratio(r, ddof=1, log=False, thresh=0.0, anlz=True, **kwargs):
    """A measure of risk-adjusted return that penalizes downside volatility.

    The Sortino Ratio is similar to the Sharpe Ratio except the Sortino Ratio
    uses annualized downside deviation for the denominator, whereas Sharpe uses
    annualized standard deviation. The numerator is the difference between the
    portfolio's annualized return and `thresh`, sometimes called the Minimum
    Acceptable Return (MAR). The denominator is the portfolio's annualized
    downside deviation.
    """

    std = semi_stdev(r, thresh=thresh, ddof=ddof, **kwargs)
    r = cumulative_return(r, anlz=anlz, log=log)
    return (r - thresh) / std


@utils.appender(_defaultdocs, passed_to='pd.rolling')
def rolling_returns(r, window=None, log=False, anlz=True, **kwargs):
    """Returns computed for overlapping rolling windows.

    For rolling excess returns, use `excess_returns` and specify `window`.
    """

    rr = (r.rolling(window=window, **kwargs)
           .apply(cumulative_return, kwargs={'anlz' : anlz, 'log' : log}))
    return rr


@utils.appender(_defaultdocs, passed_to='rolling_returns')
def min_max_return(r, window=None, anlz=True, log=False, **kwargs):
    """A pandas object describing the min and max return over the sample."""
    if window is not None:
        r = rolling_returns(r, window=window, log=log, anlz=anlz,
                            **kwargs)
    return DataFrame({'min' : r.min(), 'max' : r.max()})


@utils.appender(_defaultdocs, passed_to='rolling_returns')
def pct_negative(r, window=None, anlz=True, log=False, thresh=0.0, **kwargs):
    """Percent of periods in which returns were less than `thresh`."""
    if window is not None:
        r = rolling_returns(r, window=window, log=log, anlz=anlz, **kwargs)
    return r[r < thresh].notnull().count() / r.notnull().count()


@utils.appender(_defaultdocs, passed_to='rolling_returns')
def pct_positive(r, window=None, anlz=True, log=False, thresh=0.0, **kwargs):
    """Percent of periods in which returns were g.e.q. than `thresh`."""
    if window is not None:
        r = rolling_returns(r, window=window, log=log, anlz=anlz, **kwargs)
    return r[r > thresh].notnull().count() / r.notnull().count()


@utils.appender(_defaultdocs, passed_to='rolling_returns')
def pct_pos_neg(r, window=None, anlz=True, log=False, thresh=0.0, **kwargs):
    """Convenience function combining `pct_negative` and `pct_positive`."""
    neg = pct_negative(thresh=thresh, window=window, log=log, anlz=anlz,
                       **kwargs)
    pos = pct_positive(thresh=thresh, window=window, log=log, anlz=anlz,
                       **kwargs)
    df = DataFrame(pd.concat((pos, neg), axis=1))
    df.columns = ['pct_pos', 'pct_neg']
    return df


@utils.appender(_defaultdocs, passed_to='rolling_returns')
def excess_returns(r, benchmark, window=None, anlz=True, method='arithmetic',
                   log=False, **kwargs):
    """Returns of `r` in excess of `benchmark` for each period.

    Note that a geometric return difference is defined as:
    (1 + `r`) / (1 + `benchmark`) - 1
    """

    if window is not None:
        r = rolling_returns(r, window=window, log=log, anlz=anlz, **kwargs)
        benchmark = rolling_returns(benchmark, window=window, log=log,
                                    anlz=anlz, **kwargs)
    if method in ['arithmetic', 'arith']:
        er = r.values - benchmark.values
    elif method in ['geometric', 'geo']:
        er = np.subtract(np.divide(np.add(1., r), np.add(1., benchmark) ), 1.)
    return DataFrame(er, index=r.index, columns=r.columns)


def excess_drawdown_index(r, benchmark, method='CAER'):
    """An index of excess drawdowns from cumulative high water marks.

    Parameters
    ==========
    r : Series or DataFrame
        The core time series of periodic returns.  May be a single security
        (column vector) or multiple securities
    benchmark : Series or single-column DataFrame
        The time series of periodic returns for the benchmark.  Must be a
        single security

    Overview
    ========
    Compared to a basic (absolute) drawdown, there are additional
    specifications that must be considered and can affect the calculation
    result noticeably.

    For instance, excess drawdown could be defined as (1) cumulative return
    of the arithmetic excess returns in each period or use (2) the arithmetic
    difference of cumulative returns.  The differences arise from competing
    single-period versus multi-period models and rebalancing technique.

    Description of methods
    ======================

    **Cumulative Arithmetic Excess Return** (CAER)
    ----------------------------------------------
    1. For each period, compute an arithmetic excess return, `r` - `benchmark`
    2. Build a cumulative return index from the result of (1.).  A cumulative
       return index is the cumulative product of the return relatives, minus 1
    3. Compute an absolute drawdown on this series
    This mimics an approach that is long `r` and short `benchmark` with
    periodic rebalancing (at each period).

    **Cumulative Geometric Excess Return** (CGER)
    ---------------------------------------------
    Same approach as **Cumulative Arithmetic Excess Return**, except
    that geometric excess returns are used in step (1.).  A geometric excess
    return is (1 + `r`) / (1 + `benchmark`) - 1.  The results will be similar
    to the **Cumulative Arithmetic Excess Return* method.  Geometrix excess
    returns are rarely used but will provide less model leakage over
    multi-period samples.  For more, see:
    Bacon, Carl: Excess Returns—Arithmetic or Geometric?
    Journal of Performance Measurement, Vol. 6, No. 3 (Spring 2002): 23-31

    **Excess of Cumulative Returns** (ECR)
    --------------------------------------
    1. For each of `r` and `benchmark`, compute a cumulative return index
    2. Subtract the cumulative return index of `benchmark` from the cumulative
       return index of `r`, from step (1.)
    3. Add 1 to the result from (2.) to build a difference of cumulative
       returns index
    4. Compute an absolute drawdown on this series.
    This mimics a portfolio that is long `r` and short `benchmark`, but
    does not rebalance.  The resulting maximum drawdown will be the most
    severe of these 4 methods, because the weight to `r` relative to
    `benchmark` s effectively increased leading into the high water mark
    because of relative appreciation.

    **Excess of Cumulative Returns with Reset** (ECRR)
    --------------------------------------------------
    1. Same as step (1.) from **Excess of Cumulative Returns**
    2. Same as step (2.) from **Excess of Cumulative Returns**
    3. Same as step (3.) from **Excess of Cumulative Returns**
    4. Compute a running cumulative maximum of the result of (3.) and, for each
       period, record the cumulative return index value of `r` and `benchmark`
       on the date/index at which the running cumulative maxes, respectively,
       have been reached.  Denote these latter two values `r0` and `benchmark0`
    5. At each date, compute `r/r0` - `benchmark/benchmark0`, which results
       in a drawdown series
    This mimics a portfolio that is long `r` and short `benchmark`, and
    only rebalances to equal notional weights *once*, at the high water mark.
    """

    if method.startswith(('c', 'C')):

        if method in ['CAER', 'caer']:
            er = excess_returns(r=r, benchmark=benchmark, method='arithmetic')
        elif method in ['CGER', 'cger']:
            er = excess_returns(r=r, benchmark=benchmark, method='geometric')
        else:
            methods = ['CAER', 'CGER', 'ECR', 'ECRR']
            ve = 'Method must be one of %s (case-insensitive)' % methods
            raise ValueError(ve)
        dd = drawdown_index(er)

    elif method.startswith(('e', 'E')):

        # First 3 steps are shared
        r = return_index(r)
        benchmark = return_index(benchmark)
        er = np.add( np.subtract(r, benchmark), 1. )

        # 'ECR' & 'ECRR' diverge in process from here
        if method in ['ECR', 'ecr']:
            cummax = np.maximum.accumulate
            dd = np.subtract(np.divide(er, cummax(er)), 1.)

        # Will generalize to both n x 1 & n x m
        # Credit to: SO @piRSquared
        elif method in ['ECRR', 'ecrr']:
            cam = np.squeeze(er.expanding().apply(lambda x: x.argmax()).values)
            r0 = r.iloc[cam].values
            b0 = benchmark.iloc[cam].values
            dd = np.subtract( np.divide(r, r0), np.divide(benchmark, b0) )

    return dd


@utils.appender(_defaultdocs, passed_to='excess_returns')
def tracking_error(r, benchmark, window=None, log=False, anlz=True, ddof=1,
                   **kwargs):
    """The active risk of the portfolio: stdev of active returns."""
    er = excess_returns(r, benchmark=benchmark, window=window, log=log,
                       anlz=anlz, **kwargs)
    return stdev(er, anlz=anlz, ddof=ddof)


@utils.appender(_defaultdocs, passed_to='excess_returns')
def batting_avg(r, benchmark, window=None, log=False, anlz=True, thresh=0.0,
                **kwargs):
    """Returns batting average versus the benchmark, rolling optional.

    Batting avg. is the number of periods that the portfolio outperforms
    (or matches) the benchmark divided by the total number of periods.

    It is formulaically equivalent to `pct_positive` with `r` as an excess
    returns stream."""

    er = excess_returns(r, benchmark=benchmark, window=window, log=log,
                        anlz=anlz, **kwargs)
    return er[er >= thresh].notnull().count() / er.notnull().count()


# Up/down statistics
# -----------------------------------------------------------------------------

def downmarket_filter(r, benchmark, thresh=0.0, incl_benchmark=True):
    """Filter returns, including only periods where `benchmark` < `thresh`.

    Parameters
    ==========
    r : Series or DataFrame
        The core time series of periodic returns.  May be a single security
        (column vector) or multiple securities
    benchmark : Series or single-column DataFrame
        The time series of periodic returns for the benchmark.  Must be a
        single security
    thresh : float or 'mean', default 0.0
        The cutoff filter on which to test and filter `benchmark` returns.
        If mean, the arithmetic mean of `benchmark` will be used
    incl_benchmark : bool, default True
        If True, also return the filtered benchmark, with the result being a
        tuple of filtered `r` and filtered `benchmark`.  If False, simply
        return filtered `r`.
    """

    freq = r.index.freq.freqstr
    if thresh == 'mean':
        thresh = benchmark.mean()
    # Bring in `prep` to create compatability with further method chaining
    res = prep(r[(benchmark < thresh).values], freq=freq, in_format='dec')
    if incl_benchmark:
        b = prep(benchmark[(benchmark < thresh).values], freq=freq,
                 in_format='dec')
        res = res, b
    return res


def upmarket_filter(r, benchmark, thresh=0.0, incl_benchmark=True):
    """Filter returns, including only periods where `benchmark` > `thresh`.

    Parameters
    ==========
    r : Series or DataFrame
        The core time series of periodic returns.  May be a single security
        (column vector) or multiple securities
    benchmark : Series or single-column DataFrame
        The time series of periodic returns for the benchmark.  Must be a
        single security
    thresh : float or 'mean', default 0.0
        The cutoff filter on which to test and filter `benchmark` returns.
        If mean, the arithmetic mean of `benchmark` will be used
    incl_benchmark : bool, default True
        If True, also return the filtered benchmark, with the result being a
        tuple of filtered `r` and filtered `benchmark`.  If False, simply
        return filtered `r`.
    """

    freq = r.index.freq.freqstr
    if thresh == 'mean':
        thresh = benchmark.mean()
    res = prep(r[(benchmark >= thresh).values], freq=freq, in_format='dec')
    if incl_benchmark:
        b = prep(benchmark[(benchmark >= thresh).values], freq=freq,
                 in_format='dec')
        res = res, b
    return res


@utils.appender(_defaultdocs)
def upside_capture(r, benchmark, anlz=False, method='geometric', log=False,
                     thresh=0.0):
    """Upside capture versus the benchmark.

    Parameters
    ==========
    r : Series or DataFrame
        The core time series of periodic returns.  May be a single security
        (column vector) or multiple securities
    benchmark : Series or single-column DataFrame
        The time series of periodic returns for the benchmark.  Must be a
        single security

    Notes
    =====
    The upside capture formula for a security versus benchmark is:
    - numerator: r_security | r_benchmark >= 0
    - denominator: r_benchmark | r_benchmark >= 0

    Two variations of `method` are supported here:
    - 'geometric' uses the geometric mean return
    - 'cumulative' uses cumulative return, and is more influenced by
      outliers in both directions
    """

    r, b = upmarket_filter(r, benchmark=benchmark, thresh=thresh)
    if method in ['cum', 'cumulative']:
        dc = cumulative_return(r, anlz=anlz).values \
           / cumulative_return(b, anlz=anlz).values
    elif method in ['geo', 'geometric']:
        geomean = True if not anlz else False
        dc = cumulative_return(r, anlz=anlz, log=log, geomean=geomean).values \
           / cumulative_return(b, anlz=anlz, log=log, geomean=geomean).values
    else:
        methods = ['`cumulative`', '`geometric`']
        raise ValueError('`method` must be one of %s' % methods)
    return Series(dc, index=r.columns)


@utils.appender(_defaultdocs)
def downside_capture(r, benchmark, anlz=False, method='geometric', log=False,
                     thresh=0.0):
    """Downside capture versus the benchmark.

    Parameters
    ==========
    r : Series or DataFrame
        The core time series of periodic returns.  May be a single security
        (column vector) or multiple securities
    benchmark : Series or single-column DataFrame
        The time series of periodic returns for the benchmark.  Must be a
        single security

    Notes
    =====
    The downside capture formula for a security versus benchmark is:
    - numerator: r_security | r_benchmark < 0
    - denominator: r_benchmark | r_benchmark < 0

    Two variations of `method` are supported here:
    - 'geometric' uses the geometric mean return
    - 'cumulative' uses cumulative return, and is more influenced by
      outliers in both directions
    """

    r, b = downmarket_filter(r, benchmark=benchmark, thresh=thresh)
    if method in ['cum', 'cumulative']:
        dc = cumulative_return(r, anlz=anlz).values \
           / cumulative_return(b, anlz=anlz).values
    elif method in ['geo', 'geometric']:
        geomean = True if not anlz else False
        dc = cumulative_return(r, anlz=anlz, log=log, geomean=geomean).values \
           / cumulative_return(b, anlz=anlz, log=log, geomean=geomean).values
    else:
        methods = ['`cumulative`', '`geometric`']
        raise ValueError('`method` must be one of %s' % methods)
    return Series(dc, index=r.columns)


@utils.appender(_defaultdocs)
def capture_ratio(r, benchmark, anlz=False, method='geometric', log=False,
                  thresh=0.0):
    """Capture ratio, the ratio of upside capture to downside capture.

    Notes
    =====
    Two variations of `method` are supported here, that apply to calculation
    of both upside and downside capture:
    - 'geometric' uses the geometric mean return
    - 'cumulative' uses cumulative return, and is more influenced by
      outliers in both directions

    Parameters
    ==========
    r : Series or DataFrame
        The core time series of periodic returns.  May be a single security
        (column vector) or multiple securities
    benchmark : Series or single-column DataFrame
        The time series of periodic returns for the benchmark.  Must be a
        single security


    """

    uc = upside_capture(r, benchmark=benchmark, anlz=anlz, method=method,
                        thresh=thresh)
    dc = downside_capture(r, benchmark=benchmark, anlz=anlz, method=method,
                          thresh=thresh)

    return uc / dc


# Other utility funcs
# -----------------------------------------------------------------------------

@utils.appender(_defaultdocs)
def insert_start(r, base=1.0):
    """Prepend a row to `r` at a -1 date offset.  Returns modified copy of `r`.

    Example
    =======
    x = DataFrame(np.random.rand(5) + 1.,
                  index=pd.date_range('2010-01', freq='M', periods=5))

    print(insert_start(x))
                      0
    2009-12-31  1.00000
    2010-01-31  1.21233
    2010-02-28  1.04071
    2010-03-31  1.39719
    2010-04-30  1.23313
    2010-05-31  1.84174
    """

    r = r.copy()
    freq = r.index.freq.freqstr
    map = {'Y' : 'YearEnd', 'Q' : 'QuarterEnd', 'M' : 'QuarterEnd',
           'W' : 'Week', 'D' : 'BDay',}
    start = r.index[0] - getattr(offsets, map[freq])(1)

    # Or: r.reindex, fill_value=0
    r.loc[start] = np.full_like(r.values[0], base)
    r = r.sort_index()
    return r


# OLS regression - DataFrame/Series wrappers around pyfinance.ols results
# ----------------------------------------------------------------------------

@utils.appender(_defaultdocs)
def beta(r, benchmark, window=None):
    """The parameters (coefficients), excl. the intercept."""
    if window is None:
        model = ols.OLS(y=r, x=benchmark)
        res = DataFrame(np.atleast_1d(model.beta()), index=r.columns,
                          columns=benchmark.columns)
    else:
        model = ols.RollingOLS(y=r, x=benchmark, window=window)
        res = DataFrame(model.beta(), index=model.idx[window-1:],
                          columns=r.columns)
    return res


@utils.appender(_defaultdocs)
def alpha(r, benchmark, window=None, anlz=True):
    """The intercept term (alpha).

    Technically defined as the coefficient to a column vector of ones.
    """

    if window is None:
        model = ols.OLS(y=r, x=benchmark)
        res = Series(np.atleast_1d(model.alpha()), index=r.columns)
    else:
        model = ols.RollingOLS(y=r, x=benchmark, window=window)
        res = DataFrame(model.alpha(), index=model.idx[window-1:],
                          columns=r.columns)
    if anlz:
        freq = r.index.freq.freqstr
        res = np.subtract(np.add(res, 1.) ** utils.convertfreq(freq), 1.)
    return res


@utils.appender(_defaultdocs)
def tstat_alpha(r, benchmark, window=None):
    """The t-statistic of the intercept (alpha)."""
    if window is None:
        model = ols.OLS(y=r, x=benchmark)
        res = Series(np.atleast_1d(model.tstat_alpha()), index=r.columns)
    else:
        model = ols.RollingOLS(y=r, x=benchmark, window=window)
        res = DataFrame(model.tstat_alpha(), index=model.idx[window-1:],
                          columns=r.columns)
    return res


@utils.appender(_defaultdocs)
def tstat_beta(r, benchmark, window=None):
    """The t-statistics of the parameters, excl. the intecept."""
    if window is None:
        model = ols.OLS(y=r, x=benchmark)
        res = DataFrame(np.atleast_1d(model.tstat_beta()), index=r.columns,
                          columns=benchmark.columns)
    else:
        model = ols.RollingOLS(y=r, x=benchmark, window=window)
        res = DataFrame(model.tstat_beta(), index=model.idx[window-1:],
                          columns=r.columns)
    return res


@utils.appender(_defaultdocs)
def rsq(r, benchmark, window=None):
    """The coefficent of determination, R-squared."""
    if window is None:
        model = ols.OLS(y=r, x=benchmark)
        res = Series(np.atleast_1d(model.rsq()), index=r.columns)
    else:
        model = ols.RollingOLS(y=r, x=benchmark, window=window)
        res = DataFrame(model.rsq(), index=model.idx[window-1:],
                          columns=r.columns)
    return res


@utils.appender(_defaultdocs)
def rsq_adj(r, benchmark, window=None):
    """Adjusted R-squared."""
    if window is None:
        model = ols.OLS(y=r, x=benchmark)
        res = Series(np.atleast_1d(model.rsq_adj()), index=r.columns)
    else:
        model = ols.RollingOLS(y=r, x=benchmark, window=window)
        res = DataFrame(model.rsq_adj(), index=model.idx[window-1:],
                          columns=r.columns)
    return res


def factor_loadings(r, factors=None, scale=False, pickle_from=None,
                    pickle_to=None):
    """Security factor exposures generated through OLS regression.

    Incorporates a handful of well-known factors models.

    Parameters
    ==========
    r : Series or DataFrame
        The left-hand-side variable(s).  If `r` is nx1 shape (a Series or
        single-column DataFrame), the result will be a DataFrame.  If `r` is
        an nxm DataFrame, the result will be a dictionary of DataFrames.
    factors : DataFrame or None, default None
        Factor returns (right-hand-side variables).  If None, factor returns
        are loaded from `pyfinance.datasets.load_factors`
    scale : bool, default False
        If True, cale up/down the volatilities of all factors besides MKT & RF,
        to the vol of MKT.  Both means and the standard deviations are
        multiplied by the scale factor (ratio of MKT.std() to other stdevs)
    pickle_from : str or None, default None
        Passed to `pyfinance.datasets.load_factors` if factors is not None
    pickle_to : str or None, default None
        Passed to `pyfinance.datasets.load_factors` if factors is not None
    """

    # TODO:
    # - Might be appropriate for `returns`, will require higher dimensionality
    # - Option to subtract or not subtract RF (Jensen alpha)
    # - Annualized alpha
    # - Add variance inflation factor to output (method of `ols.OLS`)
    # - Add 'missing=drop' functionality (see statsmodels.OLS)

    if factors is None:
        factors = datasets.load_factors(pickle_from=pickle_from,
                                        pickle_to=pickle_to)

    r, factors = utils.constrain(r, factors)

    if isinstance(r, Series):
        n = 1
        r = r.subtract(factors['RF'])
    elif isinstance(r, DataFrame):
        n = r.shape[1]
        r = r.subtract(np.tile(factors['RF'], (n,1)).T)
        # r = r.subtract(factors['RF'].values.reshape(-1,1))
    else:
        raise ValueError('`r` must be one of (Series, DataFrame)')

    if scale:
        # Scale up the volatilities of all factors besides MKT & RF, to the
        # vol of MKT.  Both means and the standard deviations are multiplied
        # by the scale factor (ratio of MKT.std() to other stdevs).
        tgtvol = factors['MKT'].std()
        diff = factors.columns.difference(['MKT', 'RF']) # don't scale these
        vols = factors[diff].std()
        factors.loc[:, diff] = factors[diff] * tgtvol / vols

    # Right-hand-side dict-of-models
    rhs = OrderedDict([('Capital Asset Pricing Model (CAPM)',
                        ['MKT']),
                       ('Fama-French 3-Factor Model',
                        ['MKT', 'SMB', 'HML']),
                       ('Carhart 4-Factor Model',
                        ['MKT', 'SMB', 'HMLD', 'UMD']),
                       ('Fama-French 5-Factor Model',
                        ['MKT', 'SMB', 'HMLD', 'RMW', 'CMA']),
                       ('AQR 6-Factor Model',
                        ['MKT', 'SMB', 'HMLD', 'RMW', 'CMA', 'UMD']),
                       ('Price-Signal Model',
                        ['MKT', 'UMD', 'STR', 'LTR'])
                       ('Fung-Hsieh Trend-Following Model',
                        ['BDLB', 'FXLB', 'CMLB', 'STLB', 'SILB'])
        ])

    # Get union of keys and sort them according to `factors.columns`' order;
    # used later as columns in result
    cols = set(itertools.chain(*rhs.values()))
    cols = [o for o in factors.columns if o in cols] + ['alpha', 'rsq_adj']

    # Empty DataFrame to be populated with each regression's attributes
    stats = ['coef', 'tstat']
    idx = pd.MultiIndex.from_product([rhs.keys(), stats])
    res = DataFrame(columns=cols, index=idx)

    # Regression calls
    if n > 1:
        # Dict of DataFrames
        d = {}
        for col in r:
            for k, v in rhs.items():
                res = res.copy()
                model = ols.OLS(y=r[col], x=factors[v],
                                          hasconst=False)
                res.loc[(k, 'coef'), factors[v].columns] = model.beta()
                res.loc[(k, 'tstat'), factors[v].columns] = model.tstat_beta()
                res.loc[(k, 'coef'), 'alpha'] = model.alpha()
                res.loc[(k, 'tstat'), 'alpha'] = model.tstat_alpha()
                res.loc[(k, 'coef'), 'rsq_adj'] = model.rsq_adj()
            d[col] = res
        res = d

    else:
        # Single DataFrame
        for k, v in rhs.items():
            model = ols.OLS(y=r, x=factors[v], hasconst=False)
            res.loc[(k, 'coef'), factors[v].columns] = model.beta()
            res.loc[(k, 'tstat'), factors[v].columns] = model.tstat_beta()
            res.loc[(k, 'coef'), 'alpha'] = model.alpha()
            res.loc[(k, 'tstat'), 'alpha'] = model.tstat_alpha()
            res.loc[(k, 'coef'), 'rsq_adj'] = model.rsq_adj()

    return res


def _ewm_helper(obj, func, com, span, halflife, alpha):
    res = getattr(obj.ewm(com=com, span=span, halflife=halflife, alpha=alpha),
                  func)()
    res = res.iloc[-len(res.columns):]
    return res


class Portfolio(object):
    """Modern Portfolio Theory-based statistics for a set of returns & weights.

    See also
    ========
    - R-Squared Risk Management, 'Portfolio Risk Decomposition'
    - Sharpe, William, 'Portfolio Characteristics'
    - Qian, Edward,  'Risk Parity and Diversification'
    """

    def __init__(self, r, weights, freq=None, log=False, name=None,
                 in_format='num', com=None, span=None, halflife=None,
                 alpha=None, max_exposure=1.0):

        self.r = r
        if log:
            self.r = return_relatives(self.r, log=log)
        self.r = prep(self.r, freq=freq, name=None, in_format=in_format)

        self.weights = weights
        self.freq = self.r.index.freq.freqstr
        self._n = utils.convertfreq(self.freq)
        self.com = com
        self.span = span
        self.halflife = halflife
        self.alpha = alpha
        self.noa = self.r.shape[1]
        self.ew = utils.equal_weights(n=self.noa, sumto=max_exposure)
        self.max_exposure = max_exposure

    def portfolio(self, weights=None):
        weights = self.weights if weights is None else weights

        return np.sum(self.r * weights, axis=1)

    def expected_return(self, weights=None, anlz=False):
        weights = self.weights if weights is None else weights
        if any((self.com, self.span, self.halflife, self.alpha)):
            res = (self.r.ewm(com=self.com, span=self.span,
                              halflife=self.halflife, alpha=self.alpha)
                              .mean()
                              .iloc[-1])
        else:
            res = self.r.mean()
        if weights.ndim > 1:
            res = np.sum(res.values * weights, axis=1)
        else:
            res = np.sum(res * weights)
        if anlz:
            # TODO: arithmetic/geometric?
            res *= self._n

        return res

    def covariance(self):
        """Full covariance matrix."""
        if any((self.com, self.span, self.halflife, self.alpha)):
            res = _ewm_helper(self.r, 'cov', com=self.com, span=self.span,
                              halflife=self.halflife, alpha=self.alpha)
        else:
            res = self.r.cov()

        return res

    def correlation(self):
        """Full correlation matrix."""
        if any((self.com, self.span, self.halflife, self.alpha)):
            res = _ewm_helper(self.r, 'corr', com=self.com, span=self.span,
                              halflife=self.halflife, alpha=self.alpha)
        else:
            res = self.r.corr()

        return res

    def variance(self, weights=None, anlz=True):
        weights = self.weights if weights is None else weights

        cov = self.covariance()
        if weights.ndim > 1:
            innerdot = np.squeeze(np.dot(cov, weights[:, :, None]))
            res = np.dot(weights, innerdot)[:, 0]
        else:
            innerdot = np.dot(cov, weights)
            res = np.dot(weights.T, innerdot)
        if anlz:
            res *= self._n

        return res

    def gmvp(self):
        """Global minimum-variance portfolio."""
        cons = ({'type' : 'eq',
                 'fun' : lambda x: np.sum(x) - self.max_exposure})
        bnds = tuple((0, 1) for _ in range(self.noa))
        res = sco.minimize(self.stdev, x0=self.ew, args=(False,),
                            method='SLSQP', bounds=bnds, constraints=cons).x

        return res

    @lru_cache(maxsize=None)
    def monte_carlo(self, trials=2500):
        rw = utils.random_weights(size=(trials, self.noa))
        pvols = self.stdev(weights=rw)
        prets = self.expected_return(weights=rw)
        # TODO: annualize

        return pvols, prets

    @lru_cache(maxsize=None)
    def frontier(self, n_portfolios=50):

        # TODO: annualize

        n_portfolios = int(n_portfolios)

        # Form a range of portfolio returns across which to minimize variance.
        # The lower bound is the return at the min-variance weights;
        # the upper bound is the maximum return for any single asset
        lower = self.expected_return(weights=self.gmvp())
        upper = self.r.mean().max()
        rets = np.linspace(lower, upper, n_portfolios)

        vols = []
        opts = []
        for ret in rets:
            cons = ({'type' : 'eq',
                     'fun' : lambda x: self.expected_return(weights=x) - ret},
                    {'type' : 'eq',
                     'fun' : lambda x: np.sum(x) - self.max_exposure})
            bnds = tuple((0, 1) for _ in range(self.noa))
            optv = sco.minimize(self.stdev, x0=self.ew, args=(False,),
                               method='SLSQP', bounds=bnds, constraints=cons)
            opts.append(optv.x)
            vols.append(self.stdev(weights=optv.x))

        return {'weights' : np.array(opts), 'stdev' : vols, 'rets' : rets}

    def stdev(self, weights=None, anlz=True):
        weights = self.weights if weights is None else weights
        res = np.sqrt(self.variance(weights=weights, anlz=False))
        if anlz:
            res *= np.sqrt(self._n)

        return res

    def risk_contribution(self, weights=None):
        weights = self.weights if weights is None else weights

        cov = self.covariance()
        var = self.variance(weights=weights, anlz=False)

        # `c` -> vector of covariances between each asset with the portfolio
        c = cov.dot(weights)
        acv = weights * c
        pcv = acv / var

        return pcv

    def marginal_risk(self, weights=None):
        weights = self.weights if weights is None else weights

        cov = self.covariance()
        c = cov.dot(weights)

        return 2.0 * c

    def value_at_risk(self, r, n=21, method='analytic', normdist=False,
                      bestfit=True, dist=None, nboot=1e4, c=None, 
                      weights=None):

        com = self.com
        span = self.span
        halflife = self.halflife
        alpha = self.alpha

        if c is None:
            c = [0.99, 0.95, 0.90]
        c = np.array(c)
        a = 1. - c

        # TODO: non-portfolio (security-by-security) implementation
        port = self.portfolio(weights=weights)

        # Validate that only 1/3 params below is not False or not None.
        #     (Probably simpler: np.count_nonzero, but be explicit).
        distparams = [normdist, bestfit, dist]
        nonnull = distparams.count(False) + distparams.count(None)
        if nonnull < 2:
            raise ValueError('all but one of `normdist`, `bestfit`, and `dist`'
                             ' should be either None or False')

        elif bestfit:
            dist = getattr(scs, returns.BestFitDist(x=port).fit().best()['name'])
        elif normdist:
            dist = scs.norm
        elif dist:
            dist = getattr(scs, dist)

        if method == 'analytic':
            # If we were only working with scs.norm, we could just find `loc`
            # and `scale` outright here, but working with other distributions
            # might require additional shape parameters, so call `.fit` first
            *shape, loc, scale = dist.fit(port)
            var = -dist.ppf(a, loc=loc, scale=scale, *shape)
            cvar = (1. / a) * dist.pdf(dist.ppf(a)) * scale - loc
            var = (1. + var) ** n - 1.
            cvar = (1. + cvar) ** n - 1.
        else:
            if method == 'bootstrap':
                nboot = int(nboot)
                if any((com, span, halflife, alpha)):
                    bootstrap = general.ewm_bootstrap(port, size=(nboot, n),
                                                      com=com, span=span,
                                                      halflife=halflife,
                                                      alpha=alpha)
                else:
                    bootstrap = np.random.choice(port, size=(nboot, n))
                sample = np.prod(bootstrap + 1., axis=1) - 1.
            elif method == 'rolling':
                sample = (port.rolling(window=n)
                             .apply(lambda x: np.prod(1. + x) - 1.))
            else:
                raise ValueError("`method` must be in ('analytic',"
                                 " 'bootstrap', 'rolling')")

            var = -scs.scoreatpercentile(sample, 100. * a)
            mask = (sample[:, None] <= -var)
            # OLD: cvar = -np.array([sample[sample < -v].mean() for v in var])
            # Vectorized: sumproduct divided by sum equals mean
            cvar = -1. * (sample.dot(mask) / mask.sum(axis=0))

        res = DataFrame.from_records((var, cvar), index=['var', 'cvar'],
                                     columns=c).T

        return res


# -----------------------------------------------------------------------------

def extend_pandas():
    """Extends PandasObject (Series, DataFrame) with some funcs defined here.

    Similar functionality to `.pipe` and its facilitation of method chaining.
    """

    PandasObject.alpha = alpha
    PandasObject.batting_avg = batting_avg
    PandasObject.beta = beta
    PandasObject.bias_ratio = bias_ratio
    PandasObject.calmar_ratio = calmar_ratio
    PandasObject.capture_ratio = capture_ratio
    PandasObject.cond_correl = cond_correl
    PandasObject.cross_corr = cross_corr
    PandasObject.correl = correl
    PandasObject.covar = covar
    PandasObject.cumulative_return = cumulative_return
    PandasObject.cumulative_returns = cumulative_returns
    PandasObject.downmarket_filter = downmarket_filter
    PandasObject.downside_capture = downside_capture
    PandasObject.drawdown_index = drawdown_index
    PandasObject.excess_drawdown_index = excess_drawdown_index
    PandasObject.excess_returns = excess_returns
    PandasObject.insert_start = insert_start
    PandasObject.jarque_bera = jarque_bera
    PandasObject.max_drawdown = max_drawdown
    PandasObject.msquared = msquared
    PandasObject.prep = prep
    PandasObject.return_index = return_index
    PandasObject.return_relatives = return_relatives
    PandasObject.rollup = rollup
    PandasObject.rolling_returns = rolling_returns
    PandasObject.rsq = rsq
    PandasObject.rsq_adj = rsq_adj
    PandasObject.sharpe_ratio = sharpe_ratio
    PandasObject.sortino_ratio = sortino_ratio
    PandasObject.semi_stdev = semi_stdev
    PandasObject.stdev = stdev
    PandasObject.ulcer_index = ulcer_index
    PandasObject.upmarket_filter = upmarket_filter
    PandasObject.upside_capture = upside_capture
    PandasObject.tstat_alpha = tstat_alpha
    PandasObject.tstat_beta = tstat_beta
    PandasObject.factor_loadings = factor_loadings