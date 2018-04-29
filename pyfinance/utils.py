"""pyfinance utility functions.

Descriptions
------------

appender
    Decorator for appending commonly used parameter definitions.
avail
    Return start & end availability for each column in a DataFrame.
can_broadcast
    Logic test: can input NumPy arrays be broadcast?
convertfreq
    Convert string frequencies to periods per year.
constrain
    Constrain group of DataFrames & Series to intersection of indices.
constrain_horizon
    Constrain a Series/DataFrame to a specified lookback period.
dropcols
    Drop columns that contain NaN within [start, end] inclusive.
encode
    One-hot encode the given input strings.
equal_weights
    Generate `n` equal weights (decimals) summing to `sumto`.
expanding_stdize
    Standardize a pandas object column-wise on expanding window.
flatten
    Flatten a nested iterable.  Returns a generator object.
isiterable
    Test whether `obj` is iterable.
public_dir
    List of attributes except those starting with specified underscores.
random_tickers
    Generate a size `n` list of random ticker symbols, each len `length`.
random_weights
    Generate a np.array of `n` random weights that sum to `sumto`.
rolling_windows
    Creates rolling-window 'blocks' of length `window` from `a`.
view
    Abbreviated view similar to .head, but limit both rows & columns.
"""

__author__ = 'Brad Solomon <brad.solomon.1124@gmail.com>'
__all__ = [
    'appender', 'avail', 'can_broadcast', 'constrain',
    'constrain_horizon', 'dropcols', 'encode', 'equal_weights',
    'expanding_stdize', 'isiterable', 'public_dir',
    'random_tickers', 'random_weights', 'rolling_windows', 'view',
    'unique_everseen', 'uniqify'
    ]

from collections import Callable
import inspect
import itertools
import random
import re
import string
import textwrap
import sys

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from pandas.tseries import offsets
from pandas.tseries.frequencies import FreqGroup, get_freq_code


PY37 = sys.version_info.major == 3 and sys.version_info.minor >= 7


def appender(defaultdocs, passed_to=None):
    """Decorator for appending commonly used parameter definitions.

    Useful in cases where functions repeatedly use the same parameters.  (But
    where a class implementation is not appropriate.)

    `defaultdocs` -> dict, format as shown in Example below, with keys being
    parameters and values being descriptions.

    Example
    -------
    ddocs = {

        'a' :
        '''
        a : int, default 0
            the first parameter
        ''',

        'b' :
        '''
        b : int, default 1
            the second parameter
        '''
        }

    @appender(ddocs)
    def f(a, b):
        '''Title doc.'''
        # Params here
        pass
    """

    def _doc(func):
        params = inspect.signature(func).parameters
        params = [param.name for param in params.values()]
        msg = '\n**kwargs : passed to `%s`'
        params = ''.join([textwrap.dedent(defaultdocs.get(
            param, msg % passed_to)) for param in params])
        func.__doc__ += '\n\nParameters\n' + 10 * '=' + params
        return func

    return _doc


def avail(df):
    """Return start & end availability for each column in a DataFrame."""
    avail = DataFrame({
        'start': df.apply(lambda col: col.first_valid_index()),
        'end': df.apply(lambda col: col.last_valid_index())
                     })
    return avail[['start', 'end']]


def can_broadcast(*args):
    """Logic test: can input arrays be broadcast?"""
    try:
        np.broadcast(*args)
        return True
    except ValueError:
        return False


def constrain(*objs):
    """Constrain group of DataFrames & Series to intersection of indices.

    Parameters
    ----------
    objs : iterable
        DataFrames and/or Series to constrain

    Returns
    -------
    new_dfs : list of DataFrames, copied rather than inplace
    """

    # TODO: build in the options to first dropna on each index before finding
    #     intersection, AND to use `dropcol` from this module.  Note that this
    #     would require filtering out Series to which dropcol isn't applicable.

    # A little bit of set magic below.
    # Note that pd.Index.intersection only applies to 2 Index objects
    common_idx = pd.Index(set.intersection(*[set(o.index) for o in objs]))
    new_dfs = (o.reindex(common_idx) for o in objs)

    return tuple(new_dfs)


def constrain_horizon(r, strict=False, cust=None, years=0, quarters=0,
                      months=0, days=0, weeks=0,  year=None, month=None,
                      day=None):

    """Constrain a Series/DataFrame to a specified lookback period.

    See the documentation for dateutil.relativedelta:
    dateutil.readthedocs.io/en/stable/relativedelta.html

    Parameters
    ----------
    r : DataFrame or Series
        The target pandas object to constrain
    strict : bool, default False
        If True, raise Error if the implied start date on the horizon predates
        the actual start date of `r`.  If False, just return `r` in this
        situation
    years, months, weeks, days : int, default 0
        Relative information; specify as positive to subtract periods.  Adding
        or subtracting a relativedelta with relative information performs
        the corresponding aritmetic operation on the original datetime value
        with the information in the relativedelta
    quarters : int, default 0
        Similar to the other plural relative info periods above, but note that
        this param is custom here.  (It is not a standard relativedelta param)
    year, month, day : int, default None
        Absolute information; specify as positive to subtract periods.  Adding
        relativedelta with absolute information does not perform an aritmetic
        operation, but rather REPLACES the corresponding value in the
        original datetime with the value(s) in relativedelta
    """

    textnum = {
        'zero': 0,
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5,
        'six': 6,
        'seven': 7,
        'eight': 8,
        'nine': 9,
        'ten': 10,
        'eleven': 11,
        'twelve': 12,
        'thirteen': 13,
        'fourteen': 14,
        'fifteen': 15,
        'sixteen': 16,
        'seventeen': 17,
        'eighteen': 18,
        'nineteen': 19,
        'twenty': 20,
        'twenty four': 24,
        'thirty six': 36,
        }

    relativedeltas = years, quarters, months, days, weeks, year, month, day
    if cust is not None and any(relativedeltas):
        raise ValueError('Cannot specify competing (nonzero) values for both'
                         ' `cust` and other parameters.')
    if cust is not None:
        cust = cust.lower()

        if cust.endswith('y'):
            years = int(re.search(r'\d+', cust).group(0))

        elif cust.endswith('m'):
            months = int(re.search(r'\d+', cust).group(0))

        elif cust.endswith(('years ago', 'year ago', 'year', 'years')):
            pos = cust.find(' year')
            years = textnum[cust[:pos].replace('-', '')]

        elif cust.endswith(('months ago', 'month ago', 'month', 'months')):
            pos = cust.find(' month')
            months = textnum[cust[:pos].replace('-', '')]

        else:
            raise ValueError('`cust` not recognized.')

    # Convert quarters to months & combine for MonthOffset
    months += quarters * 3

    # Start date will be computed relative to `end`
    end = r.index[-1]

    # Establish some funky date conventions assumed in finance.  If the end
    # date is 6/30, the date *3 months prior* is 3/31, not 3/30 as would be
    # produced by dateutil.relativedelta.

    if end.is_month_end and days == 0 and weeks == 0:
        if years != 0:
            years *= 12
            months += years
        start = end - offsets.MonthBegin(months)
    else:
        start = end - offsets.DateOffset(years=years, months=months,
                                         days=days-1, weeks=weeks, year=year,
                                         month=month, day=day)
    if strict and start < r.index[0]:
        raise ValueError('`start` pre-dates first element of the Index, %s'
                         % r.index[0])
    return r[start:end]


def cumargmax(a):
    """Cumulative argmax.

    Parameters
    ----------
    a : np.ndarray

    Returns
    -------
    np.ndarray
    """

    # Thank you @Alex Riley
    # https://stackoverflow.com/a/40675969/7954504

    m = np.asarray(np.maximum.accumulate(a))
    if a.ndim == 1:
        x = np.arange(a.shape[0])
    else:
        x = np.repeat(np.arange(a.shape[0])[:, None], a.shape[1], axis=1)

    x[1:] *= m[:-1] < m[1:]
    np.maximum.accumulate(x, axis=0, out=x)
    return x


def dropcols(df, start=None, end=None):
    """Drop columns that contain NaN within [start, end] inclusive.

    A wrapper around DataFrame.dropna() that builds an easier *subset*
    syntax for tseries-indexed DataFrames.

    Parameters
    ----------
    df : DataFrame
    start : str or datetime, default None
        start cutoff date, inclusive
    end : str or datetime, default None
        end cutoff date, inclusive

    Example
    -------
    df = DataFrame(np.random.randn(10,3),
                   index=pd.date_range('2017', periods=10))

    # Drop in some NaN
    df.set_value('2017-01-04', 0, np.nan)
    df.set_value('2017-01-02', 2, np.nan)
    df.loc['2017-01-05':, 1] = np.nan

    # only col2 will be kept--its NaN value falls before `start`
    print(dropcols(df, start='2017-01-03'))
                      2
    2017-01-01  0.12939
    2017-01-02      NaN
    2017-01-03  0.16596
    2017-01-04  1.06442
    2017-01-05 -1.87040
    2017-01-06 -0.17160
    2017-01-07  0.94588
    2017-01-08  1.49246
    2017-01-09  0.02042
    2017-01-10  0.75094

    """

    if isinstance(df, Series):
        raise ValueError('func only applies to `pd.DataFrame`')
    if start is None:
        start = df.index[0]
    if end is None:
        end = df.index[-1]
    subset = df.index[(df.index >= start) & (df.index <= end)]
    return df.dropna(axis=1, subset=subset)


def dropout(a, p=0.5, inplace=False):
    """Randomly set elements from `a` equal to zero, with proportion `p`.

    Similar in concept to the dropout technique employed within
    neural networks.

    Parameters
    ----------
    a: numpy.ndarray
        Array to be modified.
    p: float in [0, 1]
        Expected proportion of elements in the result that will equal 0.
    inplace: bool

    Example
    -------
    >>> x = np.arange(10, 20, 2, dtype=np.uint8)
    >>> z = dropout(x, p=0.6)
    >>> z
    array([10, 12,  0,  0,  0], dtype=uint8)
    >>> x.dtype == z.dtype
    True
    """

    dt = a.dtype
    if p == 0.5:
        # Can't pass float dtype to `randint` directly.
        rand = np.random.randint(0, high=2, size=a.shape).astype(dtype=dt)
    else:
        rand = np.random.choice([0, 1], p=[p, 1-p], size=a.shape).astype(dt)
    if inplace:
        a *= rand
    else:
        return a * rand


def _uniquewords(*args):
    """Dictionary of words to their indices.  Helper function to `encode.`"""
    words = {}
    n = 0
    for word in itertools.chain(*args):
        if word not in words:
            words[word] = n
            n += 1
    return words


def encode(*args):
    """One-hot encode the given input strings."""
    args = [arg.split() for arg in args]
    unique = _uniquewords(*args)
    feature_vectors = np.zeros((len(args), len(unique)))
    for vec, s in zip(feature_vectors, args):
        for word in s:
            vec[unique[word]] = 1
    return feature_vectors


def equal_weights(n, sumto=1.):
    """Generate `n` equal weights (decimals) summing to `sumto`.

    Note that sum is subject to typical Python floating point limitations.

    n -> int or float; the number of weights, summing to `sumto`

    Example
    -------
    >>> equal_weights(5)
    [ 0.2  0.2  0.2  0.2  0.2]

    >>> equal_weights(5, sumto=1.2)
    [ 0.24  0.24  0.24  0.24  0.24]
    """

    return sumto * np.array(n * [1 / n])


def expanding_stdize(obj, **kwargs):
    """Standardize a pandas object column-wise on expanding window.

    **kwargs -> passed to `obj.expanding`

    Example
    -------
    df = pd.DataFrame(np.random.randn(10, 3))
    print(expanding_stdize(df, min_periods=5))
             0        1        2
    0      NaN      NaN      NaN
    1      NaN      NaN      NaN
    2      NaN      NaN      NaN
    3      NaN      NaN      NaN
    4  0.67639 -1.03507  0.96610
    5  0.95008 -0.26067  0.27761
    6  1.67793 -0.50816  0.19293
    7  1.50364 -1.10035 -0.87859
    8 -0.64949  0.08028 -0.51354
    9  0.15280 -0.73283 -0.84907
    """

    return (obj - obj.expanding(**kwargs).mean())\
        / (obj.expanding(**kwargs).std())


# Annualization factors.  The keys are multiples of 1000
# representing Pandas frequency groups.
# Technically, all of these are just an approximation.  For example,
# there are 251 NYSE trading days in 2017, 252 in 2016 & 2018.

PERIODS_PER_YEAR = {
    FreqGroup.FR_ANN: 1.,
    FreqGroup.FR_QTR: 4.,
    FreqGroup.FR_MTH: 12.,
    FreqGroup.FR_WK: 52.,
    FreqGroup.FR_BUS: 252.,
    FreqGroup.FR_DAY: 252.,  # All days are business days
    FreqGroup.FR_HR: 252. * 6.5,
    FreqGroup.FR_MIN: 252. * 6.5 * 60,
    FreqGroup.FR_SEC: 252. * 6.5 * 60 * 60,
    FreqGroup.FR_MS: 252. * 6.5 * 60 * 60,
    FreqGroup.FR_US: 252. * 6.5 * 60 * 60 * 1000,
    FreqGroup.FR_NS: 252. * 6.5 * 60 * 60 * 1000 * 1000  # someday...
    }


def get_anlz_factor(freq):
    """Find the number of periods per year given a frequency.

    Parameters
    ----------
    freq : str
        Any frequency str or anchored offset str recognized by Pandas.

    Returns
    -------
    float

    Example
    -------
    >>> get_periods_per_year('D')
    252.0
    >>> get_periods_per_year('5D')  # 5-business-day periods per year
    50.4

    >>> get_periods_per_year('Q')
    4.0
    >>> get_periods_per_year('Q-DEC')
    4.0
    >>> get_periods_per_year('BQS-APR')
    4.0
    """

    # 'Q-NOV' would give us (2001, 1); we just want (2000, 1).
    try:
        base, mult = get_freq_code(freq)
    except ValueError:
        # The above will fail for a bunch of irregular frequencies, such
        # as 'Q-NOV' or 'BQS-APR'
        freq = freq.upper()
        if freq.startswith(('A-', 'BA-', 'AS-', 'BAS-')):
            freq = 'A'
        elif freq.startswith(('Q-', 'BQ-', 'QS-', 'BQS-')):
            freq = 'Q'
        elif freq in {'MS', 'BMS'}:
            freq = 'M'
        else:
            raise ValueError('Invalid frequency: %s' % freq)
        base, mult = get_freq_code(freq)
    return PERIODS_PER_YEAR[(base // 1000) * 1000] / mult


def isiterable(obj):
    # Or: collections.Iterable
    # Notice this will return True for strings even though they "look"
    # different than other Python sequence objects.
    try:
        obj.__iter__()
        return True
    except AttributeError:
        return False


def public_dir(obj, max_underscores=0, type_=None):
    """Like `dir()` with additional options for object inspection.

    Attributes
    ----------
    obj: object
        Any object that could be passed to `dir()`
    max_underscores: int, default 0
        If > 0, names that begin with underscores repeated n or more
        times will be excluded.
    type_: None, sequence, str, or type
        Filter to objects of these type(s) only.  if 'callable' or
        'func' is passed, gets mapped to collections.Callable.  if the
        string-version of a type (i.e. 'str') is passed, it gets
        eval'd to its type.

    Examples
    --------
    >>> import os
    >>> # Get all public string constants from os.path
        public_dir(os.path, max_underscores=1, type_=str)
    ['curdir', 'defpath', 'devnull', 'extsep', 'pardir', 'pathsep', 'sep']
    >>> # Get integer constants
        public_dir(os.path, max_underscores=1, type_='int')
    ['supports_unicode_filenames']
    """

    if max_underscores > 0:
        cond1 = lambda i: not i.startswith('_' * max_underscores)  # noqa
    else:
        cond1 = lambda i: True  # noqa
    if type_:
        if isinstance(type_, str):
            if type_ in ['callable', 'func', 'function']:
                type_ = Callable
            elif 'callable' in type_ or 'func' in type_:
                type_ = [i if i not in ['callable', 'func', 'function']
                         else Callable for i in type_]
        if isinstance(type_, str):
            # 'str' --> str (class `type`)
            type_ = eval(type_)
        elif not isinstance(type_, type):
            type_ = [eval(i) if isinstance(i, str) else i for i in type_]
        # else: we have isinstance(type_, type)
        cond2 = lambda i: isinstance(getattr(obj, i), type_)  # noqa
    else:
        cond2 = lambda i: True  # noqa
    return [i for i in dir(obj) if cond1(i) and cond2(i)]


def random_tickers(length, n_tickers, endswith=None, letters=None,
                   slicer=itertools.islice):
    """Generate a length-n_tickers list of unique random ticker symbols.

    Parameters
    ----------
    length : int
        The length of each ticker string.
    n_tickers : int
        Number of tickers to generate.
    endswith : str, default None
        Specify the ending element(s) of each ticker (for example, 'X').
    letters : sequence, default None
        Sequence of possible letters to choose from.  If None, defaults to
        `string.ascii_uppercase`.

    Returns
    -------
    list of str

    Examples
    --------
    >>> from pyfinance import utils
    >>> utils.random_tickers(length=5, n_tickers=4, endswith='X')
    ['UZTFX', 'ROYAX', 'ZBVIX', 'IUWYX']

    >>> utils.random_tickers(3, 8)
    ['SBW', 'GDF', 'FOG', 'PWO', 'QDH', 'MJJ', 'YZD', 'QST']
    """

    # The trick here is that we need uniqueness.  That defeats the
    #     purpose of using NumPy because we need to generate 1x1.
    #     (Although the alternative is just to generate a "large
    #     enough" duplicated sequence and prune from it.)

    if letters is None:
        letters = string.ascii_uppercase
    if endswith:
        # Only generate substrings up to `endswith`
        length = length - len(endswith)
    join = ''.join

    def yield_ticker(rand=random.choices):
        if endswith:
            while True:
                yield join(rand(letters, k=length)) + endswith
        else:
            while True:
                yield join(rand(letters, k=length))

    tickers = itertools.islice(unique_everseen(yield_ticker()), n_tickers)
    return list(tickers)


def random_weights(size, sumto=1.):
    """Generate an array of random weights that sum to `sumto`.

    The result may be of arbitrary dimensions.  `size` is passed to
    the `size` parameter of `np.random.random`, which acts as a shape
    parameter in this case.

    Note that `sumto` is subject to typical Python floating point limitations.
    This function does not implement a softmax check.

    Parameters
    ----------
    size: int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.
    sumto: float, default 1.
        Each vector of weights should sum to this in decimal terms.

    Returns
    -------
    np.ndarray
    """

    w = np.random.random(size)
    if w.ndim == 2:
        if isinstance(sumto, (np.ndarray, list, tuple)):
            sumto = np.asarray(sumto)[:, None]
        w = sumto * w / w.sum(axis=-1)[:, None]
    elif w.ndim == 1:
        w = sumto * w / w.sum()
    else:
        raise ValueError('`w.ndim` must be 1 or 2, not %s' % w.ndim)
    return w


def rolling_windows(a, window):
    """Creates rolling-window 'blocks' of length `window` from `a`.

    Note that the orientation of rows/columns follows that of pandas.

    Example
    -------
    import numpy as np
    onedim = np.arange(20)
    twodim = onedim.reshape((5,4))

    print(twodim)
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]
     [12 13 14 15]
     [16 17 18 19]]

    print(rwindows(onedim, 3)[:5])
    [[0 1 2]
     [1 2 3]
     [2 3 4]
     [3 4 5]
     [4 5 6]]

    print(rwindows(twodim, 3)[:5])
    [[[ 0  1  2  3]
      [ 4  5  6  7]
      [ 8  9 10 11]]

     [[ 4  5  6  7]
      [ 8  9 10 11]
      [12 13 14 15]]

     [[ 8  9 10 11]
      [12 13 14 15]
      [16 17 18 19]]]
    """

    if window > a.shape[0]:
        raise ValueError('Specified `window` length of {0} exceeds length of'
                         ' `a`, {1}.'.format(window, a.shape[0]))
    if isinstance(a, (Series, DataFrame)):
        a = a.values
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    shape = (a.shape[0] - window + 1, window) + a.shape[1:]
    strides = (a.strides[0],) + a.strides
    windows = np.squeeze(np.lib.stride_tricks.as_strided(a, shape=shape,
                                                         strides=strides))
    # In cases where window == len(a), we actually want to "unsqueeze" to 2d.
    #     I.e., we still want a "windowed" structure with 1 window.
    if windows.ndim == 1:
        windows = np.atleast_2d(windows)
    return windows


def unique_everseen(iterable, filterfalse_=itertools.filterfalse):
    """Unique elements, preserving order."""
    # Itertools recipes:
    # https://docs.python.org/3/library/itertools.html#itertools-recipes
    seen = set()
    seen_add = seen.add
    for element in filterfalse_(seen.__contains__, iterable):
        seen_add(element)
        yield element


def uniqify(seq):
    """`Uniqify` a sequence, preserving order.

    A plain-vanilla version of itertools' `unique_everseen`.

    Example
    -------
    >>> s = list('zyxabccabxyz')
    >>> uniqify(s)
    ['z', 'y', 'x', 'a', 'b', 'c'

    Returns
    -------
    list
    """

    if PY37:
        # Credit: Raymond Hettinger
        return list(dict.fromkeys(seq))
    else:
        # Credit: Dave Kirby
        # https://www.peterbe.com/plog/uniqifiers-benchmark
        seen = set()
        # We don't care about truth value of `not seen.add(x)`;
        # just there to add to `seen` inplace.
        return [x for x in seq if x not in seen and not seen.add(x)]


def view(df, row=10, col=5):
    """Abbreviated view like `.head()`, but limit both rows & columns."""
    return df.iloc[:row, :col]
