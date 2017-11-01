"""Generalized core utilities in the pyfinance package.

Some do not pertain directly to finance.

Descriptions
============

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
pickle_option
    Decorator lending the option to pickle expensive functions to/from.
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
    'appender', 'avail', 'can_broadcast', 'convertfreq', 'constrain',
    'constrain_horizon', 'dropcols', 'encode', 'equal_weights',
    'expanding_stdize', 'flatten', 'isiterable', 'public_dir',
    'random_tickers', 'random_weights', 'rolling_windows', 'view'
    ]

from functools import wraps
import inspect
import itertools
import random
import re
import string
import textwrap
try:
   import cPickle as pickle
except ImportError:
   import pickle

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from pandas.tseries import offsets

def appender(defaultdocs, passed_to=None):
    """Decorator for appending commonly used parameter definitions.

    Useful in cases where functions repeatedly use the same parameters.  (But
    where a class implementation is not appropriate.)

    `defaultdocs` -> dict, format as shown in Example below, with keys being
    parameters and values being descriptions.

    Example
    =======
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
        params = ''.join([textwrap.dedent(defaultdocs
                   .get(param, msg % passed_to)) for param in params])
        func.__doc__ += '\n\nParameters\n' + 10 * '=' + params
        return func

    return _doc


def avail(df):
    """Return start & end availability for each column in a DataFrame."""
    avail = DataFrame({
        'start' : df.apply(lambda col: col.first_valid_index()),
        'end' : df.apply(lambda col: col.last_valid_index())
                     })
    return avail[['start', 'end']]


def can_broadcast(*args):
    """Logic test: can input arrays be broadcast?"""
    # TODO: `def fix_broadcast`: attempt to reshape
    try:
        np.broadcast(*args)
        return True
    except ValueError:
        return False


def convertfreq(freq):
    """Convert string frequencies to periods per year.

    Used in upsampling & downsampling.

    `freq` is case-insensitive and may be:
    - One of ('D', 'W', 'M', 'Q', 'A')
    - One of pandas anchored offsets, such as 'W-FRI':
      pandas.pydata.org/pandas-docs/stable/timeseries.html#anchored-offsets

    Example
    =======
    print(convertfreq('M'))
    12.0

    print(convertfreq('Q'))
    4.0

    print(convertfreq('BQS-DEC'))
    4.0
    """

    freq = freq.upper()

    days = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP',
              'OCT', 'NOV', 'DEC']

    weekoffsets = ['W-%s' % d for d in days]
    qtroffsets = ['Q-%s' % m for m in months] \
               + ['QS-%s' % m for m in months] \
               + ['BQ-%s' % m for m in months] \
               + ['BQS-%s' % m for m in months]
    annoffsets = ['A-%s' % m for m in months] \
               + ['AS-%s' % m for m in months] \
               + ['BA-%s' % m for m in months] \
               + ['BAS-%s' % m for m in months]

    freqs = {'D' : 252., 'W' : 52., 'M' : 12., 'Q' : 4., 'A' : 1.}
    freqs.update(zip(weekoffsets, [52.] * len(weekoffsets)))
    freqs.update(zip(qtroffsets, [4.] * len(qtroffsets)))
    freqs.update(zip(annoffsets, [1.] * len(annoffsets)))

    return freqs[freq]


def constrain(*objs):
    """Constrain group of DataFrames & Series to intersection of indices.

    Parameters
    ==========
    objs : iterable
        DataFrames and/or Series to constrain

    Returns
    =======
    new_dfs : list of DataFrames, copied rather than inplace
    """

    # TODO: build in the options to first dropna on each index before finding
    #     intersection, AND to use `dropcol` from this module.  Note that this
    #     would require filtering out Series to which dropcol isn't applicable.

    # A little bit of set magic below.
    # Note that pd.Index.intersection only applies to 2 Index objects
    common_idx = pd.Index(set.intersection(*[set(o.index) for o in objs]))
    new_dfs = [o.reindex(common_idx) for o in objs]

    return tuple(new_dfs)


def constrain_horizon(r, strict=False, cust=None, years=0, quarters=0,
                      months=0, days=0, weeks=0,  year=None, month=None,
                      day=None):

    """Constrain a Series/DataFrame to a specified lookback period.

    See the documentation for dateutil.relativedelta:
    dateutil.readthedocs.io/en/stable/relativedelta.html

    Parameters
    ==========
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


def dropcols(df, start=None, end=None):
    """Drop columns that contain NaN within [start, end] inclusive.

    A wrapper around DataFrame.dropna() that builds an easier *subset*
    syntax for tseries-indexed DataFrames.

    Parameters
    ==========
    df : DataFrame
    start : str or datetime, default None
        start cutoff date, inclusive
    end : str or datetime, default None
        end cutoff date, inclusive

    Example
    =======
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
    =======
    print(equal_weights(5))
    [ 0.2  0.2  0.2  0.2  0.2]

    print(equal_weights(5, sumto=1.2))
    [ 0.24  0.24  0.24  0.24  0.24]
    """

    return sumto * np.array(n * [1 / n])


def expanding_stdize(obj, **kwargs):
    """Standardize a pandas object column-wise on expanding window.

    **kwargs -> passed to `obj.expanding`

    Example
    =======
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

    return (obj - obj.expanding(**kwargs).mean()) \
         / (obj.expanding(**kwargs).std())


def flatten(iterable):
    """Flatten a nested iterable.  Returns a generator object.

    More flexible than list comprehension of the format:
    `[item for sublist in list for item in sublist]` because it can handle a
    'hybrid' list of lists where some elements are not iterable.

    Example
    =======
    l1 = [[1, 2], 1, [1]]
    l2 = [[1, 2], ['a', ['b']], [1]]

    print(list(flatten(l1)))
    print(list(flatten(l2)))
    [1, 2, 1, 1]
    [1, 2, 'a', 'b', 1]

    Source
    ======
    http://rightfootin.blogspot.com/2006/09/more-on-python-flatten.html
    """

    # Functional, but also throws TypeError if `iterable` is not iterable
    it = iter(iterable)

    for e in it:
        if isinstance(e, (list, tuple)):
            for f in flatten(e):
                yield f
        else:
            yield e


def isiterable(obj):
    try:
        obj.__iter__()
        return True
    except AttributeError:
        return False


def pickle_option(func):
    sig = inspect.signature(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        bound_args = sig.bind(*args, **kwargs)
        pickle_from = bound_args.arguments.get('pickle_from', \
                             sig.parameters['pickle_from'].default)
        pickle_to = bound_args.arguments.get('pickle_to', \
                             sig.parameters['pickle_to'].default)

        if pickle_from:
            with open(pickle_from + '.pickle', 'rb') as f:
                result = pickle.load(f)
        else:
            result = func(*args, **kwargs)

        if pickle_to:
            with open(pickle_to + '.pickle', 'wb') as f:
                pickle.dump(result, f)

        return result

    return wrapper


def public_dir(obj, underscores=1):
    """Wrapper around `dir`, but designed to exclude (semi-) private methods.

    Return an alphabetized list of names comprising the attributes
    of the given object, EXCEPT those starting with the specified number of
    `underscores`.  

    Example
    =======

    class Cls:
        def __init__(self):
            self.a = 1
            self._a = 2
        def _helper(self):
            pass
        def __private__(self):
            pass
        def normfunc(self):
            pass

    c = Cls()
    public_dir(c)
    Out[16]: ['a', 'normfunc']

    public_dir(c, 2)
    Out[17]: ['_a', '_helper', 'a', 'normfunc']
    """

    underscores *= '_'
    return [f for f in dir(obj) if not f.startswith(underscores)]


def random_tickers(length, n, ends_in=None, letters=None):
    """Generate a length-n list of random ticker symbols.

    Parameters
    ==========
    length : int
        the length of each ticker string
    n : int
        number of tickers to generate
    endsin : str, default None
        specify the final element(s) of each ticker (for example, 'X')
    letters : list or set, default None
        Container of possible letters to choose from.  If None, defaults to
        `string.ascii_uppercase`

    Examples
    ========
    print(random_tickers(5, 4, 'X'))
    ['UZTFX', 'ROYAX', 'ZBVIX', 'IUWYX']

    print(random_tickers(5, 4))
    ['OLHZP', 'MCAAJ', 'JMKFD', 'FFSCH']
    """

    if letters is None:
        letters = string.ascii_uppercase
    if endsin:
        length = length - len(endsin)
    res = random.sample(list(
              itertools.combinations_with_replacement(letters, length)), n)
    res = [''.join(i) for i in res]
    if endsin:
        res = [r + endsin for r in res]

    return res


def random_weights(size, sumto=1.):
    """Generate a np.array of `n` random weights that sum to `sumto`.

    Note that sumto is subject to typical Python floating point limitations.
    """

    w = np.random.random(size)
    w = sumto * w.T / np.sum(w.T, axis=0)
    return w.T


def rolling_windows(a, window):
    """Creates rolling-window 'blocks' of length `window` from `a`.

    Note that the orientation of rows/columns follows that of pandas.

    Example
    =======
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


def view(df, row=10, col=5):
    """Abbreviated view similar to .head, but limit both rows & columns."""
    return df.iloc[:row, :col]