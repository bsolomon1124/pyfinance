"""Statistical analysis of security returns time series.

Intended to mimic functionality of commercial software such as FactSet,
Axioma, & Zephyr and open-source software such as Pyfolio.

The main class of this module is `Returns`, a subclassed Pandas
DataFrame.  It implements a collection of new methods that pertain
specifically to the study of security returns, such as cumulative
return indices, annualized volatility, and drawdown.

Note
----
The Pandas developers recommend against subclassing a DataFrame in
favor of "piping" method chaining or using composition.

However, these two alternatives are suboptimal here, and quasi-private
magic is built-in here to protect against some common pitfalls,
such as mutation of the Pandas DataFrame that is passed to
the class constructor.

.. _Subclassing Pandas Data Structures:
    https://pandas.pydata.org/pandas-docs/stable/internals.html

.. _geopandas' GeoDataFrame:
    https://github.com/geopandas/geopandas/blob/d3df5eb950d649367318aba1eded63050a6eeca1/geopandas/geodataframe.py#L18  # noqa

Limitations
-----------
- This implementation currently supports only a 1-benchmark case.
  Methods that have a `benchmark` parameter take an instance
  of `Returns` that is a single-column DataFrame.  (Under the
  hood, this is converted to a subclassed Series.)  Accomodating
  multiple benchmarks by using a MultiIndex result is a
  work-in-progress.
- Support for rolling statistics is currently limited.
  However, it is still possible to create a rolling factory/
  object from an instance of `Returns`
"""

__all__ = ['TSeries', 'TFrame']
__author__ = 'Brad Solomon <brad.solomon.1124@gmail.com>'

import copy
import functools

import numpy as np
import pandas as pd

from pyfinance import utils


# TODO: fmt


class TSeries(pd.Series):
    """A time series of periodic returns for a sinngle security.

    Subclass of pandas.Series.

    Parameters
    ----------
    freq : str or None, default None
        Optionally, explicitly specify an Index frequency as str; for
        example, 'M' denotes monthly frequency.

        Hierarchy:
        - If `freq` is specified, it overrides the frequency of the
        input Index.
        - If None and the input Index has a frequency, use that.
        - If None and the input Index has no frequency, attempt to
        infer one.
        - If all the above fails, `self.freq` is None.

    fmt : {'dec', 'num'}
        The "format" (representation) of the input returns.
        'dec' means that 0.05 represents a 5% return.
        'num' means that 5.00 represents a 5% return.
        The same hierarchy applies for `fmt` as does for `freq`.

    Note that **instance methods always return 'dec' format.**
    """

    # Normal properties; these will be passed to manipulation results.
    _metadata = ['freq', 'fmt']

    def __init__(self, *args, **kwargs):
        # Copy because we would otherwise inadvertantly alter
        # mutable arguments suc as Series, _Returns, lists.
        args = [copy.deepcopy(arg) for arg in args]
        freq = kwargs.pop('freq', None)
        fmt = kwargs.pop('fmt', 'dec')
        super().__init__(*args, **kwargs)

        # Deal with cases where a Series or TSeries is passed;
        # we want to (try) to take on its `freq` and `fmt`.
        data = args[0]
        is_series_like = isinstance(data, (TSeries, pd.Series))
        if not is_series_like:
            self.freq = freq
            self.fmt = fmt
        else:
            # TODO: just take on `data`'s `freq` and return None
            # Tested that we aren't mutating a view here.
            _add_freq(self.index, inplace=True)
            # Now, we need to extract `fmt` and `freq` (str) from this.
            if self.index.freq:
                self.freq = self.index.freq.freqstr
            else:
                # `self.index.freq` is None
                self.freq = freq
            # If we're passed a Tseries, also attempt to grab its `fmt`.
            if isinstance(data, TSeries):
                self.fmt = data.fmt
            else:
                self.fmt = fmt

        # Make a final pass at getting str attributes.
        # This covers cases where the frequency is derived from
        # the index passed to Series(), i.e.
        # r = TSeries(s, index=pd.date_range('2017', periods=20, freq='M'))
        if self.index.freq and not self.freq:
            self.freq = self.index.freq.freqstr

    @property
    def _constructor(self):
        return TSeries

    @property
    def _constructor_expanddim(self):
        return TFrame

    # We don't need _constructor_sliced; it raises NotImplementedError
    # by default.

    # "New" public methods
    # -----------------------------------------------------------------

    def ret_rels(self):
        """Return-relatives, 1 + r.

        Returns
        -------
        TSeries
        """

        return self.add(1.)

    def ret_idx(self, base=1.0):
        """Return index with starting value of `base`.

        Returns
        -------
        TSeries
        """

        return self.ret_rels().cumprod().mul(base)

    def cum_ret_idx(self):
        """Cumulative return index; equal to `self.ret_idx() - 1.

        Returns
        -------
        TSeries
        """

        return self.ret_idx().sub(1.)

    def growth_of_x(self, x=1.0):
        """Growth of a starting amount of `x`.

        Returns
        -------
        float
        """

        prod = self.ret_rels().prod() * x
        assert prod > 0, 'TODO'
        return prod

    def cum_ret(self):
        """Cumulative return.

        Returns
        -------
        float
        """

        return self.growth_of_x() - 1.

    def geomean(self):
        """Geometric mean return, a scalar.

        Returns
        -------
        float
        """

        return self.ret_rels().prod() ** (1. / self.count()) - 1.

    def anlzd_ret(self):
        """Annualized return.

        Returns
        -------
        float
        """

        start = self.index[0] - 1
        end = self.index[-1]
        td = end - start
        n = (td.days - 1.) / 365.
        return self.ret_rels().prod() ** (1. / n) - 1.

    def drawdown_idx(self):
        """Percentage drawdown from the cumulative high.

        Returns
        -------
        TSeries
        """

        ri = self.ret_idx()
        return ri.div(ri.cummax()).sub(1.)

    def max_drawdown(self):
        """Maximum drawdown.

        Returns
        -------
        float
        """

        return self.drawdown_idx().min()

    def anlzd_std(self, ddof=0):
        """Annualized standard deviation with `ddof` degrees of freedom.

        Parameters
        ----------
        ddof : int, default 0
            Degrees of freedom.  Passed to `self.std()`
        """

        n = utils.convertfreq(self.freq)
        return self.std(ddof=ddof) * np.sqrt(n)

    def rollup(self, freq):
        """Downsample returns through geometric linking.

        Returns
        -------
        TSeries
        """

        # TODO: this may switch to an alias, something like
        # self.freq = 'Q-DEC'.
        return TSeries(self.resample(freq).apply(lambda f: f.cum_ret()),
                       freq=freq, fmt=self.fmt)


class TFrame(pd.DataFrame):
    _metadata = ['freq', 'fmt']

    def __init__(self, *args, **kwargs):
        freq = kwargs.pop('freq', None)
        fmt = kwargs.pop('fmt', 'dec')
        args = [copy.deepcopy(arg) for arg in args]
        super().__init__(*args, **kwargs)
        if len(args) == 1 and isinstance(args[0], TFrame):
            args[0]._copy_attrs(self)
        self.freq = freq
        self.fmt = fmt
        if self.fmt == 'num':
            self[:] = self[:] * .01
        self.index = _add_freq(self, self.freq)

    # TODO: self.freq could be None here (still) even if index hasfreq

    def _copy_attrs(self, obj):
        for attr in self._metadata:
            obj.__dict__[attr] = getattr(self, attr, None)

    @property
    def _constructor(self):
        # Used when a manipulation result has the same dimesions
        # as the original.
        def f(*args, **kwargs):
            obj = TFrame(*args, **kwargs)
            self._copy_attrs(obj)
            return obj
        return f

    @property
    def _constructor_sliced(self):
        # Used when a manipulation result has one lower dimension(s)
        # as the original, such as DataFrame single columns slicing.
        return TSeries

    # New methods
    # -----------------------------------------------------------------

    def ret_rels(self):
        """Return-relatives, 1+r."""
        return self.add(1.)

    def ret_idx(self, base=1.0):
        """Return index with starting value of `base`."""
        return self.ret_rels().cumprod(axis=0).mul(base)

    def cum_ret_idx(self):
        """Cumulative return index."""
        return self.ret_idx().sub(1.)

    def cum_ret(self):
        """Cumulative return, a scalar for each security."""
        return self.ret_rels().prod().sub(1.)

    def geomean(self):
        """Geometric mean return, a scalar for each security."""
        return self.ret_rels().prod().pow(1./self.count()).sub(1.)

    def anlzd_ret(self):
        """Annualized return."""
        start = self.index[0] - 1
        end = self.index[-1]
        td = end - start
        n = (td.days - 1) / 365.
        return self.ret_rels().prod().pow(1./n).sub(1.)

    def drawdown_idx(self):
        """Drawdown index."""
        ri = self.ret_idx()
        return ri.div(ri.cummax()).sub(1.)

    def max_drawdown(self):
        """Maximum drawdown."""
        return self.drawdown_idx().min()

    def anlzd_std(self, ddof=0):
        n = utils.convertfreq(self.index.freq.freqstr)
        return self.std(ddof=ddof).mul(np.sqrt(n))

    def rollup(self, freq):
        """Downsample returns through geometric linking."""
        return self.resample(freq).apply(lambda f: TFrame(f).cum_ret())

    def drawdown_end(self):
        """The trough date at which drawdown was most negative."""
        return self.drawdown_idx().idxmin()

    def drawdown_start(self):
        # From: @cᴏʟᴅsᴘᴇᴇᴅ
        dd = self.drawdown_idx()
        mask = (dd == dd.min()).cumsum().astype(bool)
        return dd.mask(mask)[::-1].idxmax()

    def recov_date(self):
        """Date after drawdown trough at which previous HWM reached."""
        dd = self.drawdown_idx()
        mask = (dd != dd.min()).cumprod().astype(bool)
        res = dd.mask(mask).eq(0.).idxmax()

        # If all values are False (recovery has not occured),
        #     we need a 2nd mask or else recov date will be
        #     incorrectly recognized as index[0].
        # See `idxmax` docs.
        newmask = res.eq(self.index[0])
        return res.mask(newmask)

    def drawdown_length(self):
        """Peak to trough length."""
        return self.drawdown_end().sub(self.drawdown_start())

    def drawdown_recov(self):
        """Period length of recovery from drawdown trough."""
        return self.recov_date().sub(self.drawdown_end())

    def _mkt_filter(self, benchmark, threshold, op, incl_bm=False):
        # Note: this drops; it is *not* an NaN mask
        benchmark = _try_to_squeeze(benchmark)
        if not isinstance(benchmark, (TFrame, TSeries)):
            raise ValueError
        op = getattr(benchmark, op)
        mask = op(threshold).values
        if not incl_bm:
            return self.loc[mask]
        else:
            return (self.loc[mask], benchmark.loc[mask])

    def upmkt_filter(self, benchmark, threshold=0.0, op='ge', incl_bm=False):
        return self._mkt_filter(benchmark=benchmark, threshold=threshold,
                                op=op, incl_bm=incl_bm)

    def downmkt_filter(self, benchmark, threshold=0.0, op='le', incl_bm=False):
        return self._mkt_filter(benchmark=benchmark, threshold=threshold,
                                op=op, incl_bm=incl_bm)

    def pct_positive(self, threshold=0.0):
        return self[self.gt(threshold)].count().div(self.count())

    def pct_negative(self, threshold=0.0):
        return self[self.lt(threshold)].count().div(self.count())

    def excess_ret(self, benchmark):
        benchmark = _try_to_squeeze(benchmark)
        return self.sub(benchmark, axis=0)

    def tracking_err(self, benchmark, ddof=0):
        er = self.excess_ret(benchmark=benchmark)
        return er.anlzd_std(ddof=ddof)

    def up_capture(self, benchmark, threshold=0.0, op='ge'):
        """Uses anlzd (geometric) return."""
        slf, bm = self.upmkt_filter(benchmark=benchmark, threshold=threshold,
                                    op=op, incl_bm=True)
        # bm is now `_Returns`
        return slf.geomean().div(bm.geomean())

    def down_capture(self, benchmark, threshold=0.0, op='ge'):
        """Uses anlzd (geometric) return."""
        slf, bm = self.downmkt_filter(benchmark=benchmark, threshold=threshold,
                                      op=op, incl_bm=True)
        return slf.geomean().div(bm.geomean())

    def capture_ratio(self, benchmark, threshold=0.0):
        uc = self.up_capture(benchmark=benchmark, threshold=threshold, op='ge')
        dc = self.down_capture(benchmark=benchmark, threshold=threshold,
                               op='ge')
        return uc.div(dc)

    @functools.lru_cache(maxsize=None)
    def _get_rf(self):
        from pyfinance import datasets  # noqa
        # TODO: careful with self.freq, could be None
        # TODO: load_rf calls `rollup`, circularity
        rf = _try_to_squeeze(TFrame(datasets.load_rf(freq=self.freq).dropna()))
        return rf

    def sharpe(self, ddof=0):
        """Sharpe ratio."""
        return self.anlzd_ret().sub(self._get_rf())\
            .div(self.anlzd_std(ddof=ddof))

    def calmar(self):
        """Calmar ratio."""
        return self.anlzd_ret() / self.max_drawdown().abs()

    def batting_avg(self, benchmark, window, **kwargs):
        """Outperformed in `x` pct. of rolling periods."""
        slf = self.ret_rels().rolling(window=window, **kwargs).\
            apply(np.prod).dropna()
        bm = benchmark.ret_rels().rolling(window=window, **kwargs).\
            apply(np.prod).dropna()
        return slf.gt(bm, axis=0).sum() / slf.count()

    # Incomplete
    # -----------------------------------------------------------------

    def excess_drawdown_idx(self, benchmark, **kwargs):
        raise NotImplementedError

    def sortino(self, threshold=0.0, ddof=0):
        raise NotImplementedError
        std = self.semi_std(threshold=threshold, ddof=ddof)
        return self.anlzd_ret().sub(threshold).div(std)

    def semi_std(self, threshold=0.0, ddof=0):
        # sqrt( sum([min(r_i - thresh, 0] **2 ) / (n - ddof) )
        # TODO: thresh is *not* assumed to be anlzd here
        # TODO: anlz this stdev
        raise NotImplementedError
        return self.sub(threshold).clip_lower(0.).pow(2.).sum()\
            .div(self.count().sub(ddof)).pow(0.5)

    def ulcer_idx(self):
        raise NotImplementedError

    def msquared(self):
        raise NotImplementedError

    def rolling_factory(self, **kwargs):
        # Just some braintorming...
        # Would we need to convert each subframe to Returns?
        # Keep in mind a rolling object is different than groupby.
        # rolling = self.rolling(**kwargs)\
        #     .apply(lambda f: Returns(f).somemethod()) ...
        raise NotImplementedError

    # OLS regression/factor loadings
    # ----------------

    def beta(self, benchmark):
        raise NotImplementedError

    def alpha(self, benchmark):
        raise NotImplementedError

    def tstat_beta(self, benchmark):
        raise NotImplementedError

    def tstat_alpha(self, benchmark):
        raise NotImplementedError

    def rsq(self, benchmark):
        raise NotImplementedError

    def rsq_adj(self, benchmark):
        raise NotImplementedError


def _add_freq(idx, freq=None, inplace=False):
    """Ensure Index of the DataFrame or Series has a frequency.

    We can't do many time-aware calculations within a valid Index freq.

    Rule hierarchy:
    1. If `idx` is not a DatetimeIndex or PeriodIndex, return it unchanged.
    2. If `idx` already has a frequency, do nothing.
    3. If a frequency is explicitly passed, use it.
    4. If no frequency is passed, attempt to infer.
    5. If (2) and (3) fail, raise.

    Parameters
    ----------
    idx : pd.Index-like
    freq : str
    inplace : bool
        If True, modify `idx` inplace and return None.  Otherwise,
        return modified copy.

    Example
    -------
    >>> import pandas as pd
    >>> idx = pd.Index([pd.datetime(2000, 1, 2),
                        pd.datetime(2000, 1, 3),
                        pd.datetime(2000, 1, 4)])
    >>> idx.freq is None
    True
    >>> _add_freq(idx).freq
    <Day>
    >>> type(_add_freq(idx).freq)
    pandas.tseries.offsets.Day
    """

    if not inplace:
        idx = idx.copy()
    if not isinstance(idx, (pd.DatetimeIndex, pd.PeriodIndex)):
        # Nothing more we can do.
        if not inplace:
            return idx
    if idx.freq is not None:
        freq = idx.freq
        if not inplace:
            return idx
    else:
        if freq is None:
            # First try to infer.  This will raise ValueError if fails.
            freq = pd.infer_freq(idx)  # = str
        # Now convert this to a Pandas offset.
        freq = pd.tseries.frequencies.to_offset(freq)
    idx.freq = freq
    if not inplace:
        return idx


def _try_to_squeeze(obj, raise_=False):
    """Attempt to squeeze to 1d Series."""
    if isinstance(obj, pd.Series):
        return obj
    elif isinstance(obj, pd.DataFrame) and obj.shape[-1] == 1:
        return obj.squeeze()
    else:
        if raise_:
            raise ValueError('Input cannot be squeezed.')
        return obj


def _check_fmt(obj, fmt):
    """Make sure `fmt` makes sense based on values of `obj`."""
    raise NotImplementedError
