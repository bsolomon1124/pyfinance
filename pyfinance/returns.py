"""Statistical analysis of security returns time series.

Intended to mimic functionality of commercial software such as FactSet,
Axioma, & Zephyr and open-source software such as Pyfolio and ffn.

The main classes of this module are

- TSeries, a subclassed Pandas Series, and
- TFrame, a subclassed Pandas DataFrame.

They implement a collection of new methods that pertain specifically to
investment management and the study of security returns and asset
performance, such cumulative return indices and drawdown.

Note
----
Subclassing Pandas objects is a delicate operation, but in this case
is strictly preferrable to composition or piping.

.. _Subclassing Pandas Data Structures:
    https://pandas.pydata.org/pandas-docs/stable/internals.html
"""

__all__ = ['TSeries', 'TFrame']
__author__ = 'Brad Solomon <brad.solomon.1124@gmail.com>'

import copy
from numbers import Number

import numpy as np
from numpy.lib.nanfunctions import (nanmin,
                                    nansum,
                                    nanprod,
                                    nancumsum,
                                    nancumprod,
                                    nanmean,
                                    nanstd)
import pandas as pd

from pyfinance import ols, utils

SECS_PER_CAL_YEAR = 365.25 * 24 * 60 * 60


class TSeries(pd.Series):
    # __doc__ is set after class definition.
    # See bottom of this file or `help(TSeries)`.

    def __init__(self, *args, **kwargs):
        args = tuple(copy.deepcopy(arg) for arg in args)
        freq = kwargs.pop('freq', None)
        super().__init__(*args, **kwargs)
        self.freq = freq

        # Hold off on inferring a frequency until method calls.
        # Otherwise, this routine is run unncessarily because
        # __init__ gets called somewhat frequently
        # (with __repr__, for instance).

        if not self.index.is_monotonic_increasing:
            raise FrequencyError('Input Index should be sorted.')

    @property
    def _constructor(self):
        return TSeries

    @property
    def _constructor_expanddim(self):
        return TFrame

    # We don't need _constructor_sliced.
    # It raises NotImplementedError by default.

    # "New" public methods
    # -----------------------------------------------------------------

    # Prefer operators to their corresponding Pandas methods
    # so that we can handle NumPy arrays within transformations.
    # "/" is better than x.div(y).

    def alpha(self, benchmark, **kwargs):
        """CAPM alpha.

        The return on an asset in excess of the asset’s required rate
        of return; the risk-adjusted return.
        [Source: CFA Institute]

        Parameters
        ----------
        benchmark : {pd.Series, TSeries, pd.DataFrame, np.ndarray}
            The benchmark securitie(s) to which `self` is compared.
        **kwargs
            Passed to pyfinance.ols.OLS().

        Returns
        -------
        float
        """

        return self.CAPM(benchmark, **kwargs).alpha

    def anlzd_ret(self, freq=None):
        """Annualized (geometric) return.

        Parameters
        ----------
        freq : str or None, default None
            A frequency string used to create an annualization factor.
            If None, `self.freq` will be used.  If that is also None,
            a frequency will be inferred.  If none can be inferred,
            an exception is raised.

            It may be any frequency string or anchored offset string
            recognized by Pandas, such as 'D', '5D', 'Q', 'Q-DEC', or
            'BQS-APR'.

        Returns
        -------
        float
        """

        if self.index.is_all_dates:
            # TODO: Could be more granular here,
            #       for cases with < day frequency.
            td = self.index[-1] - self.index[0]
            n = td.total_seconds() / SECS_PER_CAL_YEAR
        else:
            # We don't have a datetime-like Index, so assume
            # periods/dates are consecutive and simply count them.
            # We do, however, need an explicit frequency.
            freq = freq if freq is not None else self.freq
            if freq is None:
                raise FrequencyError('Must specify a `freq` when a'
                                     ' datetime-like index is not used.')

            n = len(self) / utils.get_anlz_factor(freq)
        return nanprod(self.ret_rels()) ** (1. / n) - 1.

    def anlzd_stdev(self, ddof=0, freq=None, **kwargs):
        """Annualized standard deviation with `ddof` degrees of freedom.

        Parameters
        ----------
        ddof : int, default 0
            Degrees of freedom, passed to pd.Series.std().
        freq : str or None, default None
            A frequency string used to create an annualization factor.
            If None, `self.freq` will be used.  If that is also None,
            a frequency will be inferred.  If none can be inferred,
            an exception is raised.

            It may be any frequency string or anchored offset string
            recognized by Pandas, such as 'D', '5D', 'Q', 'Q-DEC', or
            'BQS-APR'.
        **kwargs
            Passed to pd.Series.std().
        TODO: freq

        Returns
        -------
        float
        """

        if freq is None:
            freq = self._try_get_freq()
            if freq is None:
                raise FrequencyError(msg)
        return nanstd(self, ddof=ddof) * freq ** 0.5

    def batting_avg(self, benchmark):
        """Percentage of periods when `self` outperformed `benchmark`.

        Parameters
        ----------
        benchmark : {pd.Series, TSeries, 1d np.ndarray}
            The benchmark security to which `self` is compared.

        Returns
        -------
        float
        """

        diff = self.excess_ret(benchmark)
        return np.count_nonzero(diff > 0.) / diff.count()

    def beta(self, benchmark, **kwargs):
        """CAPM beta.

        A measure of systematic risk that is based on the covariance
        of an asset's or portfolio's return with the return of the
        overall market; a measure of the sensitivity of a given
        investment or portfolio to movements in the overall market.
        [Source: CFA Institute]

        Parameters
        ----------
        benchmark : {pd.Series, TSeries, pd.DataFrame, np.ndarray}
            The benchmark securitie(s) to which `self` is compared.

        Returns
        -------
        float or np.ndarray
            If `benchmark` is 1d, returns a scalar.
            If `benchmark` is 2d, returns a 1d ndarray.
        """

        return self.CAPM(benchmark, **kwargs).beta

    def beta_adj(self, benchmark, adj_factor=2/3, **kwargs):
        """Adjusted beta.

        Beta that is adjusted to reflect the tendency of beta to
        be mean reverting.
        [Source: CFA Institute]

        Formula:
        adj_factor * raw_beta + (1 - adj_factor)

        Parameters
        ----------
        benchmark : {pd.Series, TSeries, pd.DataFrame, np.ndarray}
            The benchmark securitie(s) to which `self` is compared.

        Returns
        -------
        float or np.ndarray
            If `benchmark` is 1d, returns a scalar.
            If `benchmark` is 2d, returns a 1d ndarray.

        Reference
        ---------
        .. _Blume, Marshall.  "Betas and Their Regression Tendencies."
            http://www.stat.ucla.edu/~nchristo/statistics417/blume_betas.pdf
        """

        beta = self.beta(benchmark=benchmark, **kwargs)
        return adj_factor * beta + (1 - adj_factor)

    def calmar_ratio(self):
        """Calmar ratio--return per unit of drawdown.

        The compound annualized rate of return over a specified
        time period divided by the absolute value of maximum
        drawdown over the same time period.
        [Source: CFA Institute]

        Returns
        -------
        float
        """

        return self.anlzd_ret() / np.abs(self.max_drawdown())

    def capture_ratio(self, benchmark, threshold=0., compare_op=('ge', 'lt')):
        """Capture ratio--ratio of upside to downside capture.

        Upside capture ratio divided by the downside capture ratio.

        Parameters
        ----------
        benchmark : {pd.Series, TSeries, 1d np.ndarray}
            The benchmark security to which `self` is compared.
        threshold : float, default 0.
            The threshold at which the comparison should be done.
            `self` and `benchmark` are "filtered" to periods where
            `benchmark` is greater than/less than `threshold`.
        compare_op : {tuple, str, list}, default ('ge', 'lt')
            Comparison operator used to compare to `threshold`.
            If a sequence, the two elements are passed to
            `self.up_capture()` and `self.down_capture()`, respectively.
            If `str`, indicates the comparison operater used in
            both method calls.

        Returns
        -------
        float
        """

        if isinstance(compare_op(tuple, list)):
            op1, op2 = compare_op
        else:
            op1, op2 = compare_op, compare_op
        uc = self.up_capture(benchmark=benchmark, threshold=threshold,
                             compare_op=op1)
        dc = self.down_capture(benchmark=benchmark, threshold=threshold,
                               compare_op=op2)
        return uc / dc

    def cuml_ret(self):
        """Cumulative return--a scalar ending percentage return.

        Returns
        -------
        float
        """

        return self.growth_of_x() - 1.

    def cuml_idx(self):
        """Cumulative return index--a TSeries of cumulative returns.

        Returns
        -------
        TSeries
        """

        return self.ret_idx() - 1.

    def down_capture(self, benchmark, threshold=0., compare_op='lt'):
        """Downside capture ratio.

        Measures the performance of `self` relative to benchmark
        conditioned on periods where `benchmark` is lt or le to
        `threshold`.

        Downside capture ratios are calculated by taking the fund's
        monthly return during the periods of negative benchmark
        performance and dividing it by the benchmark return.
        [Source: CFA Institute]

        Parameters
        ----------
        benchmark : {pd.Series, TSeries, 1d np.ndarray}
            The benchmark security to which `self` is compared.
        threshold : float, default 0.
            The threshold at which the comparison should be done.
            `self` and `benchmark` are "filtered" to periods where
            `benchmark` is lt/le `threshold`.
        compare_op : {'lt', 'le'}
            Comparison operator used to compare to `threshold`.
            'lt' is less-than; 'le' is less-than-or-equal.

        Returns
        -------
        float

        Note
        ----
        This metric uses geometric, not arithmetic, mean return.
        """

        slf, bm = self.downmarket_filter(benchmark=benchmark,
                                         threshold=threshold,
                                         compare_op=compare_op,
                                         include_benchmark=True)
        return slf.geomean() / bm.geomean()

    def downmarket_filter(self, benchmark, threshold=0., compare_op='lt',
                          include_benchmark=False):
        """Drop elementwise samples where `benchmark` > `threshold`.

        Filters `self` (and optionally, `benchmark`) to periods
        where `benchmark` < `threshold`.  (Or <= `threshold`.)

        Parameters
        ----------
        benchmark : {pd.Series, TSeries, 1d np.ndarray}
            The benchmark security to which `self` is compared.
        threshold : float, default 0.0
            The threshold at which the comparison should be done.
            `self` and `benchmark` are "filtered" to periods where
            `benchmark` is lt/le `threshold`.
        compare_op : {'lt', 'le'}
            Comparison operator used to compare to `threshold`.
            'lt' is less-than; 'le' is less-than-or-equal.
        include_benchmark : bool, default False
            If True, return tuple of (`self`, `benchmark`) both
            filtered.  If False, return only `self` filtered.

        Returns
        -------
        TSeries or tuple of TSeries
            TSeries if `include_benchmark=False`, otherwise, tuple.
        """

        return self._mkt_filter(benchmark=benchmark, threshold=threshold,
                                compare_op=compare_op,
                                include_benchmark=include_benchmark)

    def drawdown_end(self, return_date=False):
        """The date of the drawdown trough.

        Date at which the drawdown was most negative.

        Parameters
        ----------
        return_date : bool, default False
            If True, return a `datetime.date` object.
            If False, return a Pandas Timestamp object.

        Returns
        -------
        datetime.date or pandas._libs.tslib.Timestamp
        """

        end = self.drawdown_idx().idxmin()
        if return_date:
            return end.date()
        return end

    def drawdown_idx(self):
        """Drawdown index; TSeries of drawdown from running HWM.

        Returns
        -------
        TSeries
        """

        ri = self.ret_idx()
        return ri / np.maximum(ri.cummax(), 1.) - 1.

    def drawdown_length(self, return_int=False):
        """Length of drawdown in days.

        This is the duration from peak to trough.

        Parameters
        ----------
        return_int : bool, default False
            If True, return the number of days as an int.
            If False, return a Pandas Timedelta object.

        Returns
        -------
        int or pandas._libs.tslib.Timedelta
        """

        td = self.drawdown_end() - self.drawdown_start()
        if return_int:
            return td.days
        return td

    def drawdown_recov(self, return_int=False):
        """Length of drawdown recovery in days.

        This is the duration from trough to recovery date.

        Parameters
        ----------
        return_int : bool, default False
            If True, return the number of days as an int.
            If False, return a Pandas Timedelta object.

        Returns
        -------
        int or pandas._libs.tslib.Timedelta
        """

        td = self.recov_date() - self.drawdown_end()
        if return_int:
            return td.days
        return td

    def drawdown_start(self, return_date=False):
        """The date of the peak at which most severe drawdown began.

        Parameters
        ----------
        return_date : bool, default False
            If True, return a `datetime.date` object.
            If False, return a Pandas Timestamp object.

        Returns
        -------
        datetime.date or pandas._libs.tslib.Timestamp
        """

        # Thank you @cᴏʟᴅsᴘᴇᴇᴅ
        # https://stackoverflow.com/a/47892766/7954504
        dd = self.drawdown_idx()
        mask = nancumsum(dd == nanmin(dd.min)).astype(bool)
        start = dd.mask(mask)[::-1].idxmax()
        if return_date:
            return start.date()
        return start

    def excess_drawdown_idx(self, benchmark, method='caer'):
        """Excess drawdown index; TSeries of excess drawdowns.

        There are several ways of computing this metric.  For highly
        volatile returns, the `method` specified will have a
        non-negligible effect on the result.

        Parameters
        ----------
        benchmark : {pd.Series, TSeries, 1d np.ndarray}
            The benchmark security to which `self` is compared.
        method : {'caer' (0), 'cger' (1), 'ecr' (2), 'ecrr' (3)}
            Indicates the methodology used.
        """

        # TODO: plot these (compared) in docs.
        if isinstance(method, (int, float)):
            method = ['caer', 'cger', 'ecr', 'ecrr'][method]
        method = method.lower()

        if method == 'caer':
            er = self.excess_ret(benchmark=benchmark, method='arithmetic')
            return er.drawdown_idx()

        elif method == 'cger':
            er = self.excess_ret(benchmark=benchmark, method='geometric')
            return er.drawdown_idx()

        elif method == 'ecr':
            er = self.ret_idx() - benchmark.ret_idx() + 1
            if er.isnull().any():
                return er / er.cummax() - 1.
            else:
                return er / np.maximum.accumulate(er) - 1.

        elif method == 'ecrr':
            # Credit to: SO @piRSquared
            # https://stackoverflow.com/a/36848867/7954504
            p = self.ret_idx().values
            b = benchmark.ret_idx().values
            er = p - b
            if er.isnull().any():
                # The slower route but NaN-friendly.
                cam = self.expanding(min_periods=1).apply(
                    lambda x: x.argmax())
            else:
                cam = utils.cumargmax(er)
            p0 = p[cam]
            b0 = b[cam]
            return (p * b0 - b * p0) / (p0 * b0)

        else:
            raise ValueError("`method` must be one of"
                             " ('caer', 'cger', 'ecr', 'ecrr'),"
                             " case-insensitive, or"
                             " an integer mapping to these methods"
                             " (1 thru 4).")

    def excess_ret(self, benchmark, method='arithmetic'):
        """Excess return.

        Parameters
        ----------
        benchmark : {pd.Series, TSeries, 1d np.ndarray}
            The benchmark security to which `self` is compared.
        method : {{'arith', 'arithmetic'}, {'geo', 'geometric'}}
            The methodology used.  An arithmetic excess return is a
            straightforward subtraction.  A geometric excess return
            is the ratio of return-relatives of `self` to `benchmark`,
            minus one.

        Also known as: active return.

        Reference
        ---------
        .. _Essex River Analytics - A Case for Arithmetic Attribution
            http://www.northinfo.com/documents/563.pdf

        .. _Bacon, Carl.  Excess Returns - Arithmetic or Geometric?
            https://www.cfapubs.org/doi/full/10.2469/dig.v33.n1.1235
        """

        if method.startswith('arith'):
            return self - _try_to_squeeze(benchmark)
        elif method.startswith('geo'):
            # Geometric excess return,
            # (1 + `self`) / (1 + `benchmark`) - 1.
            return self.ret_rels() / _try_to_squeeze(benchmark).ret_rels() - 1.

    def gain_to_loss_ratio(self):
        """Gain-to-loss ratio, ratio of positive to negative returns.

        Formula:
        (n pos. / n neg.) * (avg. up-month return / avg. down-month return)
        [Source: CFA Institute]

        Returns
        -------
        float
        """

        gt = self > 0
        lt = self < 0
        return (nansum(gt) / nansum(lt)) * (self[gt].mean() / self[lt].mean())

    def geomean(self):
        """Geometric mean return over the entire sample.

        Returns
        -------
        float
        """

        return nanprod(self.ret_rels()) ** (1. / self.count()) - 1.

    def growth_of_x(self, x=1.):
        """Ending value from growth of `x`, a scalar.

        Returns
        -------
        float
        """

        return nanprod(self.ret_rels()) * x

    def info_ratio(self, benchmark, ddof=0):
        """Information ratio--return per unit of active risk.

        Parameters
        ----------
        benchmark : {pd.Series, TSeries, 1d np.ndarray}
            The benchmark security to which `self` is compared.
        ddof : int, default 0
            Degrees of freedom, passed to pd.Series.std().

        Returns
        -------
        float
        """

        diff = self.excess_ret(benchmark).anlzd_return()
        return diff / self.tracking_error(benchmark, ddof=ddof)

    def max_drawdown(self):
        """Most negative drawdown derived from drawdown index.

        The largest difference between a high-water point and a
        subsequent low.  A portfolio may also be said to be in a
        position of drawdown from a decline from a high-water mark
        until a new high-water mark is reached.
        [Source: CFA Institute]

        Returns
        -------
        float
        """

        return nanmin(self.drawdown_idx())

    def msquared(self, benchmark, rf=0.02, ddof=0):
        """M-squared, return scaled by relative total risk.

        A measure of what a portfolio would have returned if it had
        taken on the same *total* risk as the market index.
        [Source: CFA Institute]

        Parameters
        ----------
        benchmark : {pd.Series, TSeries, 1d np.ndarray}
            The benchmark security to which `self` is compared.
        rf : {float, TSeries, pd.Series}, default 0.02
            If float, this represents an *compounded annualized*
            risk-free rate; 2.0% is the default.
            If a TSeries or pd.Series, this represents a time series
            of periodic returns to a risk-free security.

            To download a risk-free rate return series using
            3-month US T-bill yields, see:`pyfinance.datasets.load_rf`.
        ddof : int, default 0
            Degrees of freedom, passed to pd.Series.std().

        Returns
        -------
        float
        """

        rf = self._validate_rf(rf)
        scaling = benchmark.anlzd_stdev(ddof) / self.anlzd_stdev(ddof)
        diff = self.anlzd_ret() - rf
        return rf + diff * scaling

    def pct_negative(self, threshold=0.):
        """Pct. of periods in which `self` is less than `threshold.`

        Parameters
        ----------
        threshold : {float, TSeries, pd.Series}, default 0.

        Returns
        -------
        float
        """

        return np.count_nonzero(self[self < threshold]) / self.count()

    def pct_positive(self, threshold=0.):
        """Pct. of periods in which `self` is greater than `threshold.`

        Parameters
        ----------
        threshold : {float, TSeries, pd.Series}, default 0.

        Returns
        -------
        float
        """

        return np.count_nonzero(self[self > threshold]) / self.count()

    def recov_date(self, return_date=False):
        """Drawdown recovery date.

        Date at which `self` recovered to previous high-water mark.

        Parameters
        ----------
        return_date : bool, default False
            If True, return a `datetime.date` object.
            If False, return a Pandas Timestamp object.

        Returns
        -------
        {datetime.date, pandas._libs.tslib.Timestamp, pd.NaT}
            Returns NaT if recovery has not occured.
        """

        dd = self.drawdown_idx()
        # False beginning on trough date and all later dates.
        mask = nancumprod(dd != nanmin(dd)).astype(bool)
        res = dd.mask(mask) == 0

        # If `res` is all False (recovery has not occured),
        # .idxmax() will return `res.index[0]`.
        if not res.any():
            recov = pd.NaT
        else:
            recov = res.idxmax()

        if return_date:
            return recov.date()
        return recov

    def ret_idx(self, base=1.):
        """Return index, a time series starting at `base`.

        Parameters
        ----------
        base : float, default 1.

        Returns
        -------
        float
        """

        return self.ret_rels().cumprod() * base

    def ret_rels(self):
        """Return-relatives, `1 + self`.

        A growth factor that is simply `1 + self`, elementwise.
        This method exists purely for naming purposes.

        Returns
        -------
        TSeries
        """
        return self + 1.

    def rollup(self, freq, **kwargs):
        """Downsample `self` through geometric linking.

        Parameters
        ----------
        freq : {'D', 'W', 'M', 'Q', 'A'}
            The frequency of the result.
        **kwargs
            Passed to `self.resample()`.

        Returns
        -------
        TSeries

        Example
        -------
        # Derive quarterly returns from monthly returns.
        >>> import numpy as np
        >>> from pyfinance import TSeries
        >>> np.random.seed(444)
        >>> ts = TSeries(np.random.randn(12) / 100 + 0.002,
        ...              index=pd.date_range('2016', periods=12, freq='M'))
        >>> ts.rollup('Q')
        2016-03-31    0.0274
        2016-06-30   -0.0032
        2016-09-30   -0.0028
        2016-12-31    0.0127
        Freq: Q-DEC, dtype: float64
        """

        return self.ret_rels().resample(freq, **kwargs).prod() - 1.

    def rsq(self, benchmark, **kwargs):
        """R-squared.

        The fraction of the total variation that is explained
        by the regression.
        [Source: CFA Institute]

        Also known as: coefficient of determination.

        Parameters
        ----------
        benchmark : {pd.Series, TSeries, pd.DataFrame, np.ndarray}
            The benchmark securitie(s) to which `self` is compared.
        **kwargs
            Passed to pyfinance.ols.OLS().

        Returns
        -------
        float
        """

        return self.CAPM(benchmark, **kwargs).rsq

    def rsq_adj(self, benchmark, **kwargs):
        """Adjusted R-squared.

        Formula: 1. - ((1. - `self._rsq`) * (n - 1.) / (n - k - 1.))

        Parameters
        ----------
        benchmark : {pd.Series, TSeries, pd.DataFrame, np.ndarray}
            The benchmark securitie(s) to which `self` is compared.
        **kwargs
            Passed to pyfinance.ols.OLS().

        Returns
        -------
        float
        """
        return self.CAPM(benchmark, **kwargs).rsq_adj

    def semi_stdev(self, threshold=0., ddof=0, freq=None):
        """Semi-standard deviation; stdev of downside returns.

        It is designed to address that fact that plain standard
        deviation penalizes "upside volatility.""

        Formula: `sqrt( sum([min(self - thresh, 0] **2 ) / (n - ddof) )`

        Also known as: downside deviation.

        Parameters
        ----------
        threshold : {float, TSeries, pd.Series}, default 0.
            While zero is the default, it is also customary to use
            a "minimum acceptable return" (MAR) or a risk-free rate.
            Note: this is assumed to be a *periodic*, not necessarily
            annualized, return.
        ddof : int, default 0
            Degrees of freedom, passed to pd.Series.std().
        freq : str or None, default None
            A frequency string used to create an annualization factor.
            If None, `self.freq` will be used.  If that is also None,
            a frequency will be inferred.  If none can be inferred,
            an exception is raised.

            It may be any frequency string or anchored offset string
            recognized by Pandas, such as 'D', '5D', 'Q', 'Q-DEC', or
            'BQS-APR'.

        Returns
        -------
        float
        """

        if freq is None:
            freq = self._try_get_freq()
            if freq is None:
                raise FrequencyError(msg)
        n = self.count() - ddof
        ss = (nansum(np.minimum(self - threshold, 0.) ** 2) ** 0.5) / n
        return ss * freq ** 0.5

    def sharpe_ratio(self, rf=0.02, ddof=0):
        """Return over `rf` per unit of total risk.

        The average return in excess of the risk-free rate divided
        by the standard deviation of return; a measure of the average
        excess return earned per unit of standard deviation of return.
        [Source: CFA Institute]

        Parameters
        ----------
        rf : {float, TSeries, pd.Series}, default 0.02
            If float, this represents an *compounded annualized*
            risk-free rate; 2.0% is the default.
            If a TSeries or pd.Series, this represents a time series
            of periodic returns to a risk-free security.

            To download a risk-free rate return series using
            3-month US T-bill yields, see:`pyfinance.datasets.load_rf`.
        ddof : int, default 0
            Degrees of freedom, passed to pd.Series.std().

        Returns
        -------
        float
        """

        rf = self._validate_rf(rf)
        stdev = self.anlzd_stdev(ddof=ddof)
        return (self.anlzd_ret() - rf) / stdev

    def sortino_ratio(self, threshold=0., ddof=0, freq=None):
        """Return over a threshold per unit of downside deviation.

        A performance appraisal ratio that replaces standard deviation
        in the Sharpe ratio with downside deviation.
        [Source: CFA Institute]

        Parameters
        ----------
        threshold : {float, TSeries, pd.Series}, default 0.
            While zero is the default, it is also customary to use
            a "minimum acceptable return" (MAR) or a risk-free rate.
            Note: this is assumed to be a *periodic*, not necessarily
            annualized, return.
        ddof : int, default 0
            Degrees of freedom, passed to pd.Series.std().
        freq : str or None, default None
            A frequency string used to create an annualization factor.
            If None, `self.freq` will be used.  If that is also None,
            a frequency will be inferred.  If none can be inferred,
            an exception is raised.

            It may be any frequency string or anchored offset string
            recognized by Pandas, such as 'D', '5D', 'Q', 'Q-DEC', or
            'BQS-APR'.

        Returns
        -------
        float
        """

        stdev = self.semi_stdev(threshold=threshold, ddof=ddof, freq=freq)
        return (self.anlzd_ret() - threshold) / stdev

    def tracking_error(self, benchmark, ddof=0):
        """Standard deviation of excess returns.

        The standard deviation of the differences between
        a portfolio's returns and its benchmark's returns.
        [Source: CFA Institute]

        Also known as: tracking risk; active risk

        Parameters
        ----------
        benchmark : {pd.Series, TSeries, 1d np.ndarray}
            The benchmark security to which `self` is compared.
        ddof : int, default 0
            Degrees of freedom, passed to pd.Series.std().

        Returns
        -------
        float
        """

        er = self.excess_ret(benchmark=benchmark)
        return er.anlzd_stdev(ddof=ddof)

    def treynor_ratio(self, benchmark, rf=0.02):
        """Return over `rf` per unit of systematic risk.

        A measure of risk-adjusted performance that relates a
        portfolio's excess returns to the portfolio's beta.
        [Source: CFA Institute]

        Parameters
        ----------
        benchmark : {pd.Series, TSeries, 1d np.ndarray}
            The benchmark security to which `self` is compared.
        rf : {float, TSeries, pd.Series}, default 0.02
            If float, this represents an *compounded annualized*
            risk-free rate; 2.0% is the default.
            If a TSeries or pd.Series, this represents a time series
            of periodic returns to a risk-free security.

            To download a risk-free rate return series using
            3-month US T-bill yields, see:`pyfinance.datasets.load_rf`.

        Returns
        -------
        float
        """

        benchmark = _try_to_squeeze(benchmark)
        if benchmark.ndim > 1:
            raise ValueError('Treynor ratio requires a single benchmark')
        rf = self._validate_rf(rf)
        beta = self.beta(benchmark)
        return (self.anlzd_ret() - rf) / beta

    def tstat_alpha(self, benchmark, **kwargs):
        """The t-statistic of CAPM alpha.

        Parameters
        ----------
        benchmark : {pd.Series, TSeries, pd.DataFrame, np.ndarray}
            The benchmark securitie(s) to which `self` is compared.
        **kwargs
            Passed to pyfinance.ols.OLS().

        Returns
        -------
        float
        """

        return self.CAPM(benchmark, **kwargs).tstat_alpha

    def tstat_beta(self, benchmark, **kwargs):
        """The t-statistic of CAPM beta.

        Parameters
        ----------
        benchmark : {pd.Series, TSeries, pd.DataFrame, np.ndarray}
            The benchmark securitie(s) to which `self` is compared.
        **kwargs
            Passed to pyfinance.ols.OLS().

        Returns
        -------
        float or np.ndarray
            If `benchmark` is 1d, returns a scalar.
            If `benchmark` is 2d, returns a 1d ndarray.
        """

        return self.CAPM(benchmark, **kwargs).tstat_beta

    def ulcer_idx(self):
        """Ulcer Index, a composite measure of drawdown severity.

        A measure of the depth and duration of drawdowns in prices
        from earlier highs.  It is the square root of the mean of
        the squared percentage drawdowns in value.
        [Source: Peter G. Martin]

        Returns
        -------
        float

        References
        ----------
        .. _Peter G. Martin: Ulcer Index
            http://www.tangotools.com/ui/ui.htm
        """

        return nanmean(self.drawdown_idx() ** 2) ** 0.5

    def up_capture(self, benchmark, threshold=0., compare_op='ge'):
        """Upside capture ratio.

        Measures the performance of `self` relative to benchmark
        conditioned on periods where `benchmark` is gt or ge to
        `threshold`.

        Upside capture ratios are calculated by taking the fund's
        monthly return during the periods of positive benchmark
        performance and dividing it by the benchmark return.
        [Source: CFA Institute]

        Parameters
        ----------
        benchmark : {pd.Series, TSeries, 1d np.ndarray}
            The benchmark security to which `self` is compared.
        threshold : float, default 0.
            The threshold at which the comparison should be done.
            `self` and `benchmark` are "filtered" to periods where
            `benchmark` is gt/ge `threshold`.
        compare_op : {'ge', 'gt'}
            Comparison operator used to compare to `threshold`.
            'gt' is greater-than; 'ge' is greater-than-or-equal.

        Returns
        -------
        float

        Note
        ----
        This metric uses geometric, not arithmetic, mean return.
        """

        slf, bm = self.upmarket_filter(benchmark=benchmark,
                                       threshold=threshold,
                                       compare_op=compare_op,
                                       include_benchmark=True)
        return slf.geomean() / bm.geomean()

    def upmarket_filter(self, benchmark, threshold=0., compare_op='ge',
                        include_benchmark=False):
        """Drop elementwise samples where `benchmark` < `threshold`.

        Filters `self` (and optionally, `benchmark`) to periods
        where `benchmark` > `threshold`.  (Or >= `threshold`.)

        Parameters
        ----------
        benchmark : {pd.Series, TSeries, 1d np.ndarray}
            The benchmark security to which `self` is compared.
        threshold : float, default 0.0
            The threshold at which the comparison should be done.
            `self` and `benchmark` are "filtered" to periods where
            `benchmark` is gt/ge `threshold`.
        compare_op : {'ge', 'gt'}
            Comparison operator used to compare to `threshold`.
            'gt' is greater-than; 'ge' is greater-than-or-equal.
        include_benchmark : bool, default False
            If True, return tuple of (`self`, `benchmark`) both
            filtered.  If False, return only `self` filtered.

        Returns
        -------
        TSeries or tuple of TSeries
            TSeries if `include_benchmark=False`, otherwise, tuple.
        """

        return self._mkt_filter(benchmark=benchmark, threshold=threshold,
                                compare_op=compare_op,
                                include_benchmark=include_benchmark)

    # Unfortunately, we can't cache this result because the input
    #     (benchmark) is mutable and not hashable.  And we don't want to
    #     sneak around that, in the off-chance that the input actually
    #     would be altered between calls.
    # One alternative: specify the benchmark at instantiation,
    #     and "bind" self to that.
    # The nice thing is that some properties of the resulting OLS
    #     object *are* cached, so repeated calls are pretty fast.

    def CAPM(self, benchmark, has_const=False, use_const=True):
        """Interface to OLS regression against `benchmark`.

        `self.alpha()`, `self.beta()` and several other methods
        stem from here.  For the full method set, see
        `pyfinance.ols.OLS`.

        Parameters
        ----------
        benchmark : {pd.Series, TSeries, pd.DataFrame, np.ndarray}
            The benchmark securitie(s) to which `self` is compared.
        has_const : bool, default False
            Specifies whether `benchmark` includes a user-supplied constant
            (a column vector).  If False, it is added at instantiation.
        use_const : bool, default True
            Whether to include an intercept term in the model output.  Note the
            difference between `has_const` and `use_const`: the former
            specifies whether a column vector of 1s is included in the
            input; the latter specifies whether the model itself
            should include a constant (intercept) term.  Exogenous
            data that is ~N(0,1) would have a constant equal to zero;
            specify use_const=False in this situation.

        Returns
        -------
        pyfinance.ols.OLS
        """

        return ols.OLS(y=self, x=benchmark, has_const=has_const,
                       use_const=use_const)

    def _mkt_filter(self, benchmark, threshold, compare_op,
                    include_benchmark=False):
        benchmark = _try_to_squeeze(benchmark)
        if not isinstance(benchmark, (TFrame, TSeries)):
            raise ValueError('`benchmark` must be TFrame or TSeries.')
        # Use the dunder method to be NumPy-compatible.
        compare_op = getattr(benchmark, '{0}{1}{0}'.format('__', compare_op))
        mask = compare_op(threshold).values
        if not include_benchmark:
            return self.loc[mask]
        else:
            return self.loc[mask], benchmark.loc[mask]

    def _validate_rf(self, rf):
        if not isinstance(rf, Number):
            if isinstance(rf, np.ndarray):
                if len(rf) != len(self):
                    raise ValueError('When passed as a NumPy array,'
                                     '`rf` must be of equal length'
                                     ' as `self`. to enable alignment.')
                    rf = TSeries(rf, freq=self.freq)
            else:
                rf = rf.reindex(self.index)
                if isinstance(rf, pd.Series):
                    rf = TSeries(rf, freq=self.freq)
                rf = rf.anlzd_ret()
        return rf

    def _try_get_freq(self):
        if self.freq is None:
            freq = pd.infer_freq(self.index)
            if freq is None:
                raise FrequencyError('No frequency was passed at'
                                     ' instantiation, and one cannot'
                                     ' be inferred.')
            freq = utils.get_anlz_factor(freq)
        else:
            freq = utils.get_anlz_factor(self.freq)
        return freq


def _try_to_squeeze(obj, raise_=False):
    """Attempt to squeeze to 1d Series.

    Parameters
    ----------
    obj : {pd.Series, pd.DataFrame}
    raise_ : bool, default False
    """

    if isinstance(obj, pd.Series):
        return obj
    elif isinstance(obj, pd.DataFrame) and obj.shape[-1] == 1:
        return obj.squeeze()
    else:
        if raise_:
            raise ValueError('Input cannot be squeezed.')
        return obj


class FrequencyError(ValueError):
    """Index frequency misspecified, missing, or not inferrable."""
    pass


class TFrame(object):
    def __init__(self):
        raise NotImplementedError


# Default
msg = ('A frequency was not passed to the method or at'
       ' instantiation, and could not otherwise be inferred from the'
       ' Index.')


# ---------------------------------------------------------------------
# Long-form docstring shared between TSeries & TFrame.

doc = """
Time series of periodic returns for {securities}.

Subclass of `pandas.{obj}`, with an extended set of methods
geared towards calculating common investment metrics.

It attempts to optimize for speed, not adding much incremental overhead
on top of the underlying NumPy and Pandas calls.

Parameters
----------
freq : str or None, default None
    `freq` may be passed by *keyword-argument only*.


Methods
-------
{methods}

Examples
--------
{examples}
"""

param = 'Parameters\n        -------'
ret = 'Returns\n        -------'


def _truncate_method_docstring(doc):
    p, r = doc.find(param), doc.find(ret)
    if p == r == -1:
        return doc
    else:
        m = min(p if p != -1 else r, r if r != -1 else p)
        return doc[:m]


methods = (
    'alpha',
    'anlzd_ret',
    'anlzd_stdev',
    'batting_avg',
    'beta',
    'beta_adj',
    'calmar_ratio',
    'capture_ratio',
    'cuml_ret',
    'cuml_idx',
    'down_capture',
    'downmarket_filter',
    'drawdown_end',
    'drawdown_idx',
    'drawdown_length',
    'drawdown_recov',
    'drawdown_start',
    'excess_drawdown_idx',
    'excess_ret',
    'gain_to_loss_ratio',
    'geomean',
    'growth_of_x',
    'info_ratio',
    'max_drawdown',
    'msquared',
    'pct_negative',
    'pct_positive',
    'recov_date',
    'ret_idx',
    'ret_rels',
    'rollup',
    'rsq',
    'rsq_adj',
    'semi_stdev',
    'sharpe_ratio',
    'sortino_ratio',
    'tracking_error',
    'treynor_ratio',
    'tstat_alpha',
    'tstat_beta',
    'ulcer_idx',
    'up_capture',
    'upmarket_filter',
    'CAPM'
    )

tseries_method_doc = ''
for method in methods:
    try:
        mdoc = _truncate_method_docstring(getattr(TSeries, method).__doc__)
        tseries_method_doc += '{} : {}\n\n'.format(method, mdoc.strip())
    except AttributeError:
        # No docstring (yet).
        pass

TSeries.__doc__ = doc.format(securities='a single security',
                             obj='Series',
                             methods=tseries_method_doc.strip(),
                             examples='TODO')

TFrame.__doc__ = doc.format(securities='multiple securities',
                            obj='DataFrame',
                            methods='TODO',
                            examples='TODO')
