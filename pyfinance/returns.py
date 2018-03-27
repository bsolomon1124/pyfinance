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
from numbers import Number

import numpy as np
import pandas as pd
from pandas import tseries

from pyfinance import ols, utils


class TSeries(pd.Series):
    # __doc__ is set after class definition.
    # See bottom of this file or `help(TSeries)`.

    def __init__(self, *args, **kwargs):
        args = tuple(copy.deepcopy(arg) for arg in args)
        freq = kwargs.pop('freq', None)
        super().__init__(*args, **kwargs)

        if freq is None:
            if hasattr(self.index, 'freq'):
                if self.index.freq is not None:
                    self.freq = self.index.freqstr
                else:
                    freq = pd.infer_freq(self.index)
                    if freq is None:
                        data = args[0]
                        if isinstance(data, TSeries):
                            freq = data.freq
                            if freq is None:
                                err = ('A frequency (`freq`) was not passed',
                                       ' and cannot be inferred from the',
                                       ' resulting Index.')
                                raise FrequencyError(err)
                    else:
                        self.freq = freq
                        self.index.freq = tseries.frequencies.to_offset(freq)
            else:
                # We're SOL.
                raise FrequencyError('A frequency (`freq`) was not passed, and'
                                     ' the resulting Index is not'
                                     ' datetime-like.')
        else:
            # The passed parameter takes priority.
            # But, still make sure we're working with a valid index.
            self.freq = freq
            if hasattr(self.index, 'freq'):
                self.index.freq = pd.tseries.frequencies.to_offset(freq)
            else:
                raise FrequencyError('The resulting Index must be'
                                     ' datetime-like.')

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
        return self.CAPM(benchmark, **kwargs).alpha

    def anlzd_ret(self):
        start = self.index[0] - 1
        end = self.index[-1]
        td = end - start
        n = (td.days - 1.) / 365.
        return self.ret_rels().prod() ** (1. / n) - 1.

    def anlzd_stdev(self, ddof=0):
        return self.std(ddof=ddof) * utils.convertfreq(self.freq) ** 0.5

    def batting_avg(self, benchmark):
        diff = self.excess_ret(benchmark)
        return np.count_nonzero(diff > 0.) / diff.count()

    def beta(self, benchmark, **kwargs):
        return self.CAPM(benchmark, **kwargs).beta

    def beta_adj(self, benchmark, adj_factor=2/3, **kwargs):
        """
        .. _Blume, Marshall.  "Betas and Their Regression Tendencies."
            http://www.stat.ucla.edu/~nchristo/statistics417/blume_betas.pdf
        """

        beta = self.beta(benchmark=benchmark, **kwargs)
        return adj_factor * beta + (1 - adj_factor)

    def calmar_ratio(self):
        return self.anlzd_ret() / np.abs(self.max_drawdown())

    def capture_ratio(self, benchmark, threshold=0., compare_op=('ge', 'lt')):
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
        self.growth_of_x() - 1.

    def cuml_idx(self):
        self.ret_idx() - 1.

    def down_capture(self, benchmark, threshold=0., compare_op='lt'):
        # NOTE: Uses geometric return.
        slf, bm = self.downmarket_filter(benchmark=benchmark,
                                         threshold=threshold,
                                         compare_op=compare_op,
                                         include_benchmark=True)
        return slf.geomean() / bm.geomean()

    def downmarket_filter(self, benchmark, threshold=0., compare_op='lt',
                          include_benchmark=False):
        return self._mkt_filter(benchmark=benchmark, threshold=threshold,
                                compare_op=compare_op,
                                include_benchmark=include_benchmark)

    def drawdown_end(self):
        # TODO: call .date(), but how would this work for arrays?
        return self.drawdown_idx().idxmin()

    def drawdown_idx(self):
        ri = self.ret_idx()
        return ri / np.maximum(ri.cummax(), 1.) - 1.

    def drawdown_length(self):
        return self.drawdown_end() - self.drawdown_start()

    def drawdown_recov(self):
        return self.recov_date() - self.drawdown_end()

    def drawdown_start(self):
        # From: @cᴏʟᴅsᴘᴇᴇᴅ
        dd = self.drawdown_idx()
        mask = (dd == dd.min()).cumsum().astype(bool)
        return dd.mask(mask)[::-1].idxmax()

    def excess_drawdown_idx(self, benchmark, method='caer'):
        """
        Parameters
        ----------
        method : {'caer' (0), 'cger' (1), 'ecr' (2), 'ecrr' (3)}
        """

        # TODO: plot these (compared) in docs

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
            return er / np.maximum.accumulate(er) - 1.

        elif method == 'ecrr':
            # Credit to: SO @piRSquared
            # https://stackoverflow.com/a/36848867/7954504
            p = self.ret_idx()
            b = benchmark.ret_idx()
            er = p - b
            cam = utils.cumargmax(er)
            p0 = p.values[cam]
            b0 = b.values[cam]
            return (p * b0 - b * p0) / (p0 * b0)

        else:
            raise ValueError("`method` must be one of"
                             " ('caer', 'cger', 'ecr', 'ecrr'),"
                             " case-insensitive, or"
                             " an integer mapping to these methods"
                             " (1 thru 4).")

    def excess_ret(self, benchmark, method='arithmetic'):
        """
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
        gt = self > 0
        lt = self < 0
        return (np.sum(gt) / np.sum(lt)) * (self[gt].mean() / self[lt].mean())

    def geomean(self):
        self.ret_rels().prod() ** (1. / self.count()) - 1.

    def growth_of_x(self, x=1.0):
        return self.ret_rels().prod() * x

    def info_ratio(self, benchmark, ddof=0):
        # TODO: arithmetic mean versus geomean
        diff = self.anlzd_ret() - benchmark.anlzd_ret()
        return diff / self.tracking_error(benchmark, ddof=ddof)

    def max_drawdown(self):
        return self.drawdown_idx().min()

    def msquared(self, benchmark, rf=0.02, ddof=0):
        rf = self._validate_rf(rf)
        scaling = benchmark.anlzd_stdev(ddof) / self.anlzd_stdev(ddof)
        diff = self.anlzd_ret() - rf
        return rf + diff * scaling

    def pct_negative(self, threshold=0.0):
        return self[self < threshold].count() / self.count()

    def pct_positive(self, threshold=0.0):
        return self[self > threshold].count() / self.count()

    def recov_date(self):
        dd = self.drawdown_idx()
        mask = (dd != dd.min()).cumprod().astype(bool)
        res = dd.mask(mask).eq(0.).idxmax()

        # If all values are False (recovery has not occured),
        #     we need a 2nd mask or else recov date will be
        #     incorrectly recognized as index[0].
        # See `idxmax()` docs.
        newmask = res == self.index[0]
        return res.mask(newmask)

    def ret_idx(self, base=1.):
        return self.ret_rels().cumprod() * base

    def ret_rels(self):
        return self + 1.

    def rollup(self, freq, **kwargs):
        self.ret_rels().resample(freq, **kwargs).prod() - 1.

    def rsq(self, benchmark, **kwargs):
        return self.CAPM(benchmark, **kwargs).rsq

    def rsq_adj(self, benchmark, **kwargs):
        return self.CAPM(benchmark, **kwargs).rsq_adj

    def semi_stdev(self, threshold=0., ddof=0):
        # TODO: see old docs
        # TODO: de-annualize `threshold`
        # sqrt( sum([min(r_i - thresh, 0] **2 ) / (n - ddof) )

        n = self.count() - ddof
        ss = (np.sum(np.minimum(self - threshold, 0.) ** 2) ** 0.5) / n
        return ss * utils.convertfreq(self.freq) ** 0.5

    def sharpe_ratio(self, rf=0.02, ddof=0):
        rf = self._validate_rf(rf)
        stdev = self.anlzd_std(ddof=ddof)
        return (self.anlzd_ret() - rf) / stdev

    def sortino_ratio(self, threshold=0., ddof=0):
        stdev = self.semi_stdev(threshold=threshold, ddof=ddof)
        return (self.anlzd_ret() - threshold) / stdev

    def tracking_error(self, benchmark, ddof=0):
        er = self.excess_ret(benchmark=benchmark)
        return er.anlzd_stdev(ddof=ddof)

    def treynor_ratio(self):
        raise NotImplementedError('ols (and rf)')

    def tstat_alpha(self, benchmark, **kwargs):
        return self.CAPM(benchmark, **kwargs).tstat_alpha

    def tstat_beta(self, benchmark, **kwargs):
        return self.CAPM(benchmark, **kwargs).tstat_beta

    def ulcer_idx(self):
        return np.mean(self.drawdown_idx() ** 2) ** 0.5

    def up_capture(self, benchmark, threshold=0., compare_op='ge'):
        # NOTE: Uses geometric return.
        slf, bm = self.upmarket_filter(benchmark=benchmark,
                                       threshold=threshold,
                                       compare_op=compare_op,
                                       include_benchmark=True)
        return slf.geomean() / bm.geomean()

    def upmarket_filter(self, benchmark, threshold=0., compare_op='ge',
                        include_benchmark=False):
        return self._mkt_filter(benchmark=benchmark, threshold=threshold,
                                compare_op=compare_op,
                                include_benchmark=include_benchmark)

    def _mkt_filter(self, benchmark, threshold, compare_op,
                    include_benchmark=False):
        """Filter self based on `benchmark` performance in same period.
        """

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

    # Unfortunately, we can't cache this result because the input
    #     (benchmark) is mutable and not hashable.  And we don't want to
    #     sneak around that, in the off-chance that the input actually
    #     would be altered between calls.
    # One alternative: specify the benchmark at instantiation,
    #     and "bind" self to that.
    # The nice thing is that some properties of the resulting OLS
    #     object *are* cached, so repeated calls are pretty fast.
    def CAPM(self, benchmark, has_const=False, use_const=True):
        return ols.OLS(y=self, x=benchmark, has_const=has_const,
                       use_const=use_const)


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


# ---------------------------------------------------------------------
# Long-form docstring shared between TSeries & TFrame.

doc = """
Time series of periodic returns for {securities}.

Subclass of `pandas.{obj}`, with an extended set of methods
geared towards calculating common investment metrics.

The main structural difference between `{name}` and a Pandas
{obj} is that {name} *must* have a datetime-like index with a
discernable frequency.  This is enforced strictly during
instantiation.

Parameters
----------
freq : str or None, default None
    `freq` may be passed by *keyword-argument only*.

Methods
-------
alpha : CAPM alpha.
    Definition: "The return on an asset in excess of the asset's
    required rate of return; the risk-adjusted return."
    Source: CFA Institute
ret_rels : Return-relatives.
    A growth factor which is simply `1 + self`, elementwise.
    This method exists purely for naming purposes.

Examples
--------
{examples}
"""

TSeries.__doc__ = doc.format(securities='a single security',
                             obj='Series',
                             name='TSeries',
                             examples='')

TFrame.__doc__ = doc.format(securities='multiple securities',
                            obj='DataFrame',
                            name='TFrame',
                            examples='')
