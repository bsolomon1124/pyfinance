"""Ordinary least-squares (OLS) regression.  Static and rolling cases."""


__author__ = 'Brad Solomon <brad.solomon.1124@gmail.com>'
__all__ = ['OLS', 'RollingOLS', 'PandasRollingOLS']


from functools import lru_cache

import numpy as np
from pandas import DataFrame, Series
import scipy.stats as scs
from statsmodels.tools import add_constant

from pyfinance import utils

# TODO: confidence intervals


def _rolling_lstsq(x, y):
    """Finds solution for the rolling case.  Matrix formulation."""
    return np.squeeze(np.matmul(np.linalg.inv(np.matmul(x.swapaxes(1, 2), x)),
                      np.matmul(x.swapaxes(1, 2), np.atleast_3d(y))))


def _confirm_constant(a):
    """Confirm `a` has volumn vector of 1s."""
    return np.any(np.equal(np.ptp(a, axis=0), 1.))


def _check_constant_params(a, has_const=False, use_const=True, rtol=1e-05,
                           atol=1e-08):
    """Helper func to interaction between has_const and use_const params.

    has_const   use_const   outcome
    ---------   ---------   -------
    True        True        Confirm that a has constant; return a
    False       False       Confirm that a doesn't have constant; return a
    False       True        Confirm that a doesn't have constant; add constant
    True        False       ValueError
    """

    if all((has_const, use_const)):
        if not _confirm_constant(a):
            raise ValueError('Data does not contain a constant; specify'
                             ' has_const=False')
        k = a.shape[-1] - 1
    elif not any((has_const, use_const)):
        if _confirm_constant(a):
            raise ValueError('Data already contains a constant; specify'
                             ' has_const=True')
        k = a.shape[-1]
    elif not has_const and use_const:
        # Also run a quick check to confirm that `a` is *not* ~N(0,1).
        #     In this case, constant should be zero. (exclude it entirely)
        c1 = np.allclose(a.mean(axis=0), b=0., rtol=rtol, atol=atol)
        c2 = np.allclose(a.std(axis=0), b=1., rtol=rtol, atol=atol)
        if c1 and c2:
            # TODO: maybe we want to just warn here?
            raise ValueError('Data appears to be ~N(0,1).  Specify'
                             ' use_constant=False.')
        # `has_constant` does checking on its own and raises VE if True
        a = add_constant(a, has_constant='raise')
        k = a.shape[-1] - 1
    else:
        raise ValueError('`use_const` == False implies has_const is False.')

    return k, a


def _handle_ab(solution, use_const=True):
    b = solution[1:] if use_const else solution
    b = np.asscalar(b) if b.size == 1 else b
    a = solution[0] if use_const else None
    return a, b


def _handle_rolling_ab(solution, use_const=True):
    b = solution[:, 1:] if use_const else solution
    a = solution[:, 0] if use_const else None
    return a, b


def _clean_xy(y, x=None, has_const=False, use_const=True):
    x = np.asanyarray(x) if x is not None else None
    y = np.asanyarray(y)

    # If only `y` is given (and `x=None`), `y` is assumed to be the first
    # column of `y` and `x` the remaining [1:] columns
    if x is None:
        x = y[:, 1:]
        y = y[:, 0]

    k, x = _check_constant_params(x, has_const=has_const,
                                  use_const=use_const)
    y = np.squeeze(y)
    x = np.atleast_2d(x)
    assert y.ndim == 1 and x.ndim > 1
    return x, y, k


class OLS(object):
    """Ordinary least-squares (OLS) regression.

    Implemented in NumPy.  Outputs are NumPy arrays or scalars.

    Attributes largely mimic statsmodels' OLS RegressionResultsWrapper.
    (see statsmodels.regression.linear_model.RegressionResults)

    The core of the model is calculated with the 'gelsd' LAPACK driver,
    witin numpy.linalg.lstsq, yielding the coefficients (parameters).  Most
    methods are then a derivation of these coefficients.

    Parameters
    ----------
    y : array-like
        The single y (dependent, response, endogenous) variable.
    x : array-like or None, default None
        The x (independent, explanatory, exogenous) variables.  If only `y`
        is given (and `x=None`), `y` is assumed to be the first column of
        `y` and `x` the remaining [1:] columns.
    has_const : bool, default False
        Specifies whether `x` includes a user-supplied constant (a column
        vector).  If False, it is added at instantiation.
    use_const ; bool, default True
        Whether to include an intercept term in the model output.  Note the
        difference between has_const and use_const.  The former specifies
        whether a column vector of 1s is included in the input; the latter
        specifies whether the model itself should include a constant
        (intercept) term.  Exogenous data that is ~N(0,1) would have a
        constant equal to zero; specify use_const=False in this situation.
    """

    def __init__(self, y, x=None, has_const=False, use_const=True):
        self.x, self.y, self.k = _clean_xy(y, x)
        self.n = y.shape[0]

        # np.lstsq(a,b): Solves the equation a x = b by computing a vector x
        # TODO: throws LinAlgError for 1d x, has_const=False, use_const=False.
        #       np.linalg.lstsq specifies `a` must be (M,N).
        self.solution = np.linalg.lstsq(self.x, self.y, rcond=None)[0]

        self.has_const = has_const
        self.use_const = use_const

    @property
    def alpha(self):
        """The intercept term (alpha).

        Technically defined as the coefficient to a column vector of ones.
        """

        return _handle_ab(self.solution, self.use_const)[0]

    @property
    def beta(self):
        """The parameters (coefficients), excl. the intercept."""
        return _handle_ab(self.solution, self.use_const)[1]

    @property
    def condition_number(self):
        """Condition number of x; ratio of largest to smallest eigenvalue."""
        x = np.matrix(self.x)
        ev = np.linalg.eig(x.T * x)[0]
        return np.sqrt(ev.max() / ev.min())

    @property
    def df_tot(self):
        """Total degrees of freedom, n - 1."""
        return self.n - 1

    @property
    def df_reg(self):
        """Model degrees of freedom. Equal to k."""
        return self.k

    @property
    def df_err(self):
        """Residual degrees of freedom. n - k - 1."""
        return self.n - self.k - 1

    @property
    def durbin_watson(self):
        return np.sum(np.diff(self.resids) ** 2.) / self.ss_err

    @property
    def fstat(self):
        """F-statistic of the fully specified model."""
        return self.ms_reg / self.ms_err

    @property
    def fstat_sig(self):
        """p-value of the F-statistic."""
        return 1. - scs.f.cdf(self.fstat, self.df_reg, self.df_err)

    @property
    def jarque_bera(self):
        return scs.jarque_bera(self.resids)[0]

    @property
    def ms_err(self):
        """Mean squared error the errors (residuals)."""
        return self.ss_err / self.df_err

    @property
    def ms_reg(self):
        """Mean squared error the regression (model)."""
        return self.ss_reg / self.df_reg

    @property
    def predicted(self):
        """The predicted values of y (yhat)."""
        # don't transpose - shape should match that of self.y
        return self.x.dot(self.solution)

    @property
    def _pvalues_all(self):
        """Two-tailed p values for t-stats of all parameters."""
        return 2. * (1. - scs.t.cdf(np.abs(self._tstat_all), self.df_err))

    @property
    def pvalue_alpha(self):
        """Two-tailed p values for t-stats of the intercept only."""
        return _handle_ab(self._pvalues_all, self.use_const)[0]

    @property
    def pvalue_beta(self):
        """Two-tailed p values for t-stats of parameters, excl. intercept."""
        return _handle_ab(self._pvalues_all, self.use_const)[1]

    @property
    def resids(self):
        """The residuals (errors)."""
        return self.y - self.predicted

    @property
    def rsq(self):
        """The coefficent of determination, R-squared."""
        return self.ss_reg / self.ss_tot

    @property
    def rsq_adj(self):
        """Adjusted R-squared."""
        n = self.n
        k = self.k
        return 1. - ((1. - self.rsq) * (n - 1.) / (n - k - 1.))

    @property
    def _se_all(self):
        """Standard errors (SE) for all parameters, including the intercept."""
        x = np.matrix(self.x)
        err = np.atleast_1d(self.ms_err)
        se = np.sqrt(np.diagonal(np.linalg.inv(x.T * x)) * err[:, None])
        return np.squeeze(se)

    @property
    def se_alpha(self):
        """Standard errors (SE) of the intercept (alpha) only."""
        return _handle_ab(self._se_all, self.use_const)[0]

    @property
    def se_beta(self):
        """Standard errors (SE) of the parameters, excluding the intercept."""
        return _handle_ab(self._se_all, self.use_const)[1]

    @property
    def ss_tot(self):
        """Total sum of squares."""
        return np.sum(np.square(self.y - self.ybar), axis=0)

    @property
    def ss_reg(self):
        """Sum of squares of the regression."""
        return np.sum(np.square(self.predicted - self.ybar), axis=0)

    @property
    def ss_err(self):
        """Sum of squares of the residuals (error sum of squares)."""
        return np.sum(np.square(self.resids), axis=0)

    @property
    def std_err(self):
        """Standard error of the estimate (SEE).  A scalar.

        For standard errors of parameters, see _se_all, se_alpha, and se_beta.
        """

        return np.sqrt(np.sum(np.square(self.resids), axis=0)
                       / self.df_err)

    @property
    def _tstat_all(self):
        """The t-statistics of all parameters, incl. the intecept."""
        return self.solution.T / self._se_all

    @property
    def tstat_alpha(self):
        """The t-statistic of the intercept (alpha)."""
        return _handle_ab(self._tstat_all, self.use_const)[0]

    @property
    def tstat_beta(self):
        """The t-statistics of the parameters, excl. the intecept."""
        return _handle_ab(self._tstat_all, self.use_const)[1]

    @property
    def ybar(self):
        """The mean of y."""
        return self.y.mean(axis=0)


class RollingOLS(object):
    """Rolling ordinary least-squares regression.

    Uses matrix formulation with NumPy broadcasting.  Outputs are NumPy arrays
    or scalars.

    Attributes largely mimic statsmodels' OLS RegressionResultsWrapper.
    (see statsmodels.regression.linear_model.RegressionResults)

    The core of the model is calculated with the 'gelsd' LAPACK driver,
    witin numpy.linalg.lstsq, yielding the coefficients (parameters).  Most
    methods are then a derivation of these coefficients.

    Parameters
    ----------
    y : array-like
        The single y (dependent, response, endogenous) variable
    x : array-like or None, default None
        The x (independent, explanatory, exogenous) variables.  If only `y`
        is given (and `x=None`), `y` is assumed to be the first column of
        `y` and `x` the remaining [1:] columns
    window : int
        Length of each rolling window
    has_const : bool, default False
        Specifies whether `x` includes a user-supplied constant (a column
        vector).  If False, it is added at instantiation
    use_const : bool, default True
        Whether to include an intercept term in the model output.  Note the
        difference between has_const and use_const.  The former specifies
        whether a column vector of 1s is included in the input; the latter
        specifies whether the model itself should include a constant
        (intercept) term.  Exogenous data that is ~N(0,1) would have a
        constant equal to zero; specify use_const=False in this situation
    """

    def __init__(self, y, x=None, window=None, has_const=False,
                 use_const=True):
        self.x, self.y, self.k = _clean_xy(y, x)
        self.window = self.n = window
        self.xwins = utils.rolling_windows(self.x, window=window)
        self.ywins = utils.rolling_windows(self.y, window=window)
        self.solution = _rolling_lstsq(self.xwins, self.ywins)
        self.has_const = has_const
        self.use_const = use_const

    @property
    def _alpha(self):
        """The intercept term (alpha).

        Technically defined as the coefficient to a column vector of ones.
        """

        return _handle_rolling_ab(self.solution, self.use_const)[0]

    @property
    def _beta(self):
        """The parameters (coefficients), excl. the intercept."""
        return _handle_rolling_ab(self.solution, self.use_const)[1]

    @property
    def _df_tot(self):
        """Total degrees of freedom, n - 1."""
        return self.n - 1

    @property
    def _df_reg(self):
        """Model degrees of freedom. Equal to k."""
        return self.k

    @property
    def _df_err(self):
        """Residual degrees of freedom. n - k - 1."""
        return self.n - self.k - 1

    @property
    def _std_err(self):
        """Standard error of the estimate (SEE).  A scalar.

        For standard errors of parameters, see _se_all, se_alpha, and se_beta.
        """
        return np.sqrt(np.sum(np.square(self._resids), axis=1)
                       / self._df_err)

    @property
    @lru_cache(maxsize=None)
    def _predicted(self):
        """The predicted values of y ('yhat')."""
        return np.squeeze(np.matmul(self.xwins, np.expand_dims(self.solution,
                                                               axis=-1)))

    @property
    @lru_cache(maxsize=None)
    def _resids(self):
        return self.ywins - self._predicted

    @property
    def _jarque_bera(self):
        return np.apply_along_axis(scs.jarque_bera, 1, self._resids)[:, 0]

    @property
    def _durbin_watson(self):
        return np.sum(np.square(np.diff(self._resids))
                      / np.expand_dims(self._ss_err, axis=-1), axis=1)

    @property
    @lru_cache(maxsize=None)
    def _ybar(self):
        """The mean of y."""
        return self.ywins.mean(axis=1)

    @property
    @lru_cache(maxsize=None)
    def _ss_tot(self):
        """Total sum of squares."""
        return np.sum(np.square(self.ywins - np.expand_dims(self._ybar,
                                                            axis=-1)), axis=1)

    @property
    @lru_cache(maxsize=None)
    def _ss_reg(self):
        """Sum of squares of the regression."""
        return np.sum(np.square(self._predicted
                                - np.expand_dims(self._ybar, axis=1)), axis=1)

    @property
    @lru_cache(maxsize=None)
    def _ss_err(self):
        """Sum of squares of the residuals (error sum of squares)."""
        return np.sum(np.square(self._resids), axis=1)

    @property
    def _rsq(self):
        """The coefficent of determination, R-squared."""
        return self._ss_reg / self._ss_tot

    @property
    def _rsq_adj(self):
        """Adjusted R-squared."""
        n = self.n
        k = self.k
        return 1. - ((1. - self._rsq) * (n - 1.) / (n - k - 1.))

    @property
    def _ms_err(self):
        """Mean squared error the errors (residuals)."""
        return self._ss_err / self._df_err

    @property
    def _ms_reg(self):
        """Mean squared error the regression (model)."""
        return self._ss_reg / self._df_reg

    @property
    def _fstat(self):
        """F-statistic of the fully specified model."""
        return self._ms_reg / self._ms_err

    @property
    def _fstat_sig(self):
        """p-value of the F-statistic."""
        return 1. - scs.f.cdf(self._fstat, self._df_reg, self._df_err)

    @property
    @lru_cache(maxsize=None)
    def _se_all(self):
        """Standard errors (SE) for all parameters, including the intercept."""
        err = np.expand_dims(self._ms_err, axis=1)
        t1 = np.diagonal(np.linalg.inv(np.matmul(self.xwins.swapaxes(1, 2),
                                                 self.xwins)),
                         axis1=1, axis2=2)
        return np.squeeze(np.sqrt(t1 * err))

    @property
    def _se_alpha(self):
        """Standard errors (SE) of the intercept (alpha) only."""
        return _handle_rolling_ab(self._se_all, self.use_const)[0]

    @property
    def _se_beta(self):
        """Standard errors (SE) of the parameters, excluding the intercept."""
        return _handle_rolling_ab(self._se_all, self.use_const)[1]

    @property
    @lru_cache(maxsize=None)
    def _tstat_all(self):
        """The t-statistics of all parameters, incl. the intecept."""
        return self.solution / self._se_all

    @property
    def _tstat_alpha(self):
        """The t-statistic of the intercept (alpha)."""
        return _handle_rolling_ab(self._tstat_all, self.use_const)[0]

    @property
    def _tstat_beta(self):
        """The t-statistics of the parameters, excl. the intecept."""
        return _handle_rolling_ab(self._tstat_all, self.use_const)[1]

    @property
    @lru_cache(maxsize=None)
    def _pvalues_all(self):
        """Two-tailed p values for t-stats of all parameters."""
        return 2. * (1. - scs.t.cdf(np.abs(self._tstat_all), self._df_err))

    @property
    def _pvalue_alpha(self):
        """Two-tailed p values for t-stats of the intercept only."""
        return _handle_rolling_ab(self._pvalues_all, self.use_const)[0]

    @property
    def _pvalue_beta(self):
        """Two-tailed p values for t-stats of parameters, excl. intercept."""
        return _handle_rolling_ab(self._pvalues_all, self.use_const)[1]

    @property
    def _condition_number(self):
        """Condition number of x; ratio of largest to smallest eigenvalue."""
        ev = np.linalg.eig(np.matmul(self.xwins.swapaxes(1, 2), self.xwins))[0]
        return np.sqrt(ev.max(axis=1) / ev.min(axis=1))

    # -----------------------------------------------------------------
    # "Public" results

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @property
    def df_tot(self):
        """Total degrees of freedom, n - 1."""
        return self._df_tot

    @property
    def df_reg(self):
        """Model degrees of freedom. Equal to k."""
        return self._df_reg

    @property
    def df_err(self):
        """Residual degrees of freedom. n - k - 1."""
        return self._df_err

    @property
    def std_err(self):
        """Standard error of the estimate (SEE).  A scalar.

        For standard errors of parameters, see _se_all, se_alpha, and se_beta.
        """
        return self._std_err

    @property
    def predicted(self):
        """The predicted values of y ('yhat')."""
        return self._predicted

    @property
    def resids(self):
        return self._resids

    @property
    def jarque_bera(self):
        return self._jarque_bera

    @property
    def durbin_watson(self):
        return self._durbin_watson

    @property
    def ybar(self):
        """The mean of y."""
        return self._ybar

    @property
    def ss_tot(self):
        """Total sum of squares."""
        return self._ss_tot

    @property
    def ss_reg(self):
        """Sum of squares of the regression."""
        return self._ss_reg

    @property
    def ss_err(self):
        """Sum of squares of the residuals (error sum of squares)."""
        return self._ss_err

    @property
    def rsq(self):
        """The coefficent of determination, R-squared."""
        return self._rsq

    @property
    def rsq_adj(self):
        """Adjusted R-squared."""
        return self._rsq_adj

    @property
    def ms_err(self):
        """Mean squared error the errors (residuals)."""
        return self._ms_err

    @property
    def ms_reg(self):
        """Mean squared error the regression (model)."""
        return self._ms_reg

    @property
    def fstat(self):
        """F-statistic of the fully specified model."""
        return self._fstat

    @property
    def fstat_sig(self):
        """p-value of the F-statistic."""
        return self._fstat_sig

    @property
    def se_alpha(self):
        """Standard errors (SE) of the intercept (alpha) only."""
        return self._se_alpha

    @property
    def se_beta(self):
        """Standard errors (SE) of the parameters, excluding the intercept."""
        return self._se_beta

    @property
    def tstat_alpha(self):
        """The t-statistic of the intercept (alpha)."""
        return self._tstat_alpha

    @property
    def tstat_beta(self):
        """The t-statistics of the parameters, excl. the intecept."""
        return self._tstat_beta

    @property
    def pvalue_alpha(self):
        """Two-tailed p values for t-stats of the intercept only."""
        return self._pvalue_alpha

    @property
    def pvalue_beta(self):
        """Two-tailed p values for t-stats of parameters, excl. intercept."""
        return self._pvalue_beta

    @property
    def condition_number(self):
        """Condition number of x; ratio of largest to smallest eigenvalue."""
        return self._condition_number


# TODO: Instead of quasi-private and public attributes, probably should
#       just call super() directly.  I.e.:
#
#    @property
#    def beta(self):
#        return DataFrame(super(PandasRollingOLS, self).beta())


class PandasRollingOLS(RollingOLS):
    def __init__(self, y, x=None, window=None, has_const=False, use_const=True,
                 names=None):

        # A little redundant needing to establish k...
        if not names:
            if x is None:
                if hasattr(y, 'columns'):
                    if has_const:
                        names = y.columns[1:-1]
                    else:
                        names = y.columns[1:]
                else:
                    if has_const:
                        k = y.shape[-1] - 2
                    else:
                        k = y.shape[-1] - 1
                    names = ['feature{}'.format(i) for i in range(1, k+1)]
            else:
                if hasattr(x, 'columns'):
                    if has_const:
                        names = x.columns[:-1]
                    else:
                        names = x.columns
                else:
                    if has_const:
                        k = x.shape[-1] - 1
                    else:
                        if x.ndim == 1:
                            k = 1
                        else:
                            k = x.shape[-1]
                    names = ['feature{}'.format(i) for i in range(1, k+1)]
        self.names = names

        super(PandasRollingOLS, self).__init__(y=y, x=x, window=window,
                                               has_const=has_const,
                                               use_const=use_const)

        self.index = y.index
        # Index for the rolling result starts at (window - 1)
        self.ridx = y.index[window-1:]

    def _wrap_series(self, stat, name=None):
        if name is None:
            name = stat[1:]
        return Series(getattr(self, stat), index=self.ridx, name=name)

    def _wrap_dataframe(self, stat):
        return DataFrame(getattr(self, stat), index=self.ridx,
                         columns=self.names)

    def _wrap_multidx(self, stat, name=None):
        if name is None:
            name = stat[1:]
        outer = np.repeat(self.ridx, self.window)
        inner = np.ravel(utils.rolling_windows(self.index.values,
                                               window=self.window))
        return Series(getattr(self, stat).flatten(), index=[outer, inner],
                      name=name).rename_axis(['end', 'subperiod'])

    @property
    def alpha(self):
        return self._wrap_series(stat='_alpha', name='intercept')

    @property
    def beta(self):
        return self._wrap_dataframe(stat='_beta')

    # df_tot, df_reg, df_err are scalars; no override.

    @property
    def std_err(self):
        """Standard error of the estimate (SEE).  A scalar.

        For standard errors of parameters, see _se_all, se_alpha, and se_beta.
        """
        return self._wrap_series(stat='_std_err')

    @property
    def predicted(self):
        """The predicted values of y ('yhat')."""
        return self._wrap_multidx('_predicted')

    @property
    def resids(self):
        return self._wrap_multidx('_resids')

    @property
    def jarque_bera(self):
        return self._wrap_series(stat='_jarque_bera')

    @property
    def durbin_watson(self):
        return self._wrap_series(stat='_durbin_watson')

    @property
    def ybar(self):
        """The mean of y."""
        return self._wrap_series(stat='_ybar')

    @property
    def ss_tot(self):
        """Total sum of squares."""
        return self._wrap_series(stat='_ss_tot')

    @property
    def ss_reg(self):
        """Sum of squares of the regression."""
        return self._wrap_series(stat='_ss_reg')

    @property
    def ss_err(self):
        """Sum of squares of the residuals (error sum of squares)."""
        return self._wrap_series(stat='_ss_err')

    @property
    def rsq(self):
        """The coefficent of determination, R-squared."""
        return self._wrap_series(stat='_rsq')

    @property
    def rsq_adj(self):
        """Adjusted R-squared."""
        return self._wrap_series(stat='_rsq_adj')

    @property
    def ms_err(self):
        """Mean squared error the errors (residuals)."""
        return self._wrap_series(stat='_ms_err')

    @property
    def ms_reg(self):
        """Mean squared error the regression (model)."""
        return self._wrap_series(stat='_ms_reg')

    @property
    def fstat(self):
        """F-statistic of the fully specified model."""
        return self._wrap_series(stat='_fstat')

    @property
    def fstat_sig(self):
        """p-value of the F-statistic."""
        return self._wrap_series(stat='_fstat_sig')

    @property
    def se_alpha(self):
        """Standard errors (SE) of the intercept (alpha) only."""
        return self._wrap_series(stat='_se_alpha')

    @property
    def se_beta(self):
        """Standard errors (SE) of the parameters, excluding the intercept."""
        return self._wrap_dataframe(stat='_se_beta')

    @property
    def tstat_alpha(self):
        """The t-statistic of the intercept (alpha)."""
        return self._wrap_series(stat='_tstat_alpha')

    @property
    def tstat_beta(self):
        """The t-statistics of the parameters, excl. the intecept."""
        return self._wrap_dataframe(stat='_tstat_beta')

    @property
    def pvalue_alpha(self):
        """Two-tailed p values for t-stats of the intercept only."""
        return self._wrap_series(stat='_pvalue_alpha')

    @property
    def pvalue_beta(self):
        """Two-tailed p values for t-stats of parameters, excl. intercept."""
        return self._wrap_dataframe(stat='_pvalue_beta')

    @property
    def condition_number(self):
        """Condition number of x; ratio of largest to smallest eigenvalue."""
        return self._wrap_series(stat='_condition_number')
