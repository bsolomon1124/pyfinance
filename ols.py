"""Ordinary least-squares (OLS) regression."""

__author__ = 'Brad Solomon <brad.solomon.1124@gmail.com>'

__all__ = ['OLS', 'RollingOLS']

from collections import OrderedDict
import time

import numpy as np
from pandas import DataFrame, Series
import scipy.stats as scs
from statsmodels.tools import add_constant



# TODO: why not make `y` always 2d?
# TODO: use np.atleast_1d when you're getting a 0d array of a single scalar

class OLS(object):
    """OLS regression.  Designed for simplicity & usability.

    Largely mimics statsmodels' OLS RegressionResultsWrapper.
    (see statsmodels.regression.linear_model.RegressionResults)

    The core of the model is calculated with the 'gelsd' LAPACK driver,
    witin numpy.linalg.lstsq, yielding the coefficients (parameters).  Most
    methods are then a derivation of these coefficients.

    This implementation is built to support several different cases:

    # of y vectors     # of x vectors (k)   Supported?
    ==============     ==================   =========
    1                  1                    Yes
    1                  > 1                  Yes
    > 1                1                    Yes (each y treated separately)
    > 1                > 1                  No

    Parameters
    ==========
    y : array-like (ndarray, Series, DataFrame)
        The y (dependent, response, endogenous) variable
    x : array-like (ndarray, Series, DataFrame) or None, default None
        The x (independent, explanatory, exogenous) variable.  If only `y`
        is given (and `x=None`), `y` is assumed to be the first column of 
        `y` and `x` the remaining [1:] columns
    hasconst : bool, default False
        Specifies whether `x` includes a user-supplied constant (a column
        vector).  If False, it is added at instantiation
    names : dict, default None
        User-specified names of the y and x variables.  If None, they will be 
        inferred from the input data attributes (i.e. name of Series or columns
        of a DataFrame).  If not None, specify names with keys ('y', 'x')

    Attributes
    ==========
    d : str
        Date of model instantiation
    t : str
        Time of model instantiation
    n : int
        Number of observations
    k : int
        Number of `x` terms, excluding the intercept
    x : np.ndarray
        The exogenous data as np.array(n, k + 1), with appended column vector
    y : np.ndarray
        The endogenous data as np.array(n,)
    names : dict
        Variable names with keys ('x', 'y')
    solution : 
        The returned `x` from np.linalg.lstsq
    """
    
    # TODO: what about a case with >1 y vectors and > 1 x vectors?

    def __init__(self, y, x=None, hasconst=False, names=None):
        # TODO: constrained regression case, preferably with cvxopt, not scipy

        # Date and time of class instantiation; used in model summary
        self.d = time.strftime('%Y-%m-%d')
        self.t = time.strftime('%I:%M %p')

        # Number of observations
        # TODO: handle (drop) NaNs
        self.n = len(y)

        # Retain special indices (i.e. DatetimeIndex) for later use
        # otherwise, just use a 0-indexed range(n)
        if hasattr(y, 'index'):
            self.idx = y.index
        else:
            self.idx = np.arange(self.n)

        # If only `y` is given (and `x=None`), `y` is assumed to be the first 
        # column of `y` and `x` the remaining [1:] columns
        if x is None:
            if isinstance(y, DataFrame):
                self.names = {'y' : y.columns[0].tolist(), 
                              'x' : y.columns[1:].tolist()
                             }
                x = y.iloc[:, 1:].values
                y = y.iloc[:, 0 ].values
            elif isinstance(y, np.ndarray):                 
                x = y[:, 1:]
                y = y[:, 0 ]
                self.names = {'y' : 'y', 'x' : ['var%s' % i for 
                                                i in range(x.shape[1])]}

        # Case of ndarray with x not None
        elif isinstance(y, np.ndarray):
            if x.ndim == 1:
                x = x.reshape(-1,1)
            self.names = {'y' : 'y', 'x' : ['var%s' % i for 
                                            i in range(x.shape[1])]}
            
        # Case of Series or DataFrame with x not None
        else:                     
            self.names = {'y' : y.name if isinstance(y, Series) else 
                                y.columns.tolist(),
                          'x' : x.name if isinstance(x, Series) else 
                                x.columns.tolist()}
            x = x.values
            y = y.values
            # TODO: will throw error if you're passing an array, not pandas

        if hasconst == False:
            x = add_constant(x)

        # Now we've gotten y and x down to consistently shaped ndarrays
        # regardless of input
        self.x = x
        self.y = np.squeeze(y)
        self.k = self.x.shape[1] - 1

        if names is not None:
            # Overwrite above values, no prettier logic than this
            self.names = names
        self.names['x'].insert(0, 'alpha')

        # np.lstsq(a,b): Solves the equation a x = b by computing a vector x
        self.solution = np.linalg.lstsq(self.x, self.y)[0]
    
    def alpha(self):
        """The intercept term (alpha).

        Technically defined as the coefficient to a column vector of ones.
        """

        # TODO: with a single x term, this will return a 0d array
        # such as array(1.2632862808026597); may not be ideal for rolling          
        return np.squeeze(self.solution[0])

    def anova(self):
        """Analysis of variance (ANOVA) table for the model."""
        # TODO: works, but not pretty, for count(y) > 1
        # fixes: dict of DataFrames? or MultiIndex
        if self.y.ndim > 1:
            raise RuntimeError('Method `anova` is not supported for cases with'
                               ' greater than one y variable.')
        stats = [('df', [self.df_reg(), self.df_err(), self.df_tot()]),
                 ('ss', [self.ss_reg(), self.ss_err(), self.ss_tot()]),
                 ('ms', [self.ms_reg(), self.ms_err(), np.nan]),
                 ('f', [self.fstat(), np.nan, np.nan]),
                 ('sig_f', [self.fstat_sig(), np.nan, np.nan])
                ]
        return DataFrame(OrderedDict(stats), index=['reg', 'err', 'tot'])

    def beta(self):
        """The parameters (coefficients), excl. the intercept."""
        return np.squeeze(self.solution[1:])

    def _ci_all(self, a=0.05):
        z = scs.t(self.df_err()).ppf(1. - a / 2.)
        b = self.solution.T
        se = self._se_all()
        # upper, lower
        return np.array([b - z * se, b + z * se])

    def ci_alpha(self, a=0.05):
        """Confidence interval for the intercept (alpha)."""
        ci = self._ci_all(a=a)
        ci = ci[:, 0] if ci.ndim == 2 else ci[:, :, 0]
        return ci

    def ci_beta(self, a=0.05):
        """Confidence interval for the parameters, excl. intercept.

        May need to transpose the output if using in a DataFrame.
        """
        ci = self._ci_all(a=a)
        # TODO: may need to transpose
        # TODO: check for cases with k > 1
        ci = ci[:, 1] if ci.ndim == 2 else ci[:, :, 1]      
        return ci

    def condition_number(self):
        """Condition number of x; ratio of largest to smallest eigenvalue."""
        x = np.matrix(self.x)
        ev = np.linalg.eig(x.T * x)[0]
        return np.sqrt( ev.max() / ev.min() )

    def df_tot(self):
        """Total degrees of freedom, n - 1."""
        return self.n - 1.
        
    def df_reg(self):
        """Model degrees of freedom. Equal to k."""
        return self.k

    def df_err(self):
        """Residual degrees of freedom. n - k - 1."""
        return self.n - self.k - 1.

    def durbin_watson(self):
        return np.sum( np.diff(self.resids()) ** 2. ) / self.ss_err()

    def fstat(self):
        """F-statistic of the fully specified model."""
        return self.ms_reg() / self.ms_err()

    def fstat_sig(self):
        """p-value of the F-statistic."""
        return 1 - scs.f.cdf(self.fstat(), self.df_reg(), self.df_err())

    def jarque_bera(self):
        return scs.jarque_bera(self.resids())[0]

    def ms_err(self):
        """Mean squared error the errors (residuals)."""
        return self.ss_err() / self.df_err()

    def ms_reg(self):
        """Mean squared error the regression (model)."""
        return self.ss_reg() / self.df_reg()

    def overview(self):
        # TODO: x_var will include 'alpha' here..
        stats = [('run_date', self.d),
                 ('run_time', self.t),
                 ('y_var', self.names['y']),
                 ('x_var', self.names['x']),
                 ('n', self.n),
                 ('k', self.k),
                 ('rsq', self.rsq()),
                 ('rsq_adj', self.rsq_adj()),
                 ('se', self.std_err()),
                 ('jb', self.jarque_bera()),
                 ('dw', self.durbin_watson()),
                 ('condno', self.condition_number())
                ]                     
        return Series(OrderedDict(stats))            

    def params(self):
        """Summary table for the parameters, incl. the intercept."""
        # TODO: throws error for count(y) > 1
        # issue is you're trying to shove 2d arrays into single DataFrame cols
        # fixes: dict of DataFrames?
        # or better, MultiIndex
        if self.y.ndim > 1:
            raise RuntimeError('Method `params` is not supported for cases with'
                               ' greater than one y variable.')
        stats = [('coef', self.solution),
                 ('se', self._se_all()),
                 ('tstat', self._tstat_all()),
                 ('pval', self._pvalues_all()),
                 ('lower_ci', self._ci_all()[0]),
                 ('upper_ci', self._ci_all()[1])
                ]
        return DataFrame(OrderedDict(stats), index=self.names['x'])

    def predicted(self):
        """The predicted values of y (yhat)."""
        # don't transpose - shape should match that of self.y
        return self.x.dot(self.solution)

    def _pvalues_all(self):
        """Two-tailed p values for t-stats of all parameters."""
        return 2. * (1. - scs.t.cdf(np.abs(self._tstat_all()), self.df_err()))

    def pvalue_alpha(self):
        """Two-tailed p values for t-stats of the intercept only."""
        return self._pvalues_all()[0]

    def pvalue_beta(self):
        """Two-tailed p values for t-stats of parameters, excl. intercept."""
        return self._pvalues_all()[1:]

    def resids(self, full_output=False):
        if self.y.ndim > 1 and full_output:
            raise RuntimeError('Method `resids` is not supported for cases with'
                               ' greater than one y variable.')
        """The residuals (errors).

        Parameters
        ==========
        full_output : bool, default False
            If False, return an nx1 vector of residuals only.  If True, return
            a DataFrame that also includes the actual and predicted values.
        """
        if full_output:
            resids = [('actual', self.y),
                     ('predicted', self.predicted()),
                     ('resid', self.resids())
                     ]
            return DataFrame(OrderedDict(resids), index=self.idx)
        else:
            return self.y - self.predicted()
        
    def rsq(self):
        """The coefficent of determination, R-squared."""
        return self.ss_reg() / self.ss_tot()

    def rsq_adj(self):
        """Adjusted R-squared."""
        n = self.n
        k = self.k
        return 1. - ( (1. - self.rsq()) * (n - 1.) / (n - k - 1.) )

    def _se_all(self):
        """Standard errors (SE) for all parameters, including the intercept."""
        x = np.matrix(self.x)
        err = np.atleast_1d(self.ms_err())
        se = np.sqrt(np.diagonal(np.linalg.inv(x.T * x)) * err[:,np.newaxis])
        # Squeeze now rather than taking [:, i] later and creating
        # single-element arrays
        return np.squeeze(se)

        # old
        # np.sqrt(np.diagonal(linalg.inv(x.T * x) * self.ms_err()))
        # np.array([np.sqrt(np.diagonal(np.linalg.inv(x.T * x) 
        #           * ols.ms_err()[i])) for i in range(ols.ms_err().shape[0])])

    def se_alpha(self):
        """Standard errors (SE) of the intercept (alpha) only."""
        se = self._se_all()
        se_alpha = se[0] if se.ndim == 1 else se[:, 0]
        return se_alpha

    def se_beta(self):
        """Standard errors (SE) of the parameters, excluding the intercept."""
        se = self._se_all()
        # Keep these 2d for cases with 2+ y vectors, even in cases with just
        # a single x vector.  Otherwise, can't distinguish what's what
        se_beta = se[1:] if se.ndim == 1 else se[:, 1:]
        return se_beta

    def ss_tot(self):
        """Total sum of squares."""
        return np.sum(np.square(self.y - self.ybar()), axis=0)

    def ss_reg(self):
        """Sum of squares of the regression."""
        return  np.sum(np.square(self.predicted() - self.ybar()), axis=0)
        
    def ss_err(self):
        """Sum of squares of the residuals (error sum of squares)."""     
        return np.sum(np.square(self.resids()), axis=0)

    def std_err(self):
        """Standard error of the estimate (SEE).  A scalar.

        For standard errors of parameters, see _se_all, se_alpha, and se_beta.
        """

        return np.sqrt(np.sum(np.square(self.resids()), axis=0) / self.df_err())
        
    def summary(self):
        """Summary table of regression results.  An OrderedDict of subtables."""
        if self.y.ndim > 1:
            raise RuntimeError('Method `summary` is not supported for cases' 
                               ' with greater than one y variable.')
        stats = [('overview', self.overview()),
                 ('anova', self.anova()),
                 ('params', self.params()),
                 ('resids', self.resids(full_output=True))
                ]
        return OrderedDict(stats)

    def _tstat_all(self):
        """The t-statistics of all parameters, incl. the intecept."""
        return self.solution.T / self._se_all()

    def tstat_beta(self):
        """The t-statistics of the parameters, excl. the intecept."""
        t = self._tstat_all()
        # Same logic as with standard errors and other coefficient properties
        t = t[1:] if t.ndim == 1 else t[:, 1:]
        return t

    def tstat_alpha(self):
        """The t-statistic of the intercept (alpha)."""
        t = self._tstat_all()
        t = t[0] if t.ndim == 1 else t[:, 0]
        return t

    def ybar(self):
        """The mean of y."""
        return self.y.mean(axis=0)


class RollingOLS(OLS):
    """Rolling OLS regression."""
    # TODO: docs
    def __init__(self, y, x=None, window=None, hasconst=False, names=None):
        OLS.__init__(self, y=y, x=x, hasconst=hasconst, names=names)
            
        self.xwins = rwindows(self.x, window=window)
        self.ywins = rwindows(self.y, window=window)

        # TODO: iterator?
        self.models = [OLS(y=ywin, x=xwin) for ywin, xwin 
                       in zip(self.ywins, self.xwins)]

    def _rolling_stat(self, stat, **kwargs):
        """Core generic method of the class."""
        stats = []
        for model in self.models:
            # TODO: accept additional arguments
            s = getattr(model, stat)(**kwargs)
            stats.append(s)
        return np.array(stats)
    
    def beta(self):
        return self._rolling_stat('beta')

    def alpha(self):
        return self._rolling_stat('alpha')

    def condition_number(self):
        return self._rolling_stat('condition_number')

    def fstat(self):
        return self._rolling_stat('fstat')

    def fstat_sig(self):
        return self._rolling_stat('fstat_sig')

    def predicted(self, full_output=False):
        return self._rolling_stat('predicted')

    def jarque_bera(self, full_output=False):
        return self._rolling_stat('jarque_bera')

    def pvalue_alpha(self, full_output=False):
        return self._rolling_stat('pvalue_alpha')

    def pvalue_beta(self, full_output=False):
        return self._rolling_stat('pvalue_beta')

    def resids(self, full_output=False):
        return self._rolling_stat('resids')

    def rsq(self):
        return self._rolling_stat('rsq')

    def rsq_adj(self):
        return self._rolling_stat('rsq_adj')        

    def tstat_alpha(self):
        return self._rolling_stat('tstat_alpha')

    def tstat_beta(self):
        return self._rolling_stat('tstat_beta')        
