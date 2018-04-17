"""Generalized tools for financial analysis & quantitative finance.

Descriptions
------------
activeshare
    Compute the active ahare of a fund versus an index.
amortize
    Construct an amortization schedule for a fixed-rate loan.
BestFitDist
    Which continuous distribution best models `x`?
corr_heatmap
    Wrapper around seaborn.heatmap for visualizing correlation matrix.
ewm_params
    Return corresponding param. values for exponential weighted functions.
ewm_weights
    Exponential weights as a function of position `i`.
ewm_bootstrap
    Bootstrap a new distribution through exponential weighting.
factor_loadings
    Security factor exposures generated through OLS regression.
PCA
    Performs principal component analysis (PCA) on a set of returns.
PortSim
    Basic portfolio simulation.
TEOpt
    Tracking error optimization/security replication.
variance_inflation_factor
    Calculate variance inflation factor (VIF) for each all `regressors`.
"""

__author__ = 'Brad Solomon <brad.solomon.1124@gmail.com>'
__all__ = [
    'activeshare', 'amortize', 'BestFitDist', 'corr_heatmap',
    'ewm_params', 'ewm_weights', 'ewm_bootstrap', 'factor_loadings', 'PCA',
    'PortSim', 'TEOpt', 'variance_inflation_factor'
    ]

from collections import OrderedDict
import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as sco
import scipy.stats as scs
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.utils.extmath import svd_flip
from statsmodels.regression import linear_model
from statsmodels.tools import add_constant

from pyfinance import datasets, ols, returns, utils

NUMTODEC = {'num': 1., 'dec': 0.01}


def activeshare(fund, idx, in_format='num'):
    """Compute the active ahare of a fund versus an index.

    Formula is 0.5 * sum(abs(w_fund - w_idx)).

    Parameters
    ----------
    fund: {pd.Series, pd.DataFrame}
        The fund's holdings, with tickers as the Index and weights as
        values.  If a DataFrame, each column is a ticker/portfolio.
    idx: pd.Series
        The benchmark portfolio, with tickers as the Index and weights
        as values.
    in_format: {'num', 'dec'}
        Decimal notation of the inputs.  "num" means 0.5% is denoted 0.5;
        "dec" means 0.5% is denoted 0.005.

    Returns
    -------
    act_sh : pd.Series or pd.DataFrame
        The dimension will be one-lower than that of `fund`.
        If `fund` is a Series, the result is a scalar value.
        If `fund` is a DataFrame, the result is a Series, with
        the columns of `fund` as the resulting Index.

    .. _Cremers & Petajisto, 'How Active Is Your Fund Manager?', 2009
    """

    if not (fund.index.is_unique) and (idx.index.is_unique):
        raise ValueError('Inputs must have unique indices.')
    if isinstance(fund, pd.DataFrame):
        cols = fund.columns
    fund = fund * NUMTODEC[in_format]
    idx = idx * NUMTODEC[in_format]

    union = fund.index.union(idx.index)
    fund = fund.reindex(union, fill_value=0.).values
    idx = idx.reindex(union, fill_value=0.).values

    if fund.ndim == 1:
        # Resulting active share will be a scalar
        diff = fund - idx
    else:
        diff = fund - idx[:, None]
    act_sh = np.sum(np.abs(diff) * 0.5, axis=0)
    if isinstance(act_sh, np.ndarray):
        act_sh = pd.Series(act_sh, index=cols)
    return act_sh


def amortize(rate, nper, pv, freq='M'):
    """Construct an amortization schedule for a fixed-rate loan.

    Rate -> annualized input

    Example
    -------
    # a 6.75% $200,000 loan, 30-year tenor, payments due monthly
    # view the 5 final months
    print(amortize(rate=.0675, nper=30, pv=200000).round(2).tail())
         beg_bal     prin  interest  end_bal
    356  6377.95 -1261.32    -35.88  5116.63
    357  5116.63 -1268.42    -28.78  3848.22
    358  3848.22 -1275.55    -21.65  2572.67
    359  2572.67 -1282.72    -14.47  1289.94
    360  1289.94 -1289.94     -7.26    -0.00
    """

    freq = utils.convertfreq(freq)
    rate = rate / freq
    nper = nper * freq

    periods = np.arange(1, nper + 1, dtype=int)

    principal = np.ppmt(rate, periods, nper, pv)
    interest = np.ipmt(rate, periods, nper, pv)
    pmt = np.pmt(rate, nper, pv)

    def balance(pv, rate, nper, pmt):
        dfac = (1 + rate) ** nper
        return pv * dfac - pmt * (dfac - 1) / rate

    res = pd.DataFrame({'beg_bal': balance(pv, rate, periods - 1, -pmt),
                        'prin': principal,
                        'interest': interest,
                        'end_bal': balance(pv, rate, periods, -pmt)},
                       index=periods)['beg_bal', 'prin', 'interest', 'end_bal']
    return res


class BestFitDist(object):
    """Which continuous distribution best models `x`?

    Core process within `.fit` is adopted from @tmthydvnprt (Stack Overflow).

    Parameters
    ----------
    x : 1-d array-like or Series
        Empirical input data to model/fit
    bins : int or sequence of scalars or str, optional, default 'auto'
        Passed to np.histogram; see docs
    distributions : sequence or None, default None
        If specified, limit the tests to these distribution names only.
        Example: ['normal', 'cauchy', 'laplace']
        If None, defaults to the full set; see:
        docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions

    Example
    -------
    from pyfinance import BestFitDist
    # 1000 random samples from a N~(5,3) distribution
    r = np.random.normal(loc=5, scale=3, size=1000)
    dist = ['normal', 'cauchy', 'laplace', 'lognormal', 'johnsonsu']
    fitted = BestFitDist(r, distributions=dist).fit()

    print(fitted.best())
    name                     norm
    params    (4.88130759176, ...
    sse                 0.0015757
    dtype: object

    print(fitted.all())
            name      sse               params
    3      alpha  0.22361  (3.8149848058e-0...
    4     anglit  0.02263  (5.49753868994, ...
    5    arcsine  0.10216  (-4.69316502376,...
    6       beta  0.00161  (150.926126014, ...
    7  betaprime  0.00226  (30.4979678995, ...
    8   bradford  0.06195  (1.02226022038, ...
    9       burr  0.23714  (0.662101587716,...
    0      ksone      NaN  (1.30534444533, ...
    1  kstwobign  0.01196  (-6.31494451637,...
    2       norm  0.00158  (4.88130759176, ...
    """

    def __init__(self, x, bins='auto', distributions=None):
        self.x = x
        self.bins = bins
        if distributions is None:
            distributions = scs._continuous_distns._distn_names
        self.distributions = distributions

    def fit(self):
        """Fit each distribution to `data` and calculate an SSE.

        WARNING: significant runtime. (~1min)
        """

        # Defaults/anchors
        best_sse = np.inf
        best_param = (0.0, 1.0)
        best_dist = scs.norm

        # Compute the histogram of `x`.  density=True gives a probability
        # density function at each bin, normalized such that the integral over
        # the range is 1.0
        hist, bin_edges = np.histogram(self.x, bins=self.bins, density=True)

        # The results of np.histogram will have len(bin_edges) = len(hist) + 1
        # Find the midpoint at each bin to reduce the size of bin_edges by 1
        bin_edges = (bin_edges + np.roll(bin_edges, -1))[:-1] / 2.0

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')

            sses = []
            params = []

            for dist in self.distributions:

                dist = getattr(scs, dist)

                try:
                    # The generic rv_continuous.fit() returns `mle_tuple`:
                    #    'MLEs for any shape parameters (if applicable),
                    #    followed by those for location and scale.'
                    param = *shape, loc, scale = dist.fit(self.x)

                    pdf = dist.pdf(bin_edges, loc=loc, scale=scale, *shape)
                    sse = np.sum(np.power(hist - pdf, 2.0))

                    sses.append(sse)
                    params.append(param)

                    if best_sse > sse > 0.:
                        best_dist = dist
                        best_param = param
                        best_sse = sse
                        best_pdf = pdf

                except (NotImplementedError, AttributeError):
                    sses.append(np.nan)
                    params.append(np.nan)

        self.best_dist = best_dist
        self.best_param = best_param
        self.best_sse = best_sse
        self.best_pdf = best_pdf

        self.sses = sses
        self.params = params
        self.hist = hist
        self.bin_edges = bin_edges

        return self

    def best(self):
        """The resulting best-fit distribution, its parameters, and SSE."""
        return pd.Series({'name': self.best_dist.name,
                          'params': self.best_param,
                          'sse': self.best_sse})

    def all(self, by='name', ascending=True):
        """All tested distributions, their parameters, and SSEs."""
        res = pd.DataFrame({'name': self.distributions,
                            'params': self.params,
                            'sse': self.sses})[['name', 'sse', 'params']]
        res.sort_values(by=by, ascending=ascending, inplace=True)

        return res

    def plot(self):
        """Plot the empirical histogram versus best-fit distribution's PDF."""
        plt.plot(self.bin_edges, self.hist, self.bin_edges, self.best_pdf)
        # TODO: labels


def corr_heatmap(x, mask_half=True, cmap='RdYlGn_r', vmin=-1, vmax=1,
                 linewidths=0.5, square=True, figsize=(10, 10), **kwargs):
    """Wrapper around seaborn.heatmap for visualizing correlation matrix.

    Parameters
    ----------
    x : DataFrame
        Underlying data (not a correlation matrix)
    mask_half : bool, default True
        If True, mask (whiteout) the upper right triangle of the matrix
    All other parameters passed to seaborn.heatmap:
    https://seaborn.pydata.org/generated/seaborn.heatmap.html

    Example
    -------
    # Generate some correlated data
    >>> import numpy as np
    >>> import pandas as pd
    >>> k = 10
    >>> size = 400
    >>> mu = np.random.randint(0, 10, k).astype(float)
    >>> r = np.random.ranf(k ** 2).reshape((k, k)) * 5
    >>> df = pd.DataFrame(np.random.multivariate_normal(mu, r, size=size))
    >>> corr_heatmap(df, figsize=(6, 6))
    """

    if mask_half:
        mask = np.zeros_like(x.corr().values)
        mask[np.triu_indices_from(mask)] = True
    else:
        mask = None
    with sns.axes_style('white'):
        return sns.heatmap(x.corr(), cmap=cmap, vmin=vmin, vmax=vmax,
                           linewidths=linewidths, square=square,
                           mask=mask, **kwargs)


def ewm_params(param, param_value):
    """Corresponding parameter values for exponentially weighted functions.

    Parameters
    ----------
    param : {'alpha', 'com', 'span', 'halflife'}
    param_value : float or int
        The parameter value.

    Returns
    -------
    result : dict
        Layout/index of corresponding parameters.
    """

    if param not in ['alpha', 'com', 'span', 'halflife']:
        raise NameError('`param` must be one of {alpha, com, span, halflife}')

    def input_alpha(a):
        com = 1./a - 1.
        span = 2./a - 1.
        halflife = np.log(0.5)/np.log(1. - a)
        return {'com': com, 'span': span, 'halflife': halflife}

    def output_alpha(param, p):
        eqs = {
            'com': 1./(1. + p),
            'span': 2./(p + 1.),
            'halflife': 1. - np.exp(np.log(0.5)/p)
        }
        return eqs[param]

    if param == 'alpha':
        dct = input_alpha(a=param_value)
        alpha = param_value
    else:
        alpha = output_alpha(param=param, p=param_value)
        dct = input_alpha(a=alpha)
    dct.update({'alpha': alpha})
    return dct


def ewm_weights(i, com=None, span=None, halflife=None, alpha=None):
    """Exponential weights as a function of position `i`.

    Mimics pandas' methodology with adjust=True:
    http://pandas.pydata.org/pandas-docs/stable/computation.html#exponentially-weighted-windows
    """

    if not any((com, span, halflife, alpha)):
        raise ValueError('specify one of `com`, `span`, `halflife`, `alpha`')

    params = [com, span, halflife, alpha]
    pos = next(i for (i, x) in enumerate(params) if x)
    param_value = next(item for item in params if item is not None)
    lookup = dict(enumerate(['com', 'span', 'halflife', 'alpha']))
    param = lookup[pos]
    alpha = ewm_params(param=param, param_value=param_value)['alpha']

    res = (1. - alpha) ** np.arange(i)[::-1]
    res /= res.sum()
    return res


def ewm_bootstrap(a, size=None, com=None, span=None, halflife=None,
                  alpha=None):
    """Bootstrap a new distribution through exponential weighting.

    Parameters
    ----------
    a : 1-D array-like
        Array from which to generate random sample of elements
    size : int or tuple of ints, default None
        Output shape.  If None, a single value is returned
    com : float, default None
        Center of mass; alpha = 1 / (1 + com), for com ≥ 0
    span : float, default None
        Span parameter; a = 2 / (span + 1), for span ≥ 1
    halflife : float, default None
        Halflife parameter; alpha = 1 − exp(log(0.5) / halflife),
        for halflife > 0
    alpha : float, default None
        Smoothing factor

    Example
    -------
    >>> import pandas as pd
    >>> np.random.seed(123)

    # Our bootstrapped histogram should approximate these freqs
    >>> ewm_weights(10, alpha=.10)
    [ 0.05948221  0.06609135  0.07343483  0.08159426  0.09066029  0.10073365
      0.11192628  0.12436253  0.13818059  0.15353399]

    >>> res = ewm_bootstrap(np.arange(10), size=int(1e6), alpha=.10)
    >>> res = pd.Series(res).value_counts()
    >>> (res / res.sum()).head()
    9    0.15323
    8    0.13834
    7    0.12424
    6    0.11189
    5    0.10113
    dtype: float64
    """

    if not any((com, span, halflife, alpha)):
        raise ValueError('Specify one of `com`, `span`, `halflife`, `alpha`.')
    p = ewm_weights(i=len(a), com=com, span=span, halflife=halflife,
                    alpha=alpha)
    res = np.random.choice(a=a, size=size, p=p)
    return res


def factor_loadings(r, factors=None, scale=False, pickle_from=None,
                    pickle_to=None):
    """Security factor exposures generated through OLS regression.

    Incorporates a handful of well-known factors models.

    Parameters
    ----------
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

    Example
    -------
    # TODO
    """

    # TODO:
    # - Might be appropriate for `returns`, will require higher dimensionality
    # - Option to subtract or not subtract RF (Jensen alpha)
    # - Annualized alpha
    # - Add variance inflation factor to output (method of `ols.OLS`)
    # - Add 'missing=drop' functionality (see statsmodels.OLS)
    # - Take all combinations of factors; which has highest explanatory power
    #       or lowest SSE/MSE?

    if factors is None:
        factors = datasets.load_factors(pickle_from=pickle_from,
                                        pickle_to=pickle_to)

    r, factors = utils.constrain(r, factors)

    if isinstance(r, pd.Series):
        n = 1
        r = r.subtract(factors['RF'])
    elif isinstance(r, pd.DataFrame):
        n = r.shape[1]
        r = r.subtract(np.tile(factors['RF'], (n, 1)).T)
        # r = r.subtract(factors['RF'].values.reshape(-1,1))
    else:
        raise ValueError('`r` must be one of (Series, DataFrame)')

    if scale:
        # Scale up the volatilities of all factors besides MKT & RF, to the
        # vol of MKT.  Both means and the standard deviations are multiplied
        # by the scale factor (ratio of MKT.std() to other stdevs)
        tgtvol = factors['MKT'].std()
        diff = factors.columns.difference(['MKT', 'RF'])  # don't scale these
        vols = factors[diff].std()
        factors.loc[:, diff] = factors[diff] * tgtvol / vols

    # Right-hand-side dict of models
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
    res = pd.DataFrame(columns=cols, index=idx)

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


class PCA(object):
    """Performs principal component analysis (PCA) on a set of returns.

    Designed as an alternative to sklearn's PCA implementationn; has less
    available parameters but a number of pre-canned convenience methods.  This
    class-based implementation is principally geared towards analysis of
    asset returns, such as in Rankin (2016).

    One major difference is that the sign of the eignvectors is not flipped to
    enforce deterministic output, as is done in sklearn.
        (See sklearn/decomposition/pca.py, L390)

    Parameters
    ----------
    m : array-like (DataFrame, np.matrix, np.array)
        Time series of returns.  Cannot be sparse.
    threshold : float, or str in {'Jolliffe', 'Kaiser'}
        The threshold below which eigenvalues and corresponding components
        are dropped.  'Jolliffe' is Jolliffe's criterion (0.7), while
        Kaiser's criterion is 1.0.

    Resources
    ---------
    Abdi & Williams, Principal Component Analysis, 2010
    Bro, Acar & Kolda, Resolving the Sign Ambiguity in the Singular Value
        Decomposition, 2007
    Mandel, Use of Singular Value Decomposition in Regression Analysis, 1982
    Rankin, Multi-Dimensional Diversification: Improving Portfolio Selection
        Using Principal Component Analysis, 2016
    """

    def __init__(self, m, threshold='Jolliffe', u_based_decision=False):
        # Keep a raw version of m for view and create a scaled version, ms
        # scaled to N~(0,1)
        if isinstance(m, pd.DataFrame):
            self.feature_names = m.columns
            self.m = m.values
        else:
            self.feature_names = range(m.shape[1])
            self.m = m
        if np.isnan(self.m).any():
            raise ValueError('Input contains NaN')
        self.ms = scale(self.m)

        thresholds = {'Kaiser': 1.0, 'Jolliffe': 0.7}
        if threshold is not None:
            if isinstance(threshold, str):
                self.threshold = thresholds[threshold]
            else:
                self.threshold = float(threshold)
        else:
            self.threshold = 0.
        self.u_based_decision = u_based_decision

    def fit(self):
        """Fit the model by computing full SVD on m.

        SVD factors the matrix m as u * np.diag(s) * v, where u and v are
        unitary and s is a 1-d array of m‘s singular values.  Note that the SVD
        is commonly written as a = U S V.H, and the v returned by this function
        is V.H (the Hermitian transpose).  Therefore, we denote V.H as vt, and
        back into the actual v, denoted just v.

        The decomposition uses np.linalg.svd with full_matrices=False, so for
        m with shape (M, N), then the shape of:
         - u is (M, K)
         - v is (K, N
        where K = min(M, N)

        Intertia is the percentage of explained variance.

        Returns
        -------
        self, to enable method chaining
        """

        self.n_samples, self.n_features = self.ms.shape
        self.u, self.s, self.vt = np.linalg.svd(self.ms, full_matrices=False)
        self.v = self.vt.T

        # sklearn's implementation is to guarantee that the left and right
        # singular vectors (U and V) are always the same, by imposing the
        # that the largest coefficient of U in absolute value is positive
        # This implementation uses u_based_decision=False rather than the
        # default True to flip that logic and ensure the resulting
        # components and loadings have high positive coefficients
        self.u, self.vt = svd_flip(self.u, self.v,
                                   u_based_decision=self.u_based_decision)
        self.v = self.vt.T

        # Drop eigenvalues with value > threshold
        # *keep* is number of components retained
        self.eigenvalues = self.s ** 2 / self.n_samples
        self.keep = np.count_nonzero(self.eigenvalues > self.threshold)

        self.inertia = (self.eigenvalues / self.eigenvalues.sum())[:self.keep]
        self.cumulative_inertia = self.inertia.cumsum()[:self.keep]
        self.eigenvalues = self.eigenvalues[:self.keep]

        return self

    @property
    def eigen_table(self):
        """Eigenvalues, expl. variance, and cumulative expl. variance."""
        idx = ['Eigenvalue', 'Variability (%)', 'Cumulative (%)']
        table = pd.DataFrame(np.array([self.eigenvalues,
                                       self.inertia,
                                       self.cumulative_inertia]),
                             columns=['F%s' % i for i in
                                      range(1, self.keep + 1)],
                             index=idx)

        return table

    def loadings(self):
        """Loadings = eigenvectors times sqrt(eigenvalues)."""
        loadings = self.v[:, :self.keep] * np.sqrt(self.eigenvalues)
        cols = ['PC%s' % i for i in range(1,  self.keep + 1)]
        loadings = pd.DataFrame(loadings, columns=cols,
                                index=self.feature_names)
        return loadings

    @property
    def relative_diversification(self, x=0.75):
        """Number of components needed to explain > x% of variance."""
        return 1 + np.count_nonzero(self.cumulative_inertia < x)

    def screeplot(self, title=None, xlabel='Eigenvalue',
                  ylabel='Cumulative Explained Variance (%)'):
        plt.plot(range(1, self.keep + 1), self.cumulative_inertia, 'o-')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title('Scree Plot')
        plt.show()

    def varimax(self):
        loadings = self.loadings().values
        gamma = 1.
        q = 20
        tol = 1e-6
        p, k = loadings.shape
        r = np.eye(k)
        d = 0

        for i in range(q):
            d_old = d
            lam = np.dot(loadings, r)
            u, s, vh = np.linalg.svd(np.dot(
                loadings.T, np.asarray(lam) ** 3 - (gamma / p)
                * np.dot(lam, np.diag(np.diag(np.dot(lam.T, lam))))))
            r = np.dot(u, vh)
            d = np.sum(s)
            if d / d_old < tol:
                break

        cols = ['PC%s' % i for i in range(1, self.keep + 1)]

        return pd.DataFrame(np.dot(loadings, r), columns=cols,
                            index=self.feature_names)


class PortSim(object):
    """Basic portfolio simulation.

    NOTE: work-in-progress as of 2017.08.25.

    Parameters
    ----------
    r : Series or DataFrame
        Time series of gross-of-fees security/account/fund returns.  Series are
        converted to single-column DataFrame with `prep`
    fee : float, default 0.
        The *annual* percentage fee expressed as a decimal
    fee_freq : str, int, or float, default 'Q'
        The fee frequency.  If str, it is case insensitive and should be:
        - One of ('D', 'W', 'M', 'Q', 'A')
        - One of pandas anchored offsets, such as 'W-FRI':
          pandas.pydata.org/pandas-docs/stable/timeseries.html#anchored-offsets
        If int or float, this is interpreted as the number of billing periods
        per year
    start : str or date, default None
        Date (inclusive) on which to truncate `r`
    end : str or date, default None
        Date (inclusive) on which to truncate `r`
    lookback : dict, default {}
        Keyword args dict passed to `constrain_horizon`.  See
        utils.constrain_horizon
    strict : bool, default False
        Passed to `constrain_horizon`.
        If True, raise Error if the implied start date on the horizon predates
        the actual start date of `r`.  If False, just return `r` in this
        situation
    dist_amt : int or float, default None
        The *annual* dollar distribution.
        dist_amt > 0: money withdrawn from account/fund
        dist_amt < 0: money deposited to account/fund
    v0 : int or float, default float(1e6)
        The initial investment
    include_start : bool, default True
        Whether to prepend a starting date to resulting DataFrame displays
    freq : str, default 'M'
        The frequency of `r`.  Passed to returns.prep
    name : str, default None
        For cases where a Series is converted to a single-column DataFrame,
        `name` specifies the resulting column name; it is passed to
        `pd.Series.to_frame`; passed to returns.prep
    in_format : str, {'num', 'dec'}
        Passed to returns.prep; converts percentage figures from
        numeral form to decimal form
    """

    def __init__(self, r, fee=0., fee_freq='Q', start=None, end=None,
                 lookback={}, strict=False, dist_amt=None, dist_pct=None,
                 dist_freq=None, v0=float(1e6), include_start=True, freq='M',
                 name=None,  in_format='num'):

        self.gross = returns.prep(r=r, freq=freq, name=name,
                                  in_format=in_format)

        # fee_freq: if str -> frequency; if int/float -> periods/yr
        # Get `fee` to a per-period float
        if isinstance(fee_freq, (int, float)):
            self.fee_freq = fee_freq
            self.fee = fee / self.fee_freq
        elif isinstance(fee_freq, str):
            self.fee_freq = utils.convertfreq(fee_freq)
            self.fee = fee / self.fee_freq

        # Logic for interaction of `start`, `end`, and `lookback`
        # TODO: how should lookback be passed? Consider params to
        # `constrain_horizon`
        if any((start, end)) and not lookback:
            self.gross = self.gross[start:end]
        elif lookback:
            # TODO: cleanup
            self.gross = utils.constrain_horizon(self.gross, **lookback)
        elif all((any((start, end)), lookback)):
            raise ValueError('if `lookback` is specified, both `start` and'
                             ' `end` should be None')

        self.index = self.gross.index
        self.columns = self.gross.columns

        masktypes = {12.: 'is_month_end',
                     4.: 'is_quarter_end',
                     1.: 'is_quarter_end'}

        mask = getattr(self.index, masktypes[self.fee_freq])
        self.feesched = np.where(mask, self.fee, 0.)

        # Net of fees (not yet of distributions)
        self.net = (1. + self.gross.values) \
            * (1. - self.feesched.reshape(-1, 1)) - 1.
        self.net = pd.DataFrame(self.net, index=self.index,
                                columns=self.columns)

        self.dist_amt = dist_amt
        self.dist_pct = dist_pct
        self.dist_freq = dist_freq
        self.v0 = v0
        self.include_start = include_start

    def account_value(self):

        if self.dist_amt is not None:
            ri = returns.return_index(self.net)
            t = (ri.index.is_quarter_end.reshape((-1, 1)) / ri).cumsum()
            res = ri * (self.v0 - self.dist_amt * t)
            if self.include_start:
                res = returns.insert_start(res, base=self.v0)

        elif self.dist_pct is not None:
            res = returns.return_index((1.+self.net) * (1.-self.dist_pct) - 1.,
                                       base=self.v0,
                                       include_start=self.include_start)
        return res


class TEOpt(object):
    """Tracking error optimization/security replication.

    At the end of each period, TEOpt computes an optimization over the trailing
    `window` periods, solving for the vector of weights that would have
    minimized the tracking error of `r` to `proxies` over that period.

    Example
    -------
    # TODO

    Parameters
    ----------
    r : DataFrame
        The target time series of returns to be replicated
    proxies : DataFrame
        The available proxies used towards building a replicating portfolio
    window : int
        The lookback window utilized in each rolling optimization
    sumto: float, default 1
        Optimization constraint: proxy weights should sum to this parameter
    """

    def __init__(self, r, proxies, window, sumto=1.):

        self.r = r
        self.proxies = proxies
        self.window = int(window)
        self.sumto = sumto

        self.newidx = r.index[window-1:]
        self.cols = proxies.columns
        self.n = proxies.shape[1]

        # TODO: constrain

        self._r = utils.rolling_windows(r, window=window)
        self._proxies = utils.rolling_windows(proxies, window=window)

    def optimize(self):
        """Analogous to `sklearn`'s fit.  Returns `self` to enable chaining."""

        def te(weights, r, proxies):
            """Helper func.  `pyfinance.tracking_error` doesn't work here."""
            if isinstance(weights, list):
                weights = np.array(weights)
            proxy = np.sum(proxies * weights, axis=1)
            te = np.std(proxy - r)  # not anlzd...
            return te

        ew = utils.equal_weights(n=self.n, sumto=self.sumto)
        bnds = tuple((0, 1) for x in range(self.n))
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - self.sumto})

        xs = []
        funs = []
        for i, j in zip(self._r, self._proxies):
            opt = sco.minimize(te, x0=ew, args=(i, j), method='SLSQP',
                               bounds=bnds, constraints=cons)
            x, fun = opt['x'], opt['fun']
            xs.append(x)
            funs.append(fun)
        self._xs = np.array(xs)
        self._funs = np.array(funs)

        return self

    def opt_weights(self):
        """Optimal weights (period-end)."""
        return pd.DataFrame(self._xs, index=self.newidx, columns=self.cols)

    def ex_ante_te(self):
        """Tracking error corresponding to each optimized lookback period."""
        # TODO: check manually
        return pd.Series(self._funs, index=self.newidx)

    def replicate(self):
        """Forward-month returns of the replicating portfolio."""
        return np.sum(self.proxies[self.window:] * self._xs[:-1],
                      axis=1).reindex(self.r.index)


def variance_inflation_factor(regressors, hasconst=False):
    """Calculate variance inflation factor (VIF) for each all `regressors`.

    A wrapper/modification of statsmodels:
    statsmodels.stats.outliers_influence.variance_inflation_factor

    One recommendation is that if VIF is greater than 5, then the explanatory
    variable `x` is highly collinear with the other explanatory
    variables, and the parameter estimates will have large standard errors
    because of this. [source: StatsModels]

    Parameters
    ----------
    regressors: DataFrame
        DataFrame containing the entire set of regressors
    hasconst : bool, default False
        If False, a column vector will be added to `regressors` for use in
        OLS

    Example
    -------
    # Generate some data
    from datetime import date
    from pandas_datareader.data import DataReader as dr

    syms = {'TWEXBMTH' : 'usd',
            'T10Y2YM' : 'term_spread',
            'PCOPPUSDM' : 'copper'
           }
    start = date(2000, 1, 1)
    data = (dr(syms.keys(), 'fred', start)
            .pct_change()
            .dropna())
    data = data.rename(columns = syms)

    print(variance_inflation_factor(data))
    usd            1.31609
    term_spread    1.03793
    copper         1.37055
    dtype: float64
    """

    if not hasconst:
        regressors = add_constant(regressors, prepend=False)
    k = regressors.shape[1]

    def vif_sub(x, regressors):
        x_i = regressors.iloc[:, x]
        mask = np.arange(k) != x
        x_not_i = regressors.iloc[:, mask]
        rsq = linear_model.OLS(x_i, x_not_i, missing='drop').fit().rsquared_adj
        vif = 1. / (1. - rsq)
        return vif

    vifs = pd.Series(np.arange(k), index=regressors.columns)
    vifs = vifs.apply(vif_sub, args=(regressors,))

    # Find the constant column (probably called 'const', but not necessarily
    # and drop it. `is_nonzero_const` borrowed from statsmodels.add_constant
    is_nonzero_const = np.ptp(regressors.values, axis=0) == 0
    is_nonzero_const &= np.all(regressors != 0.0, axis=0)
    vifs.drop(vifs.index[is_nonzero_const], inplace=True)
    return vifs
