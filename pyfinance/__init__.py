"""pyfinance: investment management and security returns analysis.

Modules
-------
`datasets.py`  Financial dataset download & assembly via `requests`.

`general.py`   General-purpose financial computations, such as active
               share calculation, returns distribution approximation,
               and tracking error optimization.

`ols.py`       Ordinary least-squares (OLS) regression, supporting
               static and rolling cases, built with a matrix formulation
               and implemented with NumPy.

`options.py`   Vectorized option calculations, including Black-Scholes
               Merton European option valuation, Greeks, and implied
               volatility, as well as payoff determination for common
               money-spread option strategies.

`returns.py`   Statistical analysis of financial time series through
               the CAPM framework, designed to mimic functionality of
               software such as FactSet Research Systems and Zephyr,
               with improved speed and flexibility.

`utils.py`     Utilities not fitting into any of the above.
"""

__author__ = 'Brad Solomon <brad.solomon.1124@gmail.com>'


# Echoing Pandas, bring these to top-level namespace.
from .returns import TFrame, TSeries  # noqa
