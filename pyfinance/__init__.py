"""Python package designed for financial analysis of security returns.

Modules
=======
- datasets : Financial dataset web scrubbing
- general : Generalized tools for financial analysis & quantitative finance
- ols : Ordinary least squares regression
- options : European option valuation and strategy visualization
- returns : Statistical analysis of time series security returns data
- utils : Basic utilities and helper functions
"""

__author__ = 'Brad Solomon <brad.solomon.1124@gmail.com>'

NUMTODEC = {'num' : 1., 'dec' : 0.01}

from options import *
from returns import *

from pyfinance.ols import *
from pyfinance.utils import *
from . import returns
from .datasets import *
from .general import *

returns.extend_pandas()