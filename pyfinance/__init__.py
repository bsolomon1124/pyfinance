"""Python package designed for general financial and security returns analysis.


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

from .datasets import *
from .general import *
from .ols import *
from .options import *
from .utils import *
from .returns import *

from . import returns
returns.extend_pandas()