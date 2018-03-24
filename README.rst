pyfinance
=========

**Note**: pyfinance aims for compatability with all minor releases of Python 3.x, but does not guarantee workability with Python 2.x.

----

pyfinance is a Python package built for investment management and analysis of security returns.

It is meant to be a complement to existing packages geared towards quantitative finance, such as `pyfolio
<https://github.com/quantopian/pyfolio>`_, `ffn
<https://github.com/pmorissette/ffn>`_, `pandas-datareader
<https://github.com/pydata/pandas-datareader>`_, and `fecon235
<https://github.com/rsvp/fecon235>`_.

--------
Contents
--------

pyfinance is best explored on a module-by-module basis:

===================  ===========
Module               Description
===================  ===========
:code:`datasets.py`  Financial dataset download & assembly via web-scraping.
:code:`general.py`   General-purpose financial computations, such as active share calculation, returns distribution approximation, and tracking error optimization.
:code:`ols.py`       Ordinary least-squares (OLS) regression, supporting static and rolling cases, built with a matrix formulation and implemented with NumPy.
:code:`options.py`   Vectorized option calculations, including Black-Scholes Merton European option valuation, Greeks, and implied volatility, as well as payoff determination for common money-spread option strategies.
:code:`returns.py`   Statistical analysis of financial time series through the CAPM framework, designed to mimic functionality of software such as FactSet Research Systems and Zephyr, with improved speed and flexibility.
:code:`utils.py`     Utilities not fitting into any of the above.
===================  ===========

------------
Installation
------------

pyfinance is available via `PyPI
<https://pypi.python.org/pypi/pyfinance/0.2.1>`_.  The latest version is 0.3.0 [TODO] as of March 2018.:  Install with pip::

$ pip3 install pyfinance

------------
Dependencies
------------

pyfinance relies primarily on Python's scientific stack, including NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn, and StatsModels.  Other dependencies include Beautiful Soup, Requests, Selenium, xrld, and xmltodict.

See :code:`setup.py` for specific version threshold requirements.

--------
Tutorial
--------

This is a brief walkthrough of some of pyfinance's features.

---
API
---

This section provides more detail on pyfinance call syntaxes.

-----------------
Package structure
-----------------

.. code::

    pyfinance/
    ├── CHANGELOG
    ├── CHANGELOG
    ├── LICENSE
    ├── MANIFEST.in
    ├── README.rst
    ├── pyfinance/
    │   ├── __init__.py
    │   ├── datasets.py
    │   ├── general.py
    │   ├── ols.py
    │   ├── options.py
    │   ├── returns.py
    │   └── utils.py
    ├── setup.py
    └── tests/
        ├── __init__.py
        ├── test_ols.py
        └── test_options.py
