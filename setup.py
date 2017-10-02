from setuptools import setup, find_packages


long_description = """Python package designed for financial analysis of security returns.
Modules
=======
- datasets : Financial dataset web scrubbing
- general : Generalized tools for financial analysis & quantitative finance
- ols : Ordinary least squares regression
- options : European option valuation and strategy visualization
- returns : Statistical analysis of time series security returns data
- utils : Basic utilities and helper functions
"""

setup(name='pyfinance',
      description = 'Python package designed for financial analysis, mainly of security returns.',
      long_description = long_description,
      version = '0.1.01',
      author = 'Brad Solomon',
      author_email = 'brad.solomon.1124@gmail.com',
      url = 'https://github.com/bsolomon1124/pyfinance',
      license = 'LICENSE.txt',
      install_requires=["Pandas >= 0.16.0",
                        "Numpy >= 0.7.0",
                        "Matplotlib >= 1.1",
                        "scipy >= 0.10.0",
                        "requests >= 2.11.1",
                        "xlrd >= 0.5.4",
                        "statsmodels >= 0.6.0",
                        "selenium >= 3.0",
                        "pandas_datareader >= 0.3.0",
                        "seaborn",
                        "scikit-learn"],
      packages = find_packages(exclude=['contrib', 'docs', 'tests*']))