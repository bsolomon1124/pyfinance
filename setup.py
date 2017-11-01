from setuptools import setup, find_packages


long_description = """Python package designed for general financial and security returns analysis.

Modules
=======
- datasets : Financial dataset web scrubbing
- general : Generalized tools for financial analysis & quantitative finance
- ols : Ordinary least squares regression (static & rolling cases)
- options : European option valuation and strategy visualization
- returns : Statistical analysis of time series security returns data
- utils : Basic utilities and helper functions
"""

setup(
    name='pyfinance',
    description='Python package designed for general financial and security returns analysis.',
    long_description=long_description,
    version='0.1.0',
    author='Brad Solomon',
    author_email='brad.solomon.1124@gmail.com',
    url='https://github.com/bsolomon1124/pyfinance',
    license='MIT',
    install_requires=[
        'Pandas >= 0.20.1',
        'Numpy >= 0.7.0',
        'Matplotlib >= 1.1',
        'scipy >= 0.10.0',
        'requests >= 2.11.1',
        'xlrd >= 0.5.4',
        'statsmodels >= 0.6.0'
        ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Topic :: Office/Business :: Financial',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Scientific/Engineering :: Information Analysis'
        ],
    keywords='finance investment analysis regression options securities CAPM',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    python_requires='>=3'
    )
