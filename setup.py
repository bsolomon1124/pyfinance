from codecs import open
import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# PyPI upload:
# $ python3 setup.py test sdist upload -r pypi

setup(
    name='pyfinance',
    description='Python package designed for security returns analysis.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='1.3.0',
    author='Brad Solomon',
    author_email='bsolomon@protonmail.com',
    url='https://github.com/bsolomon1124/pyfinance',
    license='MIT',
    install_requires=[
        'beautifulsoup4',
        'matplotlib',
        'numpy',
        'pandas_datareader',
        'pandas >= 0.20',
        'requests',
        'scipy',
        'seaborn',
        'scikit-learn',
        'statsmodels',
        'xlrd',
        'xmltodict'
        ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Office/Business :: Financial',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Scientific/Engineering :: Information Analysis'
        ],
    keywords=[
        'finance',
        'investment',
        'analysis',
        'regression',
        'options',
        'securities',
        'CAPM',
        'machine learning',
        'risk'
        ],
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    python_requires='>=3'
    )
