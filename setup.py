from codecs import open as _open
import os
from setuptools import setup, find_packages


# __file__ is the pathname of the file from which the module was loaded.
# `here` will be something like 'C:\Users\bsolomon\anaconda3\pyfinance'.
here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with _open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='pyfinance',
    description='Python package designed for general financial'
                ' and security returns analysis.',
    long_description=long_description,
    version='0.1.3',
    author='Brad Solomon',
    author_email='brad.solomon.1124@gmail.com',
    url='https://github.com/bsolomon1124/pyfinance',
    download_url='https://github.com/bsolomon/pyfinance/archive/0.1.tar.gz',
    license='MIT',
    install_requires=[
        'beautifulsoup4 >= 4.6.0',
        'matplotlib >= 1.1',
        'numpy >= 0.7.0',
        'pandas_datareader >= 0.5.0',
        'pandas >= 0.20.1',
        'requests >= 2.11.1',
        'scipy >= 0.10.0',
        'seaborn >= 0.8.0',
        'selenium >= 3.5.0',
        'scikit-learn >= 0.18',
        'statsmodels >= 0.6.0',
        'xlrd >= 0.5.4',
        'xmltodict >= 0.10.0'
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
