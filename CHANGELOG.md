# pyfinance changelog

Changes prior to version 0.2.0 are untracked.

Versioning attempts to follow the specifications laid out in [PEP 440](https://www.python.org/dev/peps/pep-0440/).

## 1.2.4

- Bugfix: corrected a misspelled method in `returns.py` (see 6f2970c)
- Bring this changelog up to date

## 1.2.3

- Bugfix: corrected misnaming of `get_anlz_factor()` utility function (see e4f997d)

## 1.2.2/1.2.1

- Bugfix: Fix a bug in `ols._confirm_constant()`  (See 5046dd9, 4d7d7b5, and 5d07d1e)

## 1.1.1

- Bugfix: `svd_flip()` in PCA class (see b45d12e)

## 1.1.0

- Minor docstring changes

## 1.0.0

This release overhauled `returns.py` in a backwards-incompatible manner.  It also added several NumPy wrapper functions to `utils.py`.

## 0.3.0

While technically a minor release, this update does have some immediate deprecations.
- Scraping information on active share is fully deprecated; `load_activeshare()` and `scrub_activeshare()` have been removed from the `datasets` module.
- Scraping SEC filings by CIK lookup in `load_13f()` is removed.  Pass the url of the .xml directly.
- Removed the `.diagram()` method of options strategies, which was just a light Matplotlib plotting wrapper.

Other changes:
- Updated the README with package structure, tutorial, depenencies, and API.
- Added this changelog.
- Thorough PEP8 linting via flake8.
- Added Sphinx-friendly docstrings.
- Removed Selenium + PhantomJS as a dependency.
