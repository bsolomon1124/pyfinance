# pyfinance changelog

Changes prior to version 0.2.0 are untracked.

Versioning attempts to follow the specifications laid out in [PEP 440](https://www.python.org/dev/peps/pep-0440/).

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
