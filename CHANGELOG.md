# pyfinance changelog

Changes prior to version 0.2.0 are untracked.

Versioning attempts to follow the specifications laid out in [PEP 440](https://www.python.org/dev/peps/pep-0440/).

## 1.0.0

This is a backwards-incompatible revamp of pyfinance.

Major code overhauls:


Other changes:
- Updated the README with package structure, tutorial, depenencies, and API.
- Added this changelog.
- Thorough PEP8 linting via flake8.
- Added Sphinx-friendly docstrings.
- Removed Selenium + PhantomJS as a dependency.  Scraping information on activeshare is fully deprecated.
- Scraping SEC filings by CIK lookup is now deprecated.  Pass the url of the .xml directly.
- got rid of a pickling decorator
- Removed the `.diagram()` method of options strategies
