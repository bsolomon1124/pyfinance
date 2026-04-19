# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.1] - 2026-04-19

Re-release of 2.0.0 to work around two bugs in the publish workflow
that blocked the 2.0.0 GitHub Release from completing. The published
PyPI package content is identical to 2.0.0.

### Fixed

- `publish.yml`: the SBOM was being written to
  `dist/requirements.sbom.txt`, where `pypa/gh-action-pypi-publish`
  applied twine-style validation to it and failed with
  `InvalidDistribution: Unknown distribution format`. SBOM is now
  written to `sbom/requirements.sbom.txt`, uploaded as a separate
  artifact, and attached to the GitHub Release from its own path.
- `publish.yml`: removed a vestigial `dist/*.sigstore` glob from the
  release-files list. `sigstore/gh-action-sigstore-python@v3`
  produces only `.sigstore.json` bundle files; the empty glob was
  aborting the release step because `fail_on_unmatched_files: true`.

## [2.0.0] - 2026-04-18

### Added

- `py.typed` marker ships in the wheel; PEP 561 consumers now pick up
  inline types. `Typing :: Typed` classifier set.
- `pyfinance.general.EwmParams` TypedDict as the return type of
  `ewm_params`; `pyfinance.general.EwmParam` Literal alias for the
  `param` argument.
- `pyfinance.ols.RegressionResult` Protocol describing the attributes
  common to `OLS`, `RollingOLS`, and `PandasRollingOLS`.
- Literal type aliases on public APIs: `OptionKind` (`BSM`),
  `InFormat` (`activeshare`), `RateFreq` / `RfFreq` (`datasets`).
- `Taskfile.yml` command shortcuts for the common dev workflow
  (`task install / test / test-cov / test-matrix / lint / fmt /
  fmt-check / typecheck / mdlint / check / build / sbom / pre-commit
  / clean`).
- `.pre-commit-config.yaml` (ruff-check, ruff-format, ty, hygiene
  hooks, markdownlint-cli2); compatible with `prek`.
- `.github/workflows/ci.yml` — SHA-pinned matrix CI across Python
  3.10–3.14 with lint / test / build jobs.
- `.github/workflows/publish.yml` — tag-triggered PyPI publish via
  OIDC trusted publishing, Sigstore signing, SBOM attachment, GitHub
  Release creation. All actions pinned to SHA256.
- `.github/dependabot.yml` for `github-actions` and `uv` ecosystems.
- Offline smoke tests for `pyfinance.datasets` covering the lazy
  `pandas_datareader` proxy and the public surface of the module.
- `user_agent` (required) parameter on `pyfinance.datasets.load_13f`,
  per SEC EDGAR fair-access policy.
- Shields.io badges on README.md (PyPI version, license, pyversions).

### Changed

- **BREAKING:** Minimum Python is now 3.10; supported range is
  3.10 through 3.14. Python 3.9 reached end-of-life
  (<https://devguide.python.org/versions/>).
- **BREAKING:** `pyfinance.general.factor_loadings` now requires the
  caller to pass `factors` directly; the removed `pickle_from` and
  `pickle_to` keyword arguments no longer exist.
- **BREAKING:** `pyfinance.datasets.load_13f(url, user_agent)` requires
  a descriptive `user_agent` string and raises `ValueError` otherwise.
- **BREAKING:** `pyfinance.datasets.pdr` is now a lazy proxy object,
  not the real `pandas_datareader` module. Attribute access still works
  (`pdr.DataReader(...)`); direct module-type identity checks do not.
- Migrate from `setup.py` to `pyproject.toml` using the native
  [`uv_build`](https://docs.astral.sh/uv/concepts/build-backend/)
  build backend; adopt [`uv`](https://docs.astral.sh/uv/) as the
  package manager. `uv.lock` is committed.
- Switch to a `src/` project layout.
- Pin `pandas` to `>=2,<3`; pin `numpy` compatibility for 2.x.
- Internal `freq="M"` usages updated to `"ME"` to avoid pandas 2.2+
  deprecation warnings. `load_rf(freq=...)` still accepts the legacy
  single-letter codes and translates them internally.
- `__version__` is now resolved at runtime via
  `importlib.metadata.version()`, so the version lives in a single
  place (`pyproject.toml`).
- Adopt [`ruff`](https://docs.astral.sh/ruff/) for linting and
  formatting, and [`ty`](https://github.com/astral-sh/ty) for type
  checking. Legacy `black` / `flake8` references removed. Enabled rule
  set: `E W F I UP B SIM C4 PT RET PIE PERF RUF A`.
- Coverage enforced at 75 % via `pytest-cov`; configuration lives in
  `[tool.pytest.ini_options]` and `[tool.coverage.*]` in
  `pyproject.toml`.
- Rename `LICENSE` → `LICENSE.txt`.

### Removed

- **BREAKING:** `pyfinance.datasets.load_factors`. Aggregated ~15
  external sources (bit.ly-redirected AQR/Duke/Pastor Excel files,
  restructured CBOE CSV endpoints, pre-rebrand `cboe.com/micro/...`
  paths); most no longer resolve and repairing each vendor integration
  is prohibitive.
- **BREAKING:** `pyfinance.datasets.load_shiller`. Robert Shiller's
  `ie_data.xls` is still published, but its sheet layout has drifted
  past the hardcoded `skiprows` / `skipfooter` / column-name mapping.
- **BREAKING:** `pyfinance.datasets.load_retaildata`. census.gov
  migrated away from the
  `https://www.census.gov/retail/marts/www/advNNNNN.txt` endpoints; no
  stable drop-in replacement.
- Per-module `__author__` attributes (single source of truth is now
  `pyproject.toml`).
- `MANIFEST.in` and `setup.py` (packaging metadata is fully declared
  in `pyproject.toml`).
- Dead `PY37` compatibility branch in `utils.py`.

### Fixed

- `numpy>=2.0` compatibility: import nan-aware reductions from the
  top-level `numpy` namespace instead of the removed
  `numpy.lib.nanfunctions` module.
- `pandas>=2` compatibility: replace removed
  `DataFrame.fillna(method="ffill")` with `DataFrame.ffill()`;
  replace `pd.read_csv(..., delim_whitespace=True)` with `sep=r"\s+"`;
  replace `pd.read_excel(..., skip_footer=...)` with `skipfooter=...`;
  replace removed `Index.is_all_dates` with
  `isinstance(index, pd.DatetimeIndex)` in `TSeries.anlzd_ret`.
- `pyfinance.utils.get_anlz_factor`: `PERIODS_PER_YEAR` was keyed by
  `FreqGroup` enum members but looked up with a raw integer — in
  pandas 2.x `FreqGroup` is a plain `Enum` (not `IntEnum`) so lookups
  silently failed. Keyed on `.value` now. Also accepts already-resolved
  numeric factors from `_try_get_freq`.
- `pyfinance.returns.TSeries.anlzd_stdev` and `.semi_stdev`: treated
  the `freq` string as a number, raising `TypeError: unsupported
  operand type(s) for ** or pow(): 'str' and 'float'`. Now calls
  `utils.get_anlz_factor` to resolve the frequency first.
- `pyfinance.returns.TSeries.__init__`: dropped the monotonic-index
  check that fired on every internal pandas reverse-slice (breaking
  `drawdown_start` / `drawdown_length` among others).
- `pyfinance.general.factor_loadings`: fixed a latent comma-omission
  that turned `("Price-Signal Model", [...])(...)` into a call on a
  tuple; raised `TypeError` at import/first-call on any modern
  Python. (Function is still otherwise broken — see Deprecated.)
- `pyfinance.datasets.load_rates`: the daily 1-month non-financial
  commercial paper series was listed as `DCPN30` (a non-existent FRED
  symbol); it is now `DCPN1M`.

### Deprecated

- `pyfinance.general.amortize`, `factor_loadings`, `PortSim`, and
  `corr_heatmap` / `PCA.screeplot` are marked `# pragma: no cover`
  because they depend on removed NumPy APIs (`np.ppmt` / `ipmt` /
  `pmt`), broken internal calls, or are purely visual. They still
  import cleanly but are not exercised by the test suite; planned
  for rewrite in 2.1.

## [1.3.0]

### Fixed

- Address `import` changes required by pandas versions 1.1+.

### Added

- Test on and support Python 3.8 and 3.9.
- `__version__` attribute to package.

### Changed

- Loosen requirements version specifiers in `setup.py`.

## [1.2.5]

### Fixed

- False-positive `ValueError` (see d59e7ca).

### Changed

- Lint entire package with `black`.

## [1.2.4]

### Fixed

- Misspelled method in `returns.py` (see 6f2970c).

### Changed

- Bring changelog up to date.

## [1.2.3]

### Fixed

- Misnaming of `get_anlz_factor()` utility function (see e4f997d).

## [1.2.2] / [1.2.1]

### Fixed

- Bug in `ols._confirm_constant()` (see 5046dd9, 4d7d7b5, 5d07d1e).

## [1.1.1]

### Fixed

- `svd_flip()` in PCA class (see b45d12e).

## [1.1.0]

### Changed

- Minor docstring changes.

## [1.0.0]

### Changed

- **BREAKING:** Overhaul of `returns.py`.

### Added

- Several NumPy wrapper functions in `utils.py`.

## [0.3.0]

### Removed

- **BREAKING:** Active-share scraping: `load_activeshare()` and
  `scrub_activeshare()` removed from the `datasets` module.
- **BREAKING:** SEC-filing CIK lookup in `load_13f()`; pass the URL
  of the `.xml` directly.
- `.diagram()` method of options strategies (thin Matplotlib wrapper).
- Selenium + PhantomJS as a dependency.

### Added

- This changelog.
- README sections on package structure, tutorial, dependencies, API.
- Sphinx-friendly docstrings.

### Changed

- Thorough PEP 8 linting via flake8.

[Unreleased]: https://github.com/bsolomon1124/pyfinance/compare/v2.0.1...HEAD
[2.0.1]: https://github.com/bsolomon1124/pyfinance/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/bsolomon1124/pyfinance/compare/v1.3.0...v2.0.0
[1.3.0]: https://github.com/bsolomon1124/pyfinance/compare/v1.2.5...v1.3.0
[1.2.5]: https://github.com/bsolomon1124/pyfinance/compare/v1.2.4...v1.2.5
[1.2.4]: https://github.com/bsolomon1124/pyfinance/compare/v1.2.3...v1.2.4
[1.2.3]: https://github.com/bsolomon1124/pyfinance/compare/v1.2.2...v1.2.3
[1.2.2]: https://github.com/bsolomon1124/pyfinance/compare/v1.2.1...v1.2.2
[1.2.1]: https://github.com/bsolomon1124/pyfinance/compare/v1.1.1...v1.2.1
[1.1.1]: https://github.com/bsolomon1124/pyfinance/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/bsolomon1124/pyfinance/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/bsolomon1124/pyfinance/compare/v0.3.0...v1.0.0
[0.3.0]: https://github.com/bsolomon1124/pyfinance/releases/tag/v0.3.0
