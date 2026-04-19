# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

The project is managed with [`uv`](https://docs.astral.sh/uv/) and uses
the native `uv_build` backend — not hatch, setuptools, or poetry.

Most common commands are wrapped in [`Taskfile.yml`](./Taskfile.yml) — use
`task --list` to see them, or drive `uv` directly:

```bash
uv sync                         # install runtime + dev deps from uv.lock
uv run pytest                   # full test suite + coverage (fails < 75 %)
uv run pytest --no-cov tests/test_ols.py::test_ols_onedim  # a single test
uv run ruff check               # lint
uv run ruff format --check      # format-only check
uv run ty check                 # type check
uv run prek run --all-files     # pre-commit hooks (compatible with pre-commit)
uv build                        # emit sdist + wheel to ./dist
```

To exercise the Python-version matrix locally (all 3.10–3.14 are supported):

```bash
task test-matrix   # or: for V in 3.10 3.11 3.12 3.13 3.14; ...
```

Coverage is wired through `pytest-cov`; the 75 % floor is enforced by
`--cov-fail-under=75` in `[tool.pytest.ini_options]`.

## Before pushing: mandatory pre-push gauntlet

CI runs **four** gates that can independently fail, and `ruff check`
passing does **not** imply `ruff format --check` passes. Always run
all four locally before pushing, in this order:

```bash
uv run ruff format        # REWRITES files — run first, review the diff
uv run ruff check         # lint (separate from formatting)
uv run pytest             # full suite + coverage gate
uv run ty check           # type check (best-effort, not yet in CI)
```

Equivalent one-liner: `task check`.

Or, preferred: run the full pre-commit suite so you catch everything
CI checks plus markdownlint and the hygiene hooks:

```bash
uv run prek run --all-files
```

**Do not trust a passing `ruff check` to mean formatting is clean.**
Past regressions: appending new tests without a subsequent
`ruff format` run has silently broken CI's `ruff format --check` step
twice. The `pre-commit` hook also catches this; install with
`uv run prek install` to have it run on every commit.

## High-level architecture

Six modules in `src/pyfinance/`. Public re-exports live in
`pyfinance/__init__.py`; currently only `TSeries` and `TFrame` are
surfaced at the top level.

### `returns.py` — subclassed pandas

`TSeries` and `TFrame` **subclass `pd.Series` and `pd.DataFrame`** and
add the performance-statistics surface (annualized return/vol, drawdown,
Sharpe, capture ratios, CAPM alpha/beta).

Subclassing pandas is delicate. The `_constructor` / `_constructor_expanddim`
properties are what keep operations returning a `TSeries` rather than a
plain `Series` — don't remove them. `TFrame.__init__` currently does not
pass through to `pd.DataFrame.__init__`, so `TFrame(...)` cannot be
constructed like a normal DataFrame; instances arise from operations on
a `TSeries` that dispatch through `_constructor_expanddim`. This is
latent technical debt, not a bug to fix casually.

### `ols.py` — NumPy-backed regression

`OLS` is the static case; `RollingOLS` is the 3-D matrix formulation
(rolling via `.swapaxes(1, 2)` tricks); `PandasRollingOLS` is a thin
Pandas-indexed wrapper.

Most result properties are `@cached_property` — **not** `@property +
@functools.cache` (the latter leaks references to `self`). Preserve this
when adding new result properties.

### `options.py` — vectorized BSM

`BSM` for valuation, Greeks, implied vol; inheritance hierarchy of
strategy classes (`Straddle`, `BullSpread`, `_Butterfly`, `_Condor`,
etc.) underneath `Option` / `OpStrat`.

`kind: Literal["call", "put"]` is the one narrowly-typed choice here;
keep the `Literal` in any signature that proxies to it.

### `general.py` — heterogeneous numerical tools

Active share, amortization schedules, best-fit-distribution search, PCA
on returns, portfolio simulation, tracking-error optimization, VIF.

`factor_loadings` **requires** a `factors` DataFrame from the caller.
The old implicit-fallback to `datasets.load_factors` was removed in
2.0, along with the loader itself.

### `datasets.py` — kept lean after 2.0

Public surface is `load_13f`, `load_industries`, `load_rates`, `load_rf`.
Two details matter:

- **Lazy `pandas_datareader` proxy.** `datasets.pdr` is a
  `_LazyPandasDataReader` instance, *not* the real module. This exists
  because `pandas_datareader` 0.10.0 imports the stdlib `distutils`
  module, which was removed in Python 3.12. The proxy defers the import
  to attribute-access time, so `import pyfinance.datasets` works on
  modern Pythons. Attribute-style use (`pdr.DataReader(...)`) continues
  to work; module-identity checks (`isinstance(pdr, ModuleType)`) will
  not.
- **`load_13f` requires a `user_agent` string.** SEC EDGAR refuses
  requests without a descriptive User-Agent. `load_13f` raises
  `ValueError` before making a request if the argument is empty.

### `utils.py` — frequency and array helpers

Frequency conversion, rolling-window construction, one-hot encoding,
random ticker/weights generation.

The `PERIODS_PER_YEAR` dict is keyed by `FreqGroup.<X>.value` (not the
enum members themselves), because in pandas 2.x `FreqGroup` is a plain
`Enum` rather than `IntEnum`, so a raw-integer lookup against enum-member
keys silently fails.

`get_anlz_factor` contains a try/except fallback that handles both the
old `pandas.tseries.frequencies.get_freq_code` (pandas < 1.5) and the
newer `_libs.tslibs` replacement.

## Version, licensing, metadata

- **Version** lives solely in `pyproject.toml`. `pyfinance.__version__`
  is resolved at runtime via `importlib.metadata.version(__name__)` —
  do *not* hardcode a version in `__init__.py`.
- **Supported Python:** `>=3.10,<3.15`. The ruff `target-version` is
  `py313` (set explicitly despite the minimum being 3.10 — don't change
  without asking).
- **No author emails** in any committed file (`pyproject.toml` has
  `authors = [{ name = "Brad Solomon" }]`, no `email`). RFC 2606
  placeholders like `jane@example.com` in docstring examples are fine.
- `__author__` module attributes were removed in 2.0; don't reintroduce
  them.

## Pandas 2.x gotchas that bit this codebase

All fixed, but worth knowing when editing:

- `DataFrame.fillna(method="ffill")` was removed — use `.ffill()`.
- `pd.read_csv(..., delim_whitespace=True)` was removed — use
  `sep=r"\s+"`.
- `pd.read_excel(..., skip_footer=...)` was renamed to `skipfooter=`.
- `Index.is_all_dates` was removed — use
  `isinstance(idx, pd.DatetimeIndex)`.
- Frequency strings `M`/`Q`/`A` emit `FutureWarning` on pandas 2.2+;
  prefer `ME`/`QE`/`YE` in internal code. User-facing APIs that accept
  legacy codes (e.g., `load_rf(freq="M")`) translate them internally
  before hitting pandas.
- `numpy.lib.nanfunctions` was removed in NumPy 2.0 — import `nansum`,
  `nanmean`, etc. directly from `numpy`.

## Markdown / changelog conventions

- `CHANGELOG.md` follows the [Keep a
  Changelog](https://keepachangelog.com/en/1.1.0/) format: `[Unreleased]`
  at the top, dated version sections, `Added` / `Changed` / `Deprecated`
  / `Removed` / `Fixed` / `Security` subheadings, compare-URL reference
  links at the bottom. Add new entries under `[Unreleased]`.
- `README.md` is linted by `markdownlint-cli2`. Config is in
  `.markdownlint-cli2.yaml` (line length 120, inline HTML allowed).
  CHANGELOG is in the ignore list because external link references and
  historical content don't round-trip cleanly through the linter.

## CI and publishing

Two workflows, both with all third-party actions pinned to SHA256:

- `.github/workflows/ci.yml` — branch/PR workflow: `lint` (ruff), `test`
  (pytest matrix across 3.10–3.14), `build` (sdist + wheel). `uv sync
  --locked` is used everywhere so CI respects the committed lockfile.
- `.github/workflows/publish.yml` — tag-triggered (`v*`), publishes to
  PyPI via OIDC trusted publishing (no API tokens), signs artifacts
  with Sigstore, attaches an SBOM (via `uv export`), and creates a
  GitHub Release. The publish job runs in the `pypi` Environment —
  configure manual-approval there if you want an extra human gate.
  Before publishing, the workflow verifies `git tag` matches
  `pyproject.toml:project.version`.

Dependabot (`.github/dependabot.yml`) opens weekly PRs for
`github-actions` SHAs and `uv` ecosystem updates.
