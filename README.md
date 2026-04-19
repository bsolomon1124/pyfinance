# pyfinance

[![PyPI version](https://img.shields.io/pypi/v/pyfinance.svg)](https://pypi.org/project/pyfinance/)
[![License: MIT](https://img.shields.io/pypi/l/pyfinance.svg)](https://github.com/bsolomon1124/pyfinance/blob/master/LICENSE.txt)
[![Python versions](https://img.shields.io/pypi/pyversions/pyfinance.svg)](https://pypi.org/project/pyfinance/)

pyfinance is a Python package for investment management and analysis of
security returns. It complements existing quantitative-finance packages
such as [pandas-datareader][pdr] with a focused, batteries-included
statistical toolkit.

[pdr]: https://github.com/pydata/pandas-datareader

## Status

- **Latest release:** 2.0.1
- **Python:** 3.10, 3.11, 3.12, 3.13, 3.14
- **License:** MIT

## Installation

```bash
uv add pyfinance
```

Or with `pip`:

```bash
pip install pyfinance
```

## Modules

| Module | Description |
| ------ | ----------- |
| `pyfinance.returns` | Statistical analysis of financial time series via the CAPM framework. `TSeries` / `TFrame` are Pandas subclasses that add performance statistics: annualized return/vol, Sharpe, Sortino, drawdown, capture ratios, alpha/beta, Information Ratio, and so on. |
| `pyfinance.ols` | Ordinary least-squares regression. `OLS`, `RollingOLS` (NumPy-backed), and `PandasRollingOLS` (Pandas-indexed wrapper). |
| `pyfinance.options` | Vectorized Black-Scholes-Merton valuation, Greeks, and implied volatility via `BSM`. Option strategies (`Straddle`, `BullSpread`, `Butterfly`, `Condor`, variants). |
| `pyfinance.general` | General-purpose computations: active share, amortization schedules, best-fit distribution, PCA on returns, portfolio simulation, tracking-error optimization, VIF. |
| `pyfinance.datasets` | A small set of dataset loaders: `load_13f` (SEC EDGAR), `load_industries` (Ken French), `load_rates` (FRED H.15), `load_rf` (3-month T-bill total-return series). |
| `pyfinance.utils` | Frequency conversion, rolling-window construction, one-hot encoding, random ticker/weights generation, availability reporting. |

## Quick tutorial

### `TSeries` — performance statistics

```python
import numpy as np
import pandas as pd
from pyfinance import TSeries

rng = np.random.default_rng(444)
s = rng.standard_normal(400) / 100 + 0.0008
idx = pd.date_range(start="2016-01-01", periods=len(s))
ts = TSeries(s, index=idx)

ts.anlzd_ret()       # annualized geometric return
ts.anlzd_stdev("D")  # annualized stdev of returns
ts.max_drawdown()    # worst peak-to-trough decline
ts.sharpe_ratio()    # Sharpe, annualized
```

### `BSM` — Black-Scholes-Merton options pricing

```python
from pyfinance.options import BSM

op = BSM(S0=100, K=100, T=1, r=0.04, sigma=0.20)
op.value()          # European call value
op.delta()          # Greeks
op.implied_vol(10)  # implied vol at a target price

# Vectorized across arrays of strikes.
import numpy as np
ops = BSM(S0=100, K=np.arange(100, 110), T=1, r=0.04, sigma=0.20)
ops.value()
```

### `OLS` — regression

```python
from pyfinance.ols import OLS

# y: 1-d array; x: 2-d array of explanatory variables.
model = OLS(y=y, x=x)
model.beta, model.alpha
model.rsq_adj, model.fstat, model.std_err
```

### `datasets.load_13f` — SEC EDGAR 13F parser

SEC EDGAR requires a descriptive `User-Agent`
([policy][edgar-policy]). Supply one:

[edgar-policy]: https://www.sec.gov/os/accessing-edgar-data

```python
from pyfinance import datasets

url = (
    "https://www.sec.gov/Archives/edgar/data/1040273/"
    "000108514617001787/form13fInfoTable.xml"
)
df = datasets.load_13f(url, user_agent="Jane Doe (jane@example.com)")
```

## Development

This repo is managed with [uv][uv]:

[uv]: https://docs.astral.sh/uv/

```bash
git clone https://github.com/bsolomon1124/pyfinance
cd pyfinance
uv sync               # install pinned runtime + dev deps
uv run pytest         # run tests
uv run ruff check     # lint
uv run ruff format    # format
uv build              # build sdist + wheel
```

Type checking uses [ty][ty]:

[ty]: https://github.com/astral-sh/ty

```bash
uv run ty check
```

CI runs these across Python 3.10–3.14 on every push; see
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Changelog

See [CHANGELOG.md](CHANGELOG.md). Notably, pyfinance 2.0 dropped
`load_factors`, `load_shiller`, and `load_retaildata` because their
upstream sources had drifted past reasonable repair. See the 2.0
release notes for details and migration guidance.

## License

MIT — see [LICENSE.txt](LICENSE.txt).
