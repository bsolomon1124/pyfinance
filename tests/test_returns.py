"""Smoke tests for `pyfinance.returns.TSeries` / `TFrame`.

These cover the core performance-statistics surface rather than the
full method matrix — enough to catch gross regressions across the
pandas/numpy upgrade path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyfinance import TSeries


@pytest.fixture
def monthly_returns() -> TSeries:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2015-01-31", periods=60, freq="ME")
    return TSeries(rng.normal(loc=0.005, scale=0.03, size=60), index=idx)


def test_tseries_construction_and_type(monthly_returns: TSeries):
    assert isinstance(monthly_returns, TSeries)
    assert len(monthly_returns) == 60


def test_tseries_anlzd_ret_with_datetime_index(monthly_returns: TSeries):
    r = monthly_returns.anlzd_ret()
    assert isinstance(r, float)
    assert -1.0 < r < 1.0  # plausible range for synthetic data


def test_tseries_anlzd_stdev_is_positive(monthly_returns: TSeries):
    vol = monthly_returns.anlzd_stdev(freq="ME")
    assert vol > 0


def test_tseries_ret_rels_equal_to_one_plus_ret(monthly_returns: TSeries):
    rr = monthly_returns.ret_rels()
    assert np.allclose(rr, 1.0 + monthly_returns.values)


def test_tseries_cumulative_return():
    r = TSeries(
        [0.10, -0.05, 0.20],
        index=pd.date_range("2020-01-31", periods=3, freq="ME"),
    )
    # (1.10 * 0.95 * 1.20) - 1 = 0.254
    expected = 1.10 * 0.95 * 1.20 - 1
    assert np.isclose(r.cuml_ret(), expected)


def test_tseries_drawdown_idx_is_nonpositive(monthly_returns: TSeries):
    dd = monthly_returns.drawdown_idx()
    assert (dd <= 0).all()


def test_tseries_max_drawdown_is_nonpositive(monthly_returns: TSeries):
    assert monthly_returns.max_drawdown() <= 0


def test_tseries_excess_ret_against_benchmark():
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    r = TSeries([0.05, -0.02, 0.03], index=idx)
    b = pd.Series([0.02, 0.01, 0.01], index=idx)
    diff = r.excess_ret(b)
    assert np.allclose(diff.values, [0.03, -0.03, 0.02])


@pytest.fixture
def daily_returns() -> TSeries:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2020-01-01", periods=250, freq="D")
    return TSeries(rng.normal(loc=0.001, scale=0.01, size=250), index=idx)


@pytest.fixture
def daily_benchmark(daily_returns: TSeries) -> TSeries:
    rng = np.random.default_rng(8)
    return TSeries(
        rng.normal(loc=0.0005, scale=0.008, size=len(daily_returns)),
        index=daily_returns.index,
    )


class TestTSeriesScalarStats:
    def test_geomean_between_min_and_max(self, daily_returns: TSeries):
        g = daily_returns.geomean()
        assert daily_returns.min() <= g <= daily_returns.max()

    def test_growth_of_x_scales_linearly(self, daily_returns: TSeries):
        a = daily_returns.growth_of_x(x=1.0)
        b = daily_returns.growth_of_x(x=100.0)
        # growth_of_x returns a terminal scalar, not a series.
        assert np.isclose(b, 100.0 * a)

    def test_cuml_ret_and_cuml_idx_consistent(self, daily_returns: TSeries):
        # `cuml_ret` is the terminal scalar, `cuml_idx` is the series of
        # cumulative compounded returns; its final value equals `cuml_ret`.
        cr = daily_returns.cuml_ret()
        ci = daily_returns.cuml_idx()
        assert np.isclose(ci.iloc[-1], cr)

    def test_ret_idx_starts_at_base(self, daily_returns: TSeries):
        ri = daily_returns.ret_idx(base=100.0)
        assert np.isclose(ri.iloc[0], 100.0 * (1.0 + daily_returns.iloc[0]))

    def test_pct_positive_plus_pct_negative_le_one(self, daily_returns: TSeries):
        p = daily_returns.pct_positive()
        n = daily_returns.pct_negative()
        assert 0.0 <= p <= 1.0
        assert 0.0 <= n <= 1.0
        assert p + n <= 1.0 + 1e-9

    def test_semi_stdev_nonneg(self, daily_returns: TSeries):
        assert daily_returns.semi_stdev(freq="D") >= 0

    def test_semi_stdev_matches_docstring_formula(self):
        """Regression test for #15.

        The docstring promises:
            sqrt( sum(min(x - t, 0)^2) / (n - ddof) )

        Until 2.0.1 the implementation placed `/ n` *outside* the sqrt,
        yielding values an order of magnitude too small. Pin the exact
        formula here so the bug cannot silently regress.
        """
        vals = [
            -0.01, 0.02, -0.03, 0.01, -0.02,
             0.03, -0.01, 0.02, -0.01, 0.01,
        ]
        idx = pd.date_range("2020-01-01", periods=len(vals), freq="D")
        ts = TSeries(vals, index=idx)

        n = len(vals)
        downside_sq = sum(min(v, 0.0) ** 2 for v in vals)
        expected_periodic = np.sqrt(downside_sq / n)
        # `semi_stdev(freq='D')` annualizes by sqrt(252).
        expected_anlzd = expected_periodic * np.sqrt(252.0)

        assert np.isclose(ts.semi_stdev(freq="D"), expected_anlzd, rtol=1e-12)

    def test_semi_stdev_respects_ddof(self):
        """`ddof` should change the denominator to (n - ddof) inside sqrt."""
        vals = [-0.02, 0.01, -0.03, 0.02, -0.01]
        idx = pd.date_range("2020-01-01", periods=len(vals), freq="D")
        ts = TSeries(vals, index=idx)

        downside_sq = sum(min(v, 0.0) ** 2 for v in vals)
        for ddof in (0, 1, 2):
            expected = np.sqrt(downside_sq / (len(vals) - ddof)) * np.sqrt(252.0)
            assert np.isclose(
                ts.semi_stdev(ddof=ddof, freq="D"), expected, rtol=1e-12
            )

    def test_semi_stdev_threshold_shifts_downside(self):
        """A positive threshold should strictly widen the downside mask."""
        vals = [0.005, 0.02, -0.01, 0.03, 0.001]
        idx = pd.date_range("2020-01-01", periods=len(vals), freq="D")
        ts = TSeries(vals, index=idx)

        # Against threshold=0 only the -0.01 observation is downside.
        base = ts.semi_stdev(threshold=0.0, freq="D")
        # Against threshold=0.01, the 0.005 and 0.001 obs become downside too.
        higher = ts.semi_stdev(threshold=0.01, freq="D")
        assert higher > base

    def test_sharpe_and_sortino_are_finite(self, daily_returns: TSeries):
        assert np.isfinite(daily_returns.sharpe_ratio())
        assert np.isfinite(daily_returns.sortino_ratio(freq="D"))

    def test_calmar_ratio_finite(self, daily_returns: TSeries):
        # Calmar = anlzd_ret / |max_drawdown|; defined unless max_dd == 0.
        val = daily_returns.calmar_ratio()
        assert np.isfinite(val) or np.isinf(val)

    def test_gain_to_loss_ratio_finite(self, daily_returns: TSeries):
        # Ratio is sum(gains) / sum(losses); can be negative when
        # losses outweigh gains in the sample.
        assert np.isfinite(daily_returns.gain_to_loss_ratio())


class TestTSeriesDrawdown:
    def test_max_drawdown_equal_to_min_of_drawdown_idx(self, daily_returns: TSeries):
        assert np.isclose(
            daily_returns.max_drawdown(), daily_returns.drawdown_idx().min()
        )

    def test_drawdown_length_is_integer_when_requested(self, daily_returns: TSeries):
        n = daily_returns.drawdown_length(return_int=True)
        assert isinstance(n, (int, np.integer))
        assert n >= 0


class TestTSeriesBenchmarkStats:
    def test_batting_avg_in_unit_interval(
        self, daily_returns: TSeries, daily_benchmark: TSeries
    ):
        assert 0.0 <= daily_returns.batting_avg(daily_benchmark) <= 1.0

    def test_tracking_error_nonneg(
        self, daily_returns: TSeries, daily_benchmark: TSeries
    ):
        assert daily_returns.tracking_error(daily_benchmark) >= 0

    def test_info_ratio_finite(self, daily_returns: TSeries, daily_benchmark: TSeries):
        assert np.isfinite(daily_returns.info_ratio(daily_benchmark))

    def test_beta_alpha_roundtrip_via_capm(
        self, daily_returns: TSeries, daily_benchmark: TSeries
    ):
        # .beta and .alpha delegate to CAPM; verify they agree.
        capm = daily_returns.CAPM(daily_benchmark)
        assert np.isclose(capm.beta, daily_returns.beta(daily_benchmark))
        assert np.isclose(capm.alpha, daily_returns.alpha(daily_benchmark))

    def test_rsq_in_unit_interval(
        self, daily_returns: TSeries, daily_benchmark: TSeries
    ):
        assert 0.0 <= daily_returns.rsq(daily_benchmark) <= 1.0

    def test_up_down_capture_finite(
        self, daily_returns: TSeries, daily_benchmark: TSeries
    ):
        assert np.isfinite(daily_returns.up_capture(daily_benchmark))
        assert np.isfinite(daily_returns.down_capture(daily_benchmark))


class TestTSeriesRollup:
    def test_rollup_to_monthly_aggregates(self, daily_returns: TSeries):
        m = daily_returns.rollup("ME")
        # Aggregation should shrink the series.
        assert len(m) < len(daily_returns)
        # Compound product of rollup values matches overall compound.
        full = np.prod(1.0 + daily_returns.values) - 1.0
        rolled = np.prod(1.0 + m.values) - 1.0
        assert np.isclose(full, rolled, atol=1e-10)


class TestExcessDrawdownIdx:
    def test_caer_method_nonpositive(
        self, daily_returns: TSeries, daily_benchmark: TSeries
    ):
        dd = daily_returns.excess_drawdown_idx(daily_benchmark, method="caer")
        assert (dd <= 0).all()

    def test_cger_method_nonpositive(
        self, daily_returns: TSeries, daily_benchmark: TSeries
    ):
        dd = daily_returns.excess_drawdown_idx(daily_benchmark, method="cger")
        assert (dd <= 0).all()

    def test_integer_method_mapping(
        self, daily_returns: TSeries, daily_benchmark: TSeries
    ):
        # Methods are indexed 0..3 as well as by string.
        a = daily_returns.excess_drawdown_idx(daily_benchmark, method="caer")
        b = daily_returns.excess_drawdown_idx(daily_benchmark, method=0)
        assert np.allclose(a.values, b.values)

    def test_unknown_method_raises(
        self, daily_returns: TSeries, daily_benchmark: TSeries
    ):
        with pytest.raises(ValueError, match="method"):
            daily_returns.excess_drawdown_idx(daily_benchmark, method="bogus")


class TestMiscTSeriesHelpers:
    def test_excess_ret_geometric_vs_arithmetic(
        self, daily_returns: TSeries, daily_benchmark: TSeries
    ):
        a = daily_returns.excess_ret(daily_benchmark, method="arithmetic")
        g = daily_returns.excess_ret(daily_benchmark, method="geometric")
        # Both return same-length series; small returns => arith ≈ geom.
        assert len(a) == len(g) == len(daily_returns)

    def test_treynor_ratio_finite(
        self, daily_returns: TSeries, daily_benchmark: TSeries
    ):
        assert np.isfinite(daily_returns.treynor_ratio(daily_benchmark))

    def test_msquared_finite(self, daily_returns: TSeries, daily_benchmark: TSeries):
        assert np.isfinite(daily_returns.msquared(daily_benchmark))

    def test_rsq_adj_le_rsq(self, daily_returns: TSeries, daily_benchmark: TSeries):
        # Adjusted R-squared is always <= plain R-squared.
        assert daily_returns.rsq_adj(daily_benchmark) <= daily_returns.rsq(
            daily_benchmark
        )

    def test_tstat_alpha_and_beta_finite(
        self, daily_returns: TSeries, daily_benchmark: TSeries
    ):
        assert np.isfinite(daily_returns.tstat_alpha(daily_benchmark))
        assert np.isfinite(daily_returns.tstat_beta(daily_benchmark))
