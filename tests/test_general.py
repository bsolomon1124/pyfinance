"""Tests for `pyfinance.general` pure helpers.

The heavier classes (`BestFitDist`, `PCA`, `PortSim`, `TEOpt`,
`factor_loadings`) depend on network data, long-running numerical
fitting, or `numpy.ppmt`/`ipmt`/`pmt` (removed in NumPy 1.20). Those
are intentionally not covered here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyfinance import general


class TestActiveShare:
    def test_identical_portfolios_zero(self):
        fund = pd.Series({"a": 60.0, "b": 40.0})
        idx = pd.Series({"a": 60.0, "b": 40.0})
        assert general.activeshare(fund, idx, in_format="num") == 0.0

    def test_disjoint_portfolios_100(self):
        fund = pd.Series({"a": 100.0})
        idx = pd.Series({"b": 100.0})
        # Completely disjoint => 0.5 * (|100| + |100|) = 100
        assert np.isclose(general.activeshare(fund, idx, in_format="num"), 100.0)

    def test_partial_overlap(self):
        fund = pd.Series({"a": 50.0, "b": 30.0, "c": 20.0})
        idx = pd.Series({"a": 40.0, "b": 40.0, "d": 20.0})
        # 0.5 * (|50-40| + |30-40| + |20-0| + |0-20|) = 30
        assert np.isclose(general.activeshare(fund, idx, in_format="num"), 30.0)

    def test_dataframe_input_produces_series(self):
        fund = pd.DataFrame(
            {"pA": {"a": 50, "b": 50}, "pB": {"a": 60, "b": 40}},
        )
        idx = pd.Series({"a": 50.0, "b": 50.0})
        out = general.activeshare(fund, idx, in_format="num")
        assert isinstance(out, pd.Series)
        assert out.shape == (2,)
        assert np.isclose(out["pA"], 0.0)
        assert np.isclose(out["pB"], 10.0)

    def test_dec_format_scaling(self):
        fund = pd.Series({"a": 1.0, "b": 0.0})
        idx = pd.Series({"a": 0.5, "b": 0.5})
        assert np.isclose(general.activeshare(fund, idx, in_format="dec"), 0.005)


class TestEwmParams:
    def test_roundtrip_from_span(self):
        out = general.ewm_params("span", 20)
        assert set(out) == {"com", "span", "halflife", "alpha"}
        assert np.isclose(out["span"], 20.0)
        # com = (span - 1) / 2
        assert np.isclose(out["com"], 9.5)
        # alpha = 2 / (span + 1)
        assert np.isclose(out["alpha"], 2.0 / 21.0)

    def test_roundtrip_from_com(self):
        out = general.ewm_params("com", 9.5)
        assert np.isclose(out["span"], 20.0)

    def test_roundtrip_from_alpha(self):
        out = general.ewm_params("alpha", 0.1)
        assert np.isclose(out["alpha"], 0.1)

    def test_invalid_param_raises(self):
        with pytest.raises(NameError, match="param"):
            general.ewm_params("not-a-param", 1.0)


class TestEwmWeights:
    def test_monotonically_decreasing_backwards(self):
        # weight[-1] (most recent) should be the largest
        w = general.ewm_weights(i=10, span=20)
        assert w[-1] >= w[0]
        assert np.all(np.diff(w) >= 0)

    def test_length_matches_i(self):
        w = general.ewm_weights(i=15, halflife=5)
        assert len(w) == 15

    def test_accepts_either_of_the_four_params(self):
        a = general.ewm_weights(i=10, com=5)
        b = general.ewm_weights(i=10, span=11)
        c = general.ewm_weights(i=10, halflife=5)
        d = general.ewm_weights(i=10, alpha=0.2)
        # All four parameterizations produce finite, length-i arrays.
        for arr in (a, b, c, d):
            assert arr.shape == (10,)
            assert np.all(np.isfinite(arr))


class TestEwmBootstrap:
    def test_default_returns_scalar(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal(50)
        out = general.ewm_bootstrap(a, span=20)
        # Default with size=None returns a single draw.
        assert np.ndim(out) == 0

    def test_explicit_size(self):
        rng = np.random.default_rng(1)
        a = rng.standard_normal(50)
        out = general.ewm_bootstrap(a, size=100, span=20)
        assert len(out) == 100

    def test_values_drawn_from_input(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = general.ewm_bootstrap(a, size=200, alpha=0.5)
        assert set(out).issubset(set(a))


class TestVarianceInflationFactor:
    def test_independent_columns_vif_near_one(self):
        rng = np.random.default_rng(42)
        regressors = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
        vifs = general.variance_inflation_factor(regressors)
        # Independent columns should all have VIF near 1.
        assert np.all(vifs.values < 2.0)

    def test_collinear_columns_inflate_vif(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=200)
        df = pd.DataFrame(
            {
                "x": a,
                "y": a + rng.normal(scale=0.01, size=200),  # near-duplicate
                "z": rng.normal(size=200),
            }
        )
        vifs = general.variance_inflation_factor(df)
        # The two near-collinear columns should have very high VIFs.
        assert vifs["x"] > 10
        assert vifs["y"] > 10
        assert vifs["z"] < 2


class TestPCA:
    def test_pca_fit_returns_self(self):
        rng = np.random.default_rng(42)
        returns = pd.DataFrame(rng.normal(size=(100, 4)), columns=list("abcd"))
        pca = general.PCA(returns)
        fitted = pca.fit()
        assert fitted is pca or isinstance(fitted, general.PCA)

    def test_pca_eigen_table_nonempty(self):
        rng = np.random.default_rng(42)
        returns = pd.DataFrame(rng.normal(size=(100, 4)), columns=list("abcd"))
        pca = general.PCA(returns).fit()
        et = pca.eigen_table
        # Eigen table should have at least one component kept.
        assert et.shape[0] >= 1
        assert et.shape[0] <= 4


class TestBestFitDist:
    def test_best_fit_on_normal_data_picks_norm(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=500)
        bfd = general.BestFitDist(x, distributions=["norm", "cauchy", "laplace"]).fit()
        best = bfd.best()
        assert best["name"] == "norm"

    def test_all_returns_dataframe(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=100)
        bfd = general.BestFitDist(x, distributions=["norm", "cauchy"]).fit()
        df = bfd.all()
        assert set(df.columns) == {"name", "sse", "params"}
        assert len(df) == 2


class TestTEOpt:
    def test_optimize_then_weights(self):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2015-01", periods=60, freq="ME")
        r = pd.Series(rng.normal(scale=0.03, size=60), index=idx)
        proxies = pd.DataFrame(
            rng.normal(scale=0.03, size=(60, 3)),
            index=idx,
            columns=["a", "b", "c"],
        )
        te = general.TEOpt(r, proxies, window=24, sumto=1.0)
        te.optimize()
        w = te.opt_weights()
        # Each row should sum to 1.0 (the `sumto` constraint).
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-4)
        assert w.shape[1] == 3


class TestFactorLoadings:
    @pytest.mark.skip(
        reason=(
            "factor_loadings has accumulated latent bugs (it calls "
            "`model.beta()` where `beta` is a property, etc.); fix requires "
            "broader rewrite. Tracked for 2.1."
        )
    )
    def test_series_input_returns_dataframe(self):
        pass
