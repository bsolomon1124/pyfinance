"""Tests for pyfinance.utils pure helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyfinance import utils


def test_get_anlz_factor_common_frequencies():
    assert utils.get_anlz_factor("D") == 252.0
    assert utils.get_anlz_factor("W") == 52.0
    assert utils.get_anlz_factor("M") == 12.0
    assert utils.get_anlz_factor("Q") == 4.0


def test_get_anlz_factor_anchored_variants():
    assert utils.get_anlz_factor("Q-DEC") == 4.0
    assert utils.get_anlz_factor("BQS-APR") == 4.0
    assert utils.get_anlz_factor("MS") == 12.0


def test_get_anlz_factor_invalid_raises():
    with pytest.raises(ValueError, match="Invalid frequency"):
        utils.get_anlz_factor("not-a-freq")


def test_equal_weights_default():
    w = utils.equal_weights(n=5)
    assert w.shape == (5,)
    assert np.isclose(w.sum(), 1.0)
    assert np.allclose(w, 0.2)


def test_equal_weights_scaled():
    w = utils.equal_weights(n=4, sumto=2.0)
    assert np.isclose(w.sum(), 2.0)


def test_random_weights_1d():
    w = utils.random_weights(size=5)
    assert w.shape == (5,)
    assert np.isclose(w.sum(), 1.0)


def test_random_weights_2d():
    w = utils.random_weights(size=(3, 5))
    assert w.shape == (3, 5)
    assert np.allclose(w.sum(axis=1), 1.0)


def test_random_weights_with_sumto():
    w = utils.random_weights(size=4, sumto=3.0)
    assert np.isclose(w.sum(), 3.0)


def test_random_tickers_uniqueness_and_length():
    tickers = utils.random_tickers(length=4, n_tickers=10, endswith="X")
    assert len(tickers) == 10
    assert len({*tickers}) == 10
    assert all(len(t) == 4 and t.endswith("X") for t in tickers)


def test_rolling_windows_basic():
    a = np.arange(10)
    wins = utils.rolling_windows(a, window=3)
    assert wins.shape == (8, 3)
    assert np.array_equal(wins[0], [0, 1, 2])
    assert np.array_equal(wins[-1], [7, 8, 9])


def test_rolling_windows_too_large():
    a = np.arange(5)
    with pytest.raises(ValueError, match="window"):
        utils.rolling_windows(a, window=10)


def test_public_dir_filters_underscores_when_requested():
    class _Obj:
        pass

    o = _Obj()
    o.visible = 1
    o._hidden = 2
    # max_underscores=1 excludes anything starting with '_' (one or more)
    attrs = utils.public_dir(o, max_underscores=1)
    assert "visible" in attrs
    assert "_hidden" not in attrs
    assert "__class__" not in attrs


def test_public_dir_type_filter():
    class _Obj:
        pass

    o = _Obj()
    o.an_int = 1
    o.a_str = "hello"
    attrs = utils.public_dir(o, max_underscores=1, type_=int)
    assert "an_int" in attrs
    assert "a_str" not in attrs


def test_view_limits_rows_and_columns():
    df = pd.DataFrame(np.arange(100).reshape(20, 5))
    out = utils.view(df, row=3, col=2)
    assert out.shape == (3, 2)


def test_expanding_stdize_shape_preserved():
    df = pd.DataFrame(np.random.default_rng(0).standard_normal((12, 3)))
    out = utils.expanding_stdize(df, min_periods=5)
    assert out.shape == df.shape
    assert out.iloc[:4].isna().all().all()


def test_can_broadcast_true_and_false():
    assert utils.can_broadcast(np.zeros(3), np.zeros((2, 3)))
    assert not utils.can_broadcast(np.zeros(3), np.zeros(4))


def test_isiterable():
    assert utils.isiterable([1, 2, 3])
    assert utils.isiterable("abc")
    assert not utils.isiterable(42)


def test_avail_reports_first_last_valid():
    df = pd.DataFrame(
        {"a": [np.nan, 1.0, 2.0, np.nan], "b": [0.0, np.nan, 1.0, 2.0]},
        index=pd.date_range("2020-01-01", periods=4, freq="D"),
    )
    out = utils.avail(df)
    assert out.columns.tolist() == ["start", "end"]
    assert out.loc["a", "start"] == pd.Timestamp("2020-01-02")
    assert out.loc["a", "end"] == pd.Timestamp("2020-01-03")
    assert out.loc["b", "start"] == pd.Timestamp("2020-01-01")
    assert out.loc["b", "end"] == pd.Timestamp("2020-01-04")


def test_dropcols_removes_nan_columns():
    df = pd.DataFrame(
        {"ok": [1.0, 2.0, 3.0], "bad": [1.0, np.nan, 3.0]},
        index=pd.date_range("2020-01-01", periods=3, freq="D"),
    )
    out = utils.dropcols(df)
    assert "ok" in out.columns
    assert "bad" not in out.columns


def test_encode_one_hot_on_strings():
    out = utils.encode("the cat sat", "the dog sat")
    # Both sentences share "the" and "sat"; each should be a 1/0 vector.
    assert out.shape[0] == 2
    assert int(out.sum()) > 0


def test_constrain_intersects_indices():
    a = pd.Series([1, 2, 3], index=[1, 2, 3])
    b = pd.Series([10, 20, 30], index=[2, 3, 4])
    a2, b2 = utils.constrain(a, b)
    assert set(a2.index) == set(b2.index) == {2, 3}


def test_unique_everseen_preserves_order():
    assert list(utils.unique_everseen([3, 1, 2, 3, 1, 4])) == [3, 1, 2, 4]


def test_uniqify_preserves_order():
    assert utils.uniqify([3, 1, 2, 3, 1, 4]) == [3, 1, 2, 4]


class TestConstrainHorizon:
    @pytest.fixture
    def daily(self) -> pd.Series:
        idx = pd.date_range("2015-01-01", "2020-12-31", freq="D")
        return pd.Series(np.arange(len(idx)), index=idx)

    def test_years_trims_window(self, daily: pd.Series):
        out = utils.constrain_horizon(daily, years=1)
        # Exactly 1y lookback from the last index.
        assert out.index[-1] == daily.index[-1]
        # Start should be ~365d before (allow calendar slack).
        span = (out.index[-1] - out.index[0]).days
        assert 360 <= span <= 371

    def test_cust_string_years(self, daily: pd.Series):
        out = utils.constrain_horizon(daily, cust="3y")
        span = (out.index[-1] - out.index[0]).days
        assert 1090 <= span <= 1100

    def test_cust_string_months(self, daily: pd.Series):
        out = utils.constrain_horizon(daily, cust="6m")
        span = (out.index[-1] - out.index[0]).days
        assert 180 <= span <= 185

    def test_cust_words(self, daily: pd.Series):
        out = utils.constrain_horizon(daily, cust="three years")
        span = (out.index[-1] - out.index[0]).days
        assert 1090 <= span <= 1100

    def test_cust_and_explicit_both_raise(self, daily: pd.Series):
        with pytest.raises(ValueError, match="both"):
            utils.constrain_horizon(daily, cust="1y", months=3)

    def test_strict_pre_dates_raises(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        s = pd.Series(np.arange(10), index=idx)
        with pytest.raises(ValueError, match="pre-dates"):
            utils.constrain_horizon(s, years=5, strict=True)

    def test_unknown_cust_raises(self, daily: pd.Series):
        with pytest.raises(ValueError, match="cust"):
            utils.constrain_horizon(daily, cust="nonsense")


def test_appender_decorator_adds_params_section():
    ddocs = {
        "x": "\nx : int\n    An integer parameter.\n",
        "y": "\ny : str\n    A string parameter.\n",
    }

    @utils.appender(ddocs)
    def fn(x, y):
        """Original docstring."""
        return x

    assert "Original docstring." in fn.__doc__
    assert "x : int" in fn.__doc__
    assert "y : str" in fn.__doc__
