"""Strategy-class tests for `pyfinance.options`.

These complement `test_options.py` (which focuses on `BSM` /
`Straddle` / `BullSpread` / `Butterfly`). This file exercises the
`Call`/`Put` primitives and a broader sweep of strategy payoff /
profit outputs.
"""

from __future__ import annotations

import numpy as np

from pyfinance import options


class TestCall:
    def test_long_call_payoff(self):
        call = options.Call(K=100, price=5, St=np.array([90, 100, 110]))
        # Payoff: max(St - K, 0)
        assert np.allclose(call.payoff(), [0, 0, 10])

    def test_long_call_profit(self):
        call = options.Call(K=100, price=5, St=np.array([90, 105, 120]))
        # Profit: payoff - price
        assert np.allclose(call.profit(), [-5, 0, 15])

    def test_short_call_profit(self):
        call = options.Call(K=100, price=5, St=np.array([90, 100, 120]), pos="short")
        # Short writer receives premium, pays out on upside
        assert np.allclose(call.profit(), [5, 5, -15])


class TestPut:
    def test_long_put_payoff(self):
        put = options.Put(K=100, price=3, St=np.array([80, 100, 110]))
        # Payoff: max(K - St, 0)
        assert np.allclose(put.payoff(), [20, 0, 0])

    def test_long_put_profit(self):
        put = options.Put(K=100, price=3, St=np.array([80, 97, 110]))
        assert np.allclose(put.profit(), [17, 0, -3])


class TestBullBearSpreads:
    def test_bull_spread_payoff_caps_at_width(self):
        bs = options.BullSpread(
            St=np.array([90, 100, 110, 120]),
            K1=100,
            K2=110,
            price1=5,
            price2=2,
            kind="call",
        )
        # Payoff: max(St-K1,0) - max(St-K2,0) in [0, 10]
        assert np.allclose(bs.payoff(), [0, 0, 10, 10])

    def test_bear_spread_payoff_bounded(self):
        bs = options.BearSpread(
            St=np.array([80, 90, 100, 110]),
            K1=100,
            K2=90,
            price1=5,
            price2=2,
            kind="put",
        )
        # Payoff magnitude is bounded by K1-K2=10 and is zero for St>=K1.
        payoff = bs.payoff()
        assert np.max(np.abs(payoff)) <= 10 + 1e-9
        assert payoff[2] == 0
        assert payoff[3] == 0


class TestStraddleStrangle:
    def test_straddle_payoff_is_symmetric_absolute(self):
        s = options.Straddle(
            St=np.array([80, 100, 120]),
            K=100,
            callprice=5,
            putprice=5,
        )
        # Payoff: |St-K| => 20, 0, 20
        assert np.allclose(s.payoff(), [20, 0, 20])

    def test_strangle_zero_inside_strikes(self):
        s = options.Strangle(
            St=np.array([95, 100, 105, 115]),
            K1=110,
            K2=90,
            callprice=2,
            putprice=2,
        )
        # Call strike 110, put strike 90. Zero payoff in [90,110]. At 115: 5.
        payoff = s.payoff()
        assert payoff[1] == 0  # 100
        assert payoff[2] == 0  # 105
        assert payoff[3] == 5  # 115 - 110


class TestBSMExtras:
    def test_bsm_rejects_unknown_kind(self):
        import pytest

        with pytest.raises(ValueError, match="kind"):
            options.BSM(S0=100, K=100, T=1, r=0.04, sigma=0.2, kind="foo")

    def test_bsm_put_call_parity(self):
        """C - P = S - K * exp(-rT) for identical S, K, T, r, sigma."""
        c = options.BSM(S0=100, K=100, T=1, r=0.05, sigma=0.2, kind="call").value()
        p = options.BSM(S0=100, K=100, T=1, r=0.05, sigma=0.2, kind="put").value()
        rhs = 100 - 100 * np.exp(-0.05 * 1)
        assert np.isclose(c - p, rhs, atol=1e-6)

    def test_bsm_gamma_matches_manual_call_and_put(self):
        """Gamma is the same for a call and a put at the same parameters."""
        c = options.BSM(S0=100, K=100, T=1, r=0.05, sigma=0.2, kind="call").gamma()
        p = options.BSM(S0=100, K=100, T=1, r=0.05, sigma=0.2, kind="put").gamma()
        assert np.isclose(c, p)

    def test_bsm_implied_vol_recovers_sigma(self):
        sigma = 0.22
        bsm = options.BSM(S0=100, K=100, T=0.5, r=0.03, sigma=sigma)
        price = bsm.value()
        iv = bsm.implied_vol(price)
        assert np.isclose(iv, sigma, atol=1e-4)


class TestButterflyAndCondor:
    def test_long_call_butterfly_peak_at_middle_strike(self):
        # Strikes: 90, 100, 110. Payoff peaks at K2.
        bf = options.LongButterfly(
            St=np.array([80, 90, 100, 110, 120]),
            K1=90,
            K2=100,
            K3=110,
            price1=12,
            price2=5,
            price3=1,
            kind="call",
        )
        payoff = bf.payoff()
        # Max payoff should be at middle strike (index 2).
        assert np.argmax(payoff) == 2
        # Wings pay zero.
        assert payoff[0] == 0
        assert payoff[-1] == 0

    def test_short_butterfly_inverts_sign(self):
        bf_long = options.LongButterfly(
            St=np.array([90, 100, 110]),
            K1=90,
            K2=100,
            K3=110,
            price1=12,
            price2=5,
            price3=1,
            kind="call",
        )
        bf_short = options.ShortButterfly(
            St=np.array([90, 100, 110]),
            K1=90,
            K2=100,
            K3=110,
            price1=12,
            price2=5,
            price3=1,
            kind="call",
        )
        assert np.allclose(bf_long.payoff(), -bf_short.payoff())

    def test_long_condor_payoff_shape(self):
        # Long condor: peak payoff somewhere in the middle, zero on both
        # wings. Using a symmetric example keeps the assertion robust.
        cd = options.LongCondor(
            St=np.array([70, 85, 95, 105, 115]),
            K1=80,
            K2=90,
            K3=100,
            K4=110,
            price1=15,
            price2=8,
            price3=3,
            price4=1,
            kind="call",
        )
        payoff = cd.payoff()
        # Wings are zero, middle is strictly positive.
        assert payoff[0] == 0
        assert payoff[-1] == 0
        assert np.max(payoff) > 0


class TestStripStrapShortStraddle:
    def test_strip_favors_downside(self):
        s = options.Strip(
            St=np.array([80, 100, 120]),
            K=100,
            callprice=5,
            putprice=5,
        )
        # Strip = 1 call + 2 puts; downside should dominate upside.
        payoff = s.payoff()
        assert payoff[0] > payoff[2]

    def test_short_straddle_caps_at_premium(self):
        s = options.ShortStraddle(
            St=np.array([80, 100, 120]),
            K=100,
            price=10,
        )
        profit = s.profit()
        # Max profit at K, equals premium received (2 * price).
        assert np.isclose(profit[1], 20)
