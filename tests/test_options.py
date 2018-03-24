# flake8: noqa

# Test cases taken from:
# - Thomas Ho Company LTD: financial models,
#   http://www.thomasho.com/mainpages/analysoln.asp
# - Analysis of Derivatives for the Chartered Financial Analyst® Program,
#   Don M. Chance, PhD, CFA, ©2003 CFA Institute

import types

import numpy as np
import pandas as pd

from pyfinance.options import *

np.random.seed(123)
RTOL = 1e-03

# BSM
# ---------------------------------------------------------------------

s, k, t, sigma, r = 100., 100., 1., 0.2, 0.04

greeks = {
    'call': (0.3, 0.1, 0.61791, 0.01907, 38.13878, -5.88852, 51.86609,
             6.22577),
    'put' : (0.3, 0.1, -0.38209, 0.01907, 38.13878, -2.04536, -44.21286,
             -6.36390)
    }
names = ('d1', 'd2', 'delta', 'gamma', 'vega', 'theta', 'rho', 'omega')
target = {
    'call': dict(zip(names, greeks['call'])),
    'put': dict(zip(names, greeks['put']))
    }

target['call'].update({'value': 9.92505})
target['put'].update({'value': 6.00400})

options = {
    'call': BSM(S0=s, K=k, T=t, r=r, sigma=sigma, kind='call'),
    'put': BSM(S0=s, K=k, T=t, r=r, sigma=sigma, kind='put')
    }

def test_BSM():
    for name, option in options.items():
        for k, v in target[name].items():
            if isinstance(getattr(option, k), types.MethodType):
                assert np.allclose(v, getattr(option, k)(), rtol=RTOL)
            else:
                assert np.allclose(v, getattr(option, k), rtol=RTOL)

# Put/call
# ---------------------------------------------------------------------

k, price, s = 2000., 81.75, np.array([1900., 2100.])

call = Call(K=k, price=price, St=s, pos='long')
put = Put(K=k, price=price, St=s, pos='long')


def test_put_and_call():
    assert np.allclose(call.payoff(), np.array([   0.,  100.]))
    assert np.allclose(call.profit(), np.array([-81.75,  18.25]))

    assert np.allclose(put.payoff(), np.array([ 100.,    0.]))
    assert np.allclose(put.profit(), np.array([ 18.25, -81.75]))


# Options strategies
# ---------------------------------------------------------------------

# Straddle', 'ShortStraddle', 'Strangle',
#     'ShortStrangle', 'Strip', 'Strap', 'BullSpread', 'BearSpread',
#     'LongPutLadder', 'ShortPutLadder', 'LongButterfly', 'ShortButterfly',
#     'LongIronButterfly', 'ShortIronButterfly', 'LongCondor', 'ShortCondor',
#     'LongIronCondor', 'ShortIronCondor'

s = np.array([2100, 2000, 1900])
k1 = 1950.
k2 = 2050.
p1 = 108.43
p2 = 59.98

bullspread = BullSpread(St=s, K1=k1, K2=k2, price1=p1, price2=p2)

p1 = 56.01
p2 = 107.39
bearspread = BearSpread(St=s, K1=k1, K2=k2, price1=p1, price2=p2)


# TODO
# bs = {
#     'call': BullSpread(St=s, K1=k1, K2=k2, price1=p1, price2=p2, kind='call'),
#     'put': BullSpread(St=s, K1=k1, K2=k2, price1=p1, price2=p2, kind='put')
#     }

s = np.array([1900., 1975., 2025., 2100.])
k1, k2, k3 = 1950., 2000., 2050.
p1, p2, p3 = 108.43, 81.75, 59.98

bfly = LongButterfly(St=s, K1=k1, K2=k2, K3=k3,
                     price1=p1, price2=p2, price3=p3, kind='call')

s = np.array([2100., 1900.])
k = 2000
c = 81.75
p = 79.25


straddle = Straddle(St=s, K=k, callprice=c, putprice=p)

def test_opstrats():
    assert np.allclose(bullspread.payoff(), np.array([ 100.,   50.,    0.]), rtol=RTOL)
    assert np.allclose(bullspread.profit(), np.array([ 51.55,   1.55, -48.45]), rtol=RTOL)

    assert np.allclose(bearspread.payoff(), np.array([   0.,   50.,  100.]), rtol=RTOL)
    assert np.allclose(bearspread.profit(), np.array([-51.38,  -1.38,  48.62]), rtol=RTOL)

    assert np.allclose(bfly.payoff(), np.array([  0.,  25.,  25.,   0.]), rtol=RTOL)
    assert np.allclose(bfly.profit(), np.array([ -4.91,  20.09,  20.09,  -4.91]), rtol=RTOL)

    assert np.allclose(straddle.payoff(), np.array([ 100.,  100.]), rtol=RTOL)
    assert np.allclose(straddle.profit(), np.array([-61., -61.]), rtol=RTOL)
