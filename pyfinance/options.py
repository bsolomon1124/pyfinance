"""Vectorized options calculations.

Descriptions
============
BSM
    Black-Scholes Merton European option valuation, Greeks, and implied vol.

Option strategies inheritance hierarchy:
- Option
      - Call
      - Put
- OpStrat
      - Straddle
            - Strip
            - Strap
      - ShortStraddle
      - Strangle
      - ShortStrangle
      - BullSpread
      - BearSpread
            - LongPutLadder
            - ShortPutLadder
      - _Butterfly
            - LongButterfly
            - ShortButterfly
            - LongIronButterfly
            - ShortIronButterfly
      - _Condor
            - LongCondor
            - ShortCondor
            - LongIronCondor
            - ShortIronCondor
"""

__author__ = 'Brad Solomon <brad.solomon.1124@gmail.com>'

__all__ = [
    'BSM', 'Call', 'Put', 'OpStrat', 'Straddle', 'ShortStraddle', 'Strangle', 
    'ShortStrangle', 'Strip', 'Strap', 'BullSpread', 'BearSpread', 
    'LongPutLadder', 'ShortPutLadder', 'LongButterfly', 'ShortButterfly', 
    'LongIronButterfly', 'ShortIronButterfly', 'LongCondor', 'ShortCondor', 
    'LongIronCondor', 'ShortIronCondor'
    ]

from collections import OrderedDict
import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from scipy.stats import norm

# TODO:
# - __repr__/__str__ for strategies
# - time spreads

class BSM(object):
    """Compute European option value, greeks, and implied vol, using BSM.

    Parameters
    ==========
    S0 : int or float
        initial asset value
    K : int or float
        strike
    T : int or float
        time to expiration as a fraction of one year
    r : int or float
        continuously compounded risk free rate, annualized
    sigma : int or float
        continuously compounded standard deviation of returns, annualized
    kind : str, {'call', 'put'}, default 'call'
        type of option

    Example
    =======
    op = Option(S0=100, K=100, T=1, r=.04, sigma=.2)

    print(op)
    Option(kind=call, S0=100.00, K=100.00, T=1.00, r=0.04, sigma=0.20)

    op.summary()
    Out[89]:
    OrderedDict([('Value', 9.9250537172744373),
                 ('d1', 0.29999999999999999),
                 ('d2', 0.099999999999999978),
                 ('Delta', 0.61791142218895256),
                 ('Gamma', 0.019069390773026208),
                 ('Vega', 38.138781546052414),
                 ('Theta', -5.8885216946700742),
                 ('Rho', 51.866088501620823),
                 ('Omega', 6.2257740843607241)])

    # This class supports vectorized inputs.  Use NumPy arrays.

    ops = Option(S0=100, K=np.arange(100, 110), T=1, r=.04, sigma=.2)
    print(ops.value())
    [ 9.92505372  9.41590973  8.92571336  8.4543027   8.00147434  7.56698592
      7.15055895  6.75188165  6.37061188  6.00638013]

    # Note that if multiple parameters are array-like, the resulting methods
    #     will be evaluated in a zipped element-wise fashion between them.

    ops = Option(S0=np.arange(100, 110), K=np.arange(100, 110), T=1, r=.04,
                 sigma=.2)

    print(ops.value())
    [  9.92505372  10.02430425  10.12355479  10.22280533  10.32205587
      10.4213064   10.52055694  10.61980748  10.71905801  10.81830855]
    """

    def __init__(self, S0, K, T, r, sigma, kind='call'):
        kind = kind.lower()
        if kind not in ['call', 'put']:
            raise ValueError("`kind` must be in ('call', 'put')")

        self.kind = kind
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

        self.d1 = ((np.log(self.S0 / self.K)
                + (self.r + 0.5 * self.sigma ** 2) * self.T)
                / (self.sigma * np.sqrt(self.T)))
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)

        # Several greeks use negated terms dependent on option type.
        # For example, delta of call is N(d1) and delta put is N(d1) - 1
        #     and theta may use N(d2) or N(-d2).  In these lists:
        #     - element 0 (0, -1) is used in delta and omega
        #     - element 1 (1, -1) is used in rho and theta
        #     - negated 1 (-1, 1) is used in theta

        self._sign = {'call' : [0, 1], 'put' : [-1, -1]}
        self._sign = self._sign[self.kind]

    def __repr__(self):
        return ('BSM(kind={0}, S0={1:0.2f}, K={2:0.2f}, '
                'T={3:0.2f}, r={4:0.2f}, sigma={5:0.2f})'
                .format(self.kind, self.S0, self.K,
                 self.T, self.r, self.sigma))

    def value(self):
        """Compute option value according to BSM model."""
        return self._sign[1] * self.S0 \
                   * norm.cdf(self._sign[1] * self.d1, 0.0, 1.0) \
                   - self._sign[1] * self.K * np.exp(-self.r * self.T) \
                   * norm.cdf(self._sign[1] * self.d2, 0.0, 1.0)

    def delta(self):
        return norm.cdf(self.d1, 0.0, 1.0) + self._sign[0]

    def gamma(self):
        return (norm.pdf(self.d1, 0.0, 1.0)
                   / (self.S0 * self.sigma * np.sqrt(self.T)))

    def vega(self):
        return self.S0 * norm.pdf(self.d1, 0.0, 1.0) * np.sqrt(self.T)

    def theta(self):
        return -1. * (self.S0 * norm.pdf(self.d1, 0.0, 1.0) * self.sigma) \
                   / (2. * np.sqrt(self.T)) \
                   - 1. * self._sign[1] * self.r * self.K \
                   * np.exp(-self.r * self.T) * norm.cdf(self._sign[1] \
                   * self.d2, 0.0, 1.0)

    def rho(self):
        return (self._sign[1] * self.K * self.T * np.exp(-self.r * self.T)
                   * norm.cdf(self._sign[1] * self.d2, 0.0, 1.0))

    def omega(self):
        return ((norm.cdf(self.d1, 0.0, 1.0) + self._sign[0])
                   * self.S0 / self.value())

    def implied_vol(self, value, precision=1.0e-5, iters=100):
        """Get implied vol at the specified price using an iterative approach.

        There is no closed-form inverse of BSM-value as a function of sigma,
        so start at an anchoring volatility level from Brenner & Subrahmanyam
        (1988) and work iteratively from there.

        Resources
        =========
        Brenner & Subrahmanyan, A Simple Formula to Compute the Implied
            Standard Deviation, 1988.
        """

        vol = np.sqrt(2. * np.pi / self.T) * (value / self.S0)
        for _ in itertools.repeat(None, iters):  # Faster than range
            opt = BSM(S0=self.S0, K=self.K, T=self.T, r=self.r, sigma=vol,
                      kind=self.kind)
            diff = value - opt.value()
            if abs(diff) < precision:
                return vol
            vol = vol + diff / opt.vega()

        return vol

    def summary(self, name=None):
        res = OrderedDict([('Value', self.value()),
                           ('d1', self.d1),
                           ('d2', self.d2),
                           ('Delta', self.delta()),
                           ('Gamma', self.gamma()),
                           ('Vega', self.vega()),
                           ('Theta', self.theta()),
                           ('Rho', self.rho()),
                           ('Omega', self.omega())
                          ])

        return res


# Put & call: building blocks for more complex strategies
# ----------------------------------------------------------------------------

SIGN = {'long' : 1., 'Long' : 1., 'l' : 1., 'L' : 1.,
         'short' : -1., 'Short' : -1., 's' : -1., 'S' : -1.}

class Option(object):
    def __init__(self, K=None, price=None, St=None, kind='call', pos='long'):
        self.K = K
        self.price = price
        self.St = St
        self.kind = kind
        self.pos = pos

    def __repr__(self):
        if self.St is None:
            return ('{0}(K={1:0.2f}, price={2:0.2f}, St=None)'
                    .format(self.kind.title(), self.K, self.price))
        else:
            return ('{0}(K={1:0.2f}, price={2:0.2f}, St={2:0.2f})'
                    .format(self.kind.title(), self.K, self.price, self.St))


class Call(Option):
    def __init__(self, K=None, price=None, St=None, pos='long'):
        Option.__init__(self, K=K, price=price, St=St, kind='call', pos=pos)

    def payoff(self, St=None):
        St = self.St if St is None else St
        return SIGN[self.pos] * np.maximum(0., St - self.K)

    def profit(self, St=None):
        St = self.St if St is None else St
        return self.payoff(St=St) - SIGN[self.pos] * self.price


class Put(Option):
    def __init__(self, K=None, price=None, St=None, pos='long'):
        Option.__init__(self, K=K, price=price, St=St, kind='put', pos=pos)

    def payoff(self, St=None):
        St = self.St if St is None else St
        return SIGN[self.pos] * np.maximum(0., self.K - St)

    def profit(self, St=None):
        St = self.St if St is None else St
        return self.payoff(St=St) - SIGN[self.pos] * self.price


# Options strategies: combinations of multiple options.  `OpStrat` is a
#     generic options class from which other (specifically-named) options
#     strategies inherit
# ----------------------------------------------------------------------------


class OpStrat(object):
    """Generic option strategy construction."""
    def __init__(self, St=None):
        self.St = St
        self.options = []

    def add_option(self, K=None, price=None, St=None, kind='call', pos='long'):
        """Add an option to the object's `options` container."""
        kinds = {'call' : Call, 'Call' : Call, 'c' : Call, 'C' : Call,
                 'put' : Put, 'Put' : Put, 'p' : Put, 'P' : Put}
        St = self.St if St is None else St
        option = kinds[kind](St=St, K=K, price=price, pos=pos)
        self.options.append(option)

    def summary(self, St=None):
        St = self.St if St is None else St
        if self.options:
            payoffs = [op.payoff(St=St) for op in self.options]
            profits = [op.profit(St=St) for op in self.options]
            strikes = [op.K for op in self.options]
            prices = [op.price for op in self.options]
            exprs = [St] * len(self.options)
            kinds = [op.kind for op in self.options]
            poss = [op.pos for op in self.options]
            res = OrderedDict([('kind', kinds),
                               ('position', poss),
                               ('strike', strikes),
                               ('price', prices),
                               ('St', exprs),
                               ('payoff', payoffs),
                               ('profit', profits),
                              ])

            return DataFrame(res)
        else:
            return None

    def payoff(self, St=None):
        return np.sum([op.payoff(St=St) for op in self.options], axis=0)

    def profit(self, St=None):
        return np.sum([op.profit(St=St) for op in self.options], axis=0)

    def grid(self, start=None, stop=None, St=None, **kwargs):
        """Returns tuple of St, payoffs, profits."""
        lb = 0.75
        rb = 1.25
        if not any((start, stop, St, self.St)):
            St = np.mean([op.K for op in self.options], axis=0)
            start = St * lb
            stop = St * rb
        elif not any((start, stop)):
            St = self.St if St is None else St
            start = np.max(St) * lb
            stop = np.max(St) * rb
        St = np.linspace(start, stop, **kwargs)
        payoffs = self.payoff(St=St)
        profits = self.profit(St=St)

        return St, payoffs, profits

    def diagram(self, start=None, stop=None, St=None, **kwargs):
        """Plot `grid`."""
        St, payoffs, profits = self.grid(start=start, stop=stop, St=St, 
                                         **kwargs)
        plt.plot(St, payoffs, St, profits)


class Straddle(OpStrat):
    """Long-volatility exposure.  Long a put and call, both at K."""
    def __init__(self, St=None, K=None, price=None):
        OpStrat.__init__(self, St=St)
        self.K = K
        self.price = price
        self.add_option(K=K, price=price, St=St, kind='call')
        self.add_option(K=K, price=price, St=St, kind='put')


class Strip(Straddle):
    """Combination of a straddle with a put.  Long 1 call & 2 puts at K."""
    def __init__(self, St=None, K=None, price=None):
        Straddle.__init__(self, St=St, K=K, price=price)
        self.add_option(K=K, price=price, St=St, kind='put')


class Strap(Straddle):
    """Combination of a straddle with a call.  Long 2 calls & 1 put at K."""
    def __init__(self, St=None, K=None, price=None):
        Straddle.__init__(self, St=St, K=K, price=price)
        self.add_option(K=K, price=price, St=St, kind='call')


class ShortStraddle(OpStrat):
    """Short-volatility exposure.  Short a put and call, both at K."""
    def __init__(self, St=None, K=None, price=None):
        OpStrat.__init__(self, St=St)
        self.K = K
        self.price = price
        self.add_option(K=K, price=price, St=St, kind='call', pos='short')
        self.add_option(K=K, price=price, St=St, kind='put', pos='short')


class Strangle(OpStrat):
    """Straddle in which the put and call have different strikes.

    - Long K1 (call)
    - Long K2 (put)
    """

    def __init__(self, St=None, K1=None, K2=None, callprice=None,
                 putprice=None):
        OpStrat.__init__(self, St=St)
        self.K1 = K1
        self.K2 = K2
        self.callprice = callprice
        self.putprice = putprice
        self.add_option(K=K1, price=callprice, St=St, kind='call')
        self.add_option(K=K2, price=putprice, St=St, kind='put')


class ShortStrangle(OpStrat):
    """Short straddle in which the put and call have different strikes.

    - Short K1 (call)
    - Long K2 (put)
    """

    def __init__(self, St=None, K1=None, K2=None, callprice=None,
                 putprice=None):
        OpStrat.__init__(self, St=St)
        self.K1 = K1
        self.K2 = K2
        self.callprice = callprice
        self.putprice = putprice
        self.add_option(K=K1, price=callprice, St=St, kind='call', pos='short')
        self.add_option(K=K2, price=putprice, St=St, kind='put', pos='short')


class BullSpread(OpStrat):
    """Bullish strategy but with limited gain and limited loss.

    Combination of 2 puts or 2 calls.

    - Long K1 (put or call)
    - Short K2 (put or call)
    """

    def __init__(self, St=None, K1=None, K2=None, price1=None, price2=None,
                 kind='call'):
        OpStrat.__init__(self, St=St)
        self.K1 = K1
        self.K2 = K2
        self.price1 = price1
        self.price2 = price2
        self.add_option(K=K1, price=price1, St=St, kind=kind, pos='long')
        self.add_option(K=K2, price=price2, St=St, kind=kind, pos='short')


class BearSpread(OpStrat):
    """Bearish strategy but with limited loss and limited gain.

    Combination of 2 puts or 2 calls.

    - Short K1 (put or call)
    - Long K2 (put or call)
    """

    def __init__(self, St=None, K1=None, K2=None, price1=None, price2=None,
                 kind='put'):
        OpStrat.__init__(self, St=St)
        self.K2 = K1
        self.K2 = K2
        self.price1 = price1
        self.price2 = price2
        self.add_option(K=K1, price=price1, St=St, kind=kind, pos='short')
        self.add_option(K=K2, price=price2, St=St, kind=kind, pos='long')


class LongPutLadder(BearSpread):
    """Bear put spread combined with selling another lower-strike put."""
    def __init__(self, St=None, K1=None, K2=None, K3=None, price1=None,
                 price2=None, price3=None):
        BearSpread.__init__(self, St=St, K1=K2, K2=K3, price1=price2, 
                            price2=price3)
        self.K1 = K1
        self.price1 = price1
        self.add_option(K=K1, price=price1, St=St, kind='put', pos='short')


class ShortPutLadder(BearSpread):
    """Bull put spread combined with buying another lower-strike put."""
    def __init__(self, St=None, K1=None, K2=None, K3=None, price1=None,
                 price2=None, price3=None):
        BullSpread.__init__(self, St=St, K1=K2, K2=K3, price1=price2, 
                            price2=price3,
                            kind='put')
        self.K1 = K1
        self.price1 = price1
        self.add_option(K=K1, price=price1, St=St, kind='put', pos='long')


class _Butterfly(OpStrat):
    def __init__(self, St=None, K1=None, K2=None, K3=None, price1=None,
                 price2=None, price3=None):
        if not np.allclose(np.mean([K1, K3]), K2):
            warnings.warn('specified strikes are not equidistant.')
        OpStrat.__init__(self, St=St)
        self.K1 = K1
        self.K2 = K2
        self.K3 = K3
        self.price1 = price1
        self.price2 = price2
        self.price3 = price3


class LongButterfly(_Butterfly):
    """Combination of 4 calls or 4 puts.  Short volatility exposure.

    - Long K1 (put or call)
    - Short 2x K2 (put or call)
    - Long K3 (put or call)
    """

    def __init__(self, St=None, K1=None, K2=None, K3=None, price1=None,
                 price2=None, price3=None, kind='call'):
        _Butterfly.__init__(self, St=St, K1=K1, K2=K2, K3=K3, price1=price1,
                           price2=price2, price3=price3)
        self.kind = kind
        self.add_option(K=K1, price=price1, St=St, kind=kind, pos='long')
        self.add_option(K=K2, price=price2, St=St, kind=kind, pos='short')
        self.add_option(K=K2, price=price2, St=St, kind=kind, pos='short')
        self.add_option(K=K3, price=price3, St=St, kind=kind, pos='long')


class ShortButterfly(_Butterfly):
    """Combination of 4 calls or 4 puts.  Long volatility exposure.

    - Short K1 (put or call)
    - Long 2x K2 (put or call)
    - Short K3 (put or call)
    """

    def __init__(self, St=None, K1=None, K2=None, K3=None, price1=None,
                 price2=None, price3=None, kind='call'):
        _Butterfly.__init__(self, St=St, K1=K1, K2=K2, K3=K3, price1=price1,
                           price2=price2, price3=price3)
        self.kind = kind
        self.add_option(K=K1, price=price1, St=St, kind=kind, pos='short')
        self.add_option(K=K2, price=price2, St=St, kind=kind, pos='long')
        self.add_option(K=K2, price=price2, St=St, kind=kind, pos='long')
        self.add_option(K=K3, price=price3, St=St, kind=kind, pos='short')


class LongIronButterfly(_Butterfly):
    """Combination of 2 puts and 2 calls.  Long volatility exposure.

    - Short K1 (put)
    - Long 2x K2 (1 put, 1 call)
    - Short K3 (call)
    """

    def __init__(self, St=None, K1=None, K2=None, K3=None, price1=None,
                 price2=None, price3=None):
        _Butterfly.__init__(self, St=St, K1=K1, K2=K2, K3=K3, price1=price1,
                           price2=price2, price3=price3)
        self.add_option(K=K1, price=price1, St=St, kind='put', pos='short')
        self.add_option(K=K2, price=price2, St=St, kind='put', pos='long')
        self.add_option(K=K2, price=price2, St=St, kind='call', pos='long')
        self.add_option(K=K3, price=price3, St=St, kind='call', pos='short')


class ShortIronButterfly(_Butterfly):
    """Combination of 2 puts and 2 calls.  Long volatility exposure.

    - Long K1 (put)
    - Short 2x K2 (1 put, 1 call)
    - Long K3 (call)
    """

    def __init__(self, St=None, K1=None, K2=None, K3=None, price1=None,
                 price2=None, price3=None):
        _Butterfly.__init__(self, St=St, K1=K1, K2=K2, K3=K3, price1=price1,
                           price2=price2, price3=price3)
        self.add_option(K=K1, price=price1, St=St, kind='put', pos='long')
        self.add_option(K=K2, price=price2, St=St, kind='put', pos='short')
        self.add_option(K=K2, price=price2, St=St, kind='call', pos='short')
        self.add_option(K=K3, price=price3, St=St, kind='call', pos='long')


class _Condor(OpStrat):
    def __init__(self, St=None, K1=None, K2=None, K3=None, K4=None,
                 price1=None, price2=None, price3=None, price4=None):
        if not np.allclose(K2 - K1, K4 - K3):
            warnings.warn('specified wings are not equidistant.')
        OpStrat.__init__(self, St=St)
        self.K1 = K1
        self.K2 = K2
        self.K3 = K3
        self.K4 = K4
        self.price1 = price1
        self.price2 = price2
        self.price3 = price3
        self.price4 = price4


class LongCondor(_Condor):
    """Combination of 4 calls or 4 puts.  Short volatility exposure.

    - Long K1 (put or call)
    - Short K2 != K3 (put or call)
    - Long K4 (put or call)
    """

    def __init__(self, St=None, K1=None, K2=None, K3=None, K4=None,
                 price1=None, price2=None, price3=None, price4=None,
                 kind='call'):
        _Condor.__init__(self, St=St, K1=K1, K2=K2, K3=K3, K4=K4,
                         price1=price1, price2=price2, price3=price3,
                         price4=price4)
        self.kind = kind
        self.add_option(K=K1, price=price1, St=St, kind=kind, pos='long')
        self.add_option(K=K2, price=price2, St=St, kind=kind, pos='short')
        self.add_option(K=K3, price=price3, St=St, kind=kind, pos='short')
        self.add_option(K=K4, price=price4, St=St, kind=kind, pos='long')


class ShortCondor(_Condor):
    """Combination of 4 calls or 4 puts.  Long volatility exposure.

    - Short K1 (put or call)
    - Long K2 != K3 (put or call)
    - Short K4 (put or call)
    """

    def __init__(self, St=None, K1=None, K2=None, K3=None, K4=None,
                 price1=None, price2=None, price3=None, price4=None,
                 kind='call'):
        _Condor.__init__(self, St=St, K1=K1, K2=K2, K3=K3, K4=K4,
                         price1=price1, price2=price2, price3=price3,
                         price4=price4)
        self.kind = kind
        self.add_option(K=K1, price=price1, St=St, kind=kind, pos='short')
        self.add_option(K=K2, price=price2, St=St, kind=kind, pos='long')
        self.add_option(K=K3, price=price3, St=St, kind=kind, pos='long')
        self.add_option(K=K4, price=price4, St=St, kind=kind, pos='short')


class LongIronCondor(_Condor):
    """Combination of 2 puts and 2 calls.  Long volatility exposure.

    - Short K1 (put)
    - Long K2 (put)
    - Long K3 (call)
    - Short K4 (call)
    """

    def __init__(self, St=None, K1=None, K2=None, K3=None, K4=None,
                 price1=None, price2=None, price3=None, price4=None):
        _Condor.__init__(self, St=St, K1=K1, K2=K2, K3=K3, K4=K4,
                         price1=price1, price2=price2, price3=price3,
                         price4=price4)
        self.add_option(K=K1, price=price1, St=St, kind='put', pos='short')
        self.add_option(K=K2, price=price2, St=St, kind='put', pos='long')
        self.add_option(K=K3, price=price3, St=St, kind='call', pos='long')
        self.add_option(K=K4, price=price4, St=St, kind='call', pos='short')


class ShortIronCondor(_Condor):
    """Combination of 2 puts and 2 calls.  Short volatility exposure.

    - Long K1 (put)
    - Short K2 (put)
    - Short K3 (call)
    - Long K4 (call)
    """

    def __init__(self, St=None, K1=None, K2=None, K3=None, K4=None,
                 price1=None, price2=None, price3=None, price4=None):
        _Condor.__init__(self, St=St, K1=K1, K2=K2, K3=K3, K4=K4,
                         price1=price1, price2=price2, price3=price3,
                         price4=price4)
        self.add_option(K=K1, price=price1, St=St, kind='put', pos='long')
        self.add_option(K=K2, price=price2, St=St, kind='put', pos='short')
        self.add_option(K=K3, price=price3, St=St, kind='call', pos='short')
        self.add_option(K=K4, price=price4, St=St, kind='call', pos='long')