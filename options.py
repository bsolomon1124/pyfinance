"""Vectorized options calculations.

Descriptions
============
# TODO
"""

__author__ = 'Brad Solomon <brad.solomon.1124@gmail.com>'

__all__ = [
    'BSM', 'Call', 'Put', 'OpStrat', 'Straddle', 'BullSpread',
    'BearSpread', 'LongButterfly', 'ShortButterfly'
    ]

from collections import OrderedDict

import numpy as np
from pandas import DataFrame
from scipy.stats import norm

# TODO: covered call, protective/married put, collar,
#       long butterfly, short butterfly, long iron butterfly,
#       short iron butterfly, long condor, short condor, long iron condor,
#       short iron condor

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

    Resources
    =========
    Thomas Ho Company Ltd: Financial Models
      http://www.thomasho.com/mainpages/?analysoln

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
        self.d2 = ((np.log(self.S0 / self.K)
                + (self.r - 0.5 * self.sigma ** 2) * self.T)
                / (self.sigma * np.sqrt(self.T)))

        # Several greeks use negated terms dependent on option type
        # For example, delta of call is N(d1) and delta put is N(d1) - 1
        # and theta may use N(d2) or N(-d2).  In these lists:
        # - element 0 (0, -1) is used in delta and omega
        # - element 1 (1, -1) is used in rho and theta
        # - the negative of element 1 (-1, 1) is used in theta

        self._sign = {'call' : [0, 1], 'put' : [-1, -1]}
        self._sign = self._sign[self.kind]

    def __repr__(self):
        return ('BSM(kind={0}, S0={1:0.2f}, K={2:0.2f}, '
                'T={3:0.2f}, r={4:0.2f}, sigma={5:0.2f})'
                .format(self.kind, self.S0, self.K,
                 self.T, self.r, self.sigma))

    def value(self):
        """Compute option value."""
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

    def implied_vol(self, value, iter=100):
        """Get implied vol at the specified price.  Iterative approach."""
        vol = 0.4
        precision = 1.0e-5
        # TODO: itertools.repeat
        for _ in range(iter):
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

# Put & call: the building blocks of all other options strategies
# ----------------------------------------------------------------------------

_sign = {'long' : 1., 'Long' : 1., 'l' : 1., 'L' : 1.,
         'short' : -1., 'Short' : -1., 's' : -1., 'S' : -1.}

class Call(object):

    def __init__(self, K=None, price=None, St=None, pos='long'):
        self.kind = 'call'
        self.K = K
        self.price = price
        self.St = St
        self.pos = pos

    def payoff(self, St=None):
        St = self.St if St is None else St
        return _sign[self.pos] * np.maximum(0., St - self.K)

    def profit(self, St=None):
        St = self.St if St is None else St
        return self.payoff(St=St) - _sign[self.pos] * self.price

    def __repr__(self):
        return ('Call(K={0:0.2f}, price={1:0.2f}, St={2:0.2f})'
                .format(self.K, self.price, self.St))

class Put(object):

    def __init__(self, K=None, price=None, St=None, pos='long'):
        self.kind = 'put'
        self.K = K
        self.price = price
        self.St = St
        self.pos = pos

    def payoff(self, St=None):
        St = self.St if St is None else St
        return _sign[self.pos] * np.maximum(0., self.K - St)

    def profit(self, St=None):
        St = self.St if St is None else St
        return self.payoff(St=St) - _sign[self.pos] * self.price

    def __repr__(self):
        return ('Put(K={0:0.2f}, price={1:0.2f}, St={2:0.2f})'
                .format(self.K, self.price, self.St))

# Options strategies: combinations of multiple options.  `OpStrat` is a
# generic options class from which other (specifically-named) options
# strategies inherit
# ----------------------------------------------------------------------------

class OpStrat(object):
    """Generic option strategy construction."""
    def __init__(self, St=None):
        self.St = St
        self.options = []

    def add_option(self, K=None, price=None, St=None, kind='call', pos='long'):
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

    def diagram(self, start=None, stop=None, St=None, **kwargs):
        # Case 1: start=None, stop=None, St=None, self.St=None -> ValueError
        # Case 2: start=None, stop=None, St=None, self.St not None ->
        #         St = self.St, 0.9/1.1 bound start/stop
        # Case 3: start not None & stop not None -> use as-is

        if not any((start, stop, St, self.St)):
            raise ValueError('must specify one of (`start`, `stop`, `St`).'
                             ' `St` is midpoint of x-axis')
        elif not any((start, stop)):
            St = self.St if St is None else St
            start = np.max(St) * 0.9
            stop = np.max(St) * 1.1

        St = np.linspace(start, stop, **kwargs)
        payoffs = self.payoff(St=St)
        profits = self.profit(St=St)
        plt.plot(St, payoffs, St, profits)
        # TODO: title, legend, etc
        # or maybe just return tuple of arrays?

class Straddle(OpStrat):
    def __init__(self, St=None, K=None, callprice=None, putprice=None):
        OpStrat.__init__(self, St=St)

        self.K = K
        self.callprice = callprice
        self.putprice = putprice

        self.add_option(K=K, price=callprice, St=St, kind='call')
        self.add_option(K=K, price=putprice, St=St, kind='put')

class Strangle(OpStrat):
    """A lower cost alternative to a straddle.

    The investor purchases a long call and a long put on the same underlying
    security with the same expiration but where the put strike price is lower
    than the call strike price.
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

class BullSpread(OpStrat):
    def __init__(self, St=None, K1=None, K2=None, price1=None, price2=None,
                 kind='call'):
        OpStrat.__init__(self, St=St)

        self.K = K
        self.price1 = price1
        self.price2 = price2

        self.add_option(K=K1, price=price1, St=St, kind=kind, pos='long')
        self.add_option(K=K2, price=price2, St=St, kind=kind, pos='short')

class BearSpread(OpStrat):
    """
    Example
    =======
    b = BearSpread(St=np.array([1900, 2000, 2100]), K1=1950, K2=2050,
                   price1=56.01, price2=107.39)
    print(b.profit())
    [ 48.62  -1.38 -51.38]
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

class LongButterfly(OpStrat):
    """A (long) butterfly spread.

    Consists of all calls or puts and is established by purchasing the low
    strike price, selling 2 at a middle strike price, and then buying the
    highest strike price. The low and high strikes are equidistant from the
    middle strike. Used by people who feel the underlying will trade in a
    narrow range.
    """

    # TODO: equidistant strike check/logic

    def __init__(self, St=None, K1=None, K2=None, K3=None, price1=None,
                 price2=None, price3=None, kind='call'):
        OpStrat.__init__(self, St=St)

        self.K1 = K1
        self.K2 = K2
        self.K3 = K3
        self.price1 = price1
        self.price2 = price2
        self.price3 = price3

        self.add_option(K=K1, price=price1, St=St, kind=kind, pos='long')
        self.add_option(K=K2, price=price2, St=St, kind=kind, pos='short')
        self.add_option(K=K2, price=price2, St=St, kind=kind, pos='short')
        self.add_option(K=K3, price=price3, St=St, kind=kind, pos='long')

class ShortButterfly(OpStrat):

    def __init__(self, St=None, K1=None, K2=None, K3=None, price1=None,
                 price2=None, price3=None, kind='call'):
        OpStrat.__init__(self, St=St)

        self.K1 = K1
        self.K2 = K2
        self.K3 = K3
        self.price1 = price1
        self.price2 = price2
        self.price3 = price3

        self.add_option(K=K1, price=price1, St=St, kind=kind, pos='short')
        self.add_option(K=K2, price=price2, St=St, kind=kind, pos='long')
        self.add_option(K=K2, price=price2, St=St, kind=kind, pos='long')
        self.add_option(K=K3, price=price3, St=St, kind=kind, pos='short')

