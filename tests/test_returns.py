import pytest

import numpy as np
import pandas as pd

from pyfinance.returns import TSeries, TFrame, FrequencyError  # noqa

np.random.seed(123)


def test_freq_from_broken_idx():
    # This should raise - can't infer here.
    idx = [pd.datetime(2000, 1, 1),
           pd.datetime(2000, 1, 2),
           pd.datetime(2000, 1, 4)]
    with pytest.raises(FrequencyError):
        r = TSeries([.01, .02, .03], index=idx)

    # This should not.
    idx = [pd.datetime(2000, 1, 1),
           pd.datetime(2000, 1, 2),
           pd.datetime(2000, 1, 3)]
    r = TSeries([.01, .02, .03], index=idx)
    assert r.freq == 'D'


def test_construction_methods():
    s = pd.Series([.0, .1, .2],
                  index=pd.date_range('2017', periods=3, freq='Q'))
    r = TSeries(s)
    assert r.freq == 'Q-Dec'
    assert r.index.freqstr == 'Q-DEC'

    # TODO: others


def test_retain_type_and_attrs():
    """Make sure that method calls retain type and attributes.

    Methods should return an instance of TSeries with the same
    `freq` (*usually*) as self.
    """

    s = np.random.randn(20)
    r = TSeries(s, index=pd.date_range('2017', periods=20, freq='M'))
    freq = r.freq
    methods = ('ret_rels', 'ret_idx', 'cuml_ret_idx')
    for m in methods:
        res = getattr(r, m)()
        assert isinstance(res, TSeries), res
        assert res.freq == freq, res
