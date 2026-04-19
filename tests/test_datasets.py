"""Smoke tests for `pyfinance.datasets`.

The loaders in this module call out to live third-party endpoints
(Ken French data library, FRED, SEC EDGAR, CBOE, census.gov, etc.),
which drift in both schema and availability. Running them in CI would
be flaky. These tests stay offline: they verify that the module
imports, that its public surface is intact, and that the lazy
`pandas_datareader` proxy behaves as designed without actually
triggering the real `pandas_datareader` import (which itself is
broken on Python 3.12+ because of stdlib `distutils` removal).
"""

from unittest.mock import patch

import pytest


def test_module_imports():
    from pyfinance import datasets

    assert set(datasets.__all__) == {
        "load_13f",
        "load_industries",
        "load_rates",
        "load_rf",
    }
    for name in datasets.__all__:
        assert hasattr(datasets, name), f"missing public name: {name}"


def test_load_13f_requires_user_agent():
    """Per SEC EDGAR fair-access policy, a User-Agent is mandatory."""
    from pyfinance import datasets

    with pytest.raises(ValueError, match="user_agent"):
        datasets.load_13f("https://example.test/form.xml", user_agent="")
    with pytest.raises(ValueError, match="user_agent"):
        datasets.load_13f("https://example.test/form.xml", user_agent=None)


def test_dstart_constant():
    from pyfinance import datasets

    assert datasets.DSTART == "1950-01"


def test_pdr_is_lazy_proxy():
    """`datasets.pdr` must be a proxy, not the real module."""
    from pyfinance import datasets

    assert isinstance(datasets.pdr, datasets._LazyPandasDataReader)


def test_pdr_defers_import_until_attribute_access():
    """Accessing an attribute on the proxy must import pandas_datareader,
    and only then."""
    from pyfinance import datasets

    sentinel = object()

    class _FakePDR:
        DataReader = sentinel

    with patch.dict("sys.modules", {"pandas_datareader": _FakePDR()}):
        assert datasets.pdr.DataReader is sentinel


def test_pdr_propagates_import_errors():
    """If pandas_datareader fails to import, the proxy surfaces the error
    at attribute-access time (not at module-import time)."""
    from pyfinance import datasets

    def _raise(name, *args, **kwargs):
        if name == "pandas_datareader":
            raise ModuleNotFoundError("No module named 'distutils'")
        return __import__(name, *args, **kwargs)

    with (
        patch("builtins.__import__", side_effect=_raise),
        pytest.raises(ModuleNotFoundError, match="distutils"),
    ):
        datasets.pdr.DataReader  # noqa: B018 — attribute access triggers the lazy import
