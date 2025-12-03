import pytest

from backend.symbols import normalize_symbol


def test_normalize_nse_default():
    assert normalize_symbol("TCS") == "TCS.NS"


def test_normalize_nse_explicit():
    assert normalize_symbol("TCS.NS") == "TCS.NS"


def test_normalize_bse_numeric():
    assert normalize_symbol("BSE:500325") == "500325.BSE"


def test_normalize_bse_explicit():
    assert normalize_symbol("RELIANCE.BSE") == "RELIANCE.BSE"