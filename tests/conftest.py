import pytest
import pandas as pd


@pytest.fixture
def fake_df():
    """Provide a deterministic fake price series for tests."""
    data = {"Close": [100 + i for i in range(300)]}
    return pd.DataFrame(data)