import pytest
from faker import Faker


@pytest.fixture(scope="session")
def fake():
    return Faker()
