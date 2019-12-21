"""Tests the utils functions."""

import pytest
from wtsp.utils import parse_kwargs


@pytest.mark.parametrize(
    "string, expected",
    [
        ("key=value", {"key": "value"}),
        ("key1=value1,key2=value2", {"key1": "value1", "key2": "value2"}),
        ("key=10", {"key": 10}),
        ("key='two words'", {"key": "two words"}),
        ('key="two words"', {"key": "two words"}),
        ("key=1.4", {"key": 1.4}),
        ("key='34.2;-118.2'", {"key": '34.2;-118.2'}),
        ("key=hello, key2=world", {"key": "hello", "key2": "world"})
    ]
)
def test_parse_kwargs(string, expected):
    assert parse_kwargs(string) == expected
