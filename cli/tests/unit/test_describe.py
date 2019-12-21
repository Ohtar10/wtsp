import pytest

from tests import common, tests_path
from wtsp.describe.describe import Describer
from wtsp.exceptions import InvalidArgumentException, DescribeException, DataLoadException


@pytest.mark.parametrize(
    "filters",
    [
        "lalala"
    ]
)
def test_describe_invalid_filters_should_fail(filters):
    with pytest.raises(InvalidArgumentException) as e:
        Describer("", "", "", "", filters, 10)
    assert "Filter value is invalid. use: key=value" in str(e.value)


def test_describe_with_invalid_filter_value():
    filters = "lalala=US"
    describer = Describer("", "place_name", "tweet", "tweets", filters)
    input_data = common.get_full_path(tests_path, common.RAW_TWEETS_PATH)
    with pytest.raises(DescribeException) as e:
        describer.describe(input_data)
    assert "There is a problem processing the data, see the error message" in str(e.value)


@pytest.mark.parametrize(
    "path",
    [
        "lalala",
        ""
    ]
)
def test_describe_invalid_input_path_should_fail(path):
    describer = Describer("", "", "", "", "key=value")
    with pytest.raises(InvalidArgumentException) as e:
        describer.describe(path)
    assert "The provided input data path is not valid" in str(e.value)


def test_describe_invalid_data_should_fail(tmpdir):
    describer = Describer("", "", "", "", "key=value")
    p = tmpdir.mkdir("sub").join("hello.txt")
    p.write("content")
    with pytest.raises(DataLoadException) as e:
        describer.describe(p)
    assert "The provided input data is not a valid parquet file" in str(e.value)
