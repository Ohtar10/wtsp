import pytest

from wtsp.exceptions import InvalidArgumentException, DataLoadException
from wtsp.train.tweets import TweetsTrainer


@pytest.mark.parametrize(
    "filters",
    [
        "lalala",
        "",
        None
    ]
)
def test_train_tweets_invalid_filters_should_fail(filters):
    with pytest.raises(InvalidArgumentException) as e:
        TweetsTrainer("", filters, "", "")
        assert "Filter value is invalid. use: key=value" in str(e.value)


@pytest.mark.parametrize(
    "params",
    [
        "lalala",
        "",
        None
    ]
)
def test_train_tweets_invalid_params_should_fail(params):
    with pytest.raises(InvalidArgumentException) as e:
        TweetsTrainer("", "key=value", params, "")
        assert "Params value is invalid. use: key=value" in str(e.value)


@pytest.mark.parametrize(
    "path",
    [
        "lalala",
        ""
    ]
)
def test_train_tweets_invalid_input_path_should_fail(path):
    filters = "key=value"
    params = "key=value"
    trainer = TweetsTrainer("nearest-neighbors", filters, params, "")
    with pytest.raises(InvalidArgumentException) as e:
        trainer.train(path)
        assert "The provided input data path is not valid" in str(e.value)


def test_train_tweets_invalid_data_should_fail(tmpdir):
    filters = "key=value"
    params = "key=value"
    trainer = TweetsTrainer("nearest-neighbors", filters, params, "")
    p = tmpdir.mkdir("sub").join("hello.txt")
    p.write("content")
    with pytest.raises(DataLoadException) as e:
        trainer.train(p)
        assert "The provided input data is not a valid parquet file" in str(e.value)
