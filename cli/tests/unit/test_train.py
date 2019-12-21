import pytest

from wtsp.exceptions import InvalidArgumentException, DataLoadException
from wtsp.train.products import ProductsTrainer
from wtsp.train.tweets import TweetsTrainer
from wtsp.transform.transformers import WhereToSellProductsTransformer


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


@pytest.mark.parametrize(
    "work_dir",
    [
        "",
        None
    ]
)
def test_train_product_invalid_working_directory_should_fail(work_dir):
    with pytest.raises(InvalidArgumentException) as e:
        ProductsTrainer(work_dir, "", "key=value")
    assert "The working directory is required." in str(e.value)


@pytest.mark.parametrize(
    "model",
    [
        "model",
        "",
        "pepito",
        None
    ]
)
def test_train_product_invalid_model_should_fail(model):
    trainer = ProductsTrainer("lalala", model, "key=value")
    with pytest.raises(InvalidArgumentException) as e:
        trainer.train("")
    assert f"There is no '{model}' model to train" in str(e.value)


@pytest.mark.parametrize(
    "params",
    [
        "lalala",
        "",
        None
    ]
)
def test_train_product_invalid_params_should_fails(params):
    with pytest.raises(InvalidArgumentException) as e:
        ProductsTrainer("", "embeddings", params)
    assert "Params value is invalid. use: key=value" in str(e.value)


@pytest.mark.parametrize(
    ("model", "path"),
    [
        ("embeddings", "lalala"),
        ("classifier", "lalala"),
        ("embeddings", ""),
        ("classifier", ""),
        ("embeddings", None),
        ("classifier", None)
    ]
)
def test_train_product_invalid_data_path_should_fail(model, path):
    trainer = ProductsTrainer("lalala", model, "classes=10")
    with pytest.raises(InvalidArgumentException) as e:
        trainer.train(path)
    assert "The provided input data path is not valid" in str(e.value)


@pytest.mark.parametrize(
    "model",
    [
        "embeddings",
        "classifier"
    ]
)
def test_train_product_invalid_data_should_fail(tmpdir, model):
    trainer = ProductsTrainer("lalala", model, "classes=10")
    p = tmpdir.mkdir("sub").join("hello.txt")
    p.write("content")
    with pytest.raises(DataLoadException) as e:
        trainer.train(p)
    assert "The provided input data is not a valid parquet file" in str(e.value)


@pytest.mark.parametrize(
    "filters",
    [
        "lalala",
        "",
        None
    ]
)
def test_predict_wtsp_invalid_filters_should_fail(filters):
    with pytest.raises(InvalidArgumentException) as e:
        WhereToSellProductsTransformer("", filters, "")
    assert "Filter value is invalid. use: key=value" in str(e.value)


@pytest.mark.parametrize(
    "params",
    [
        "lalala",
        "",
        None
    ]
)
def test_predict_wtsp_invalid_params_should_fail(params):
    with pytest.raises(InvalidArgumentException) as e:
        WhereToSellProductsTransformer("", "key=value", params)
    assert "Params value is invalid. use: key=value" in str(e.value)


@pytest.mark.parametrize(
    "path",
    [
        "lalala",
        "",
        None
    ]
)
def test_predict_wtsp_invalid_data_path_should_fail(path):
    transformer = WhereToSellProductsTransformer("lalala", "key=value", "classes=10")
    with pytest.raises(InvalidArgumentException) as e:
        transformer.transform(path)
    assert "The provided input data path is not valid" in str(e.value)


def test_predict_wtsp_invalid_data_should_fail(tmpdir):
    transformer = WhereToSellProductsTransformer("lalala", "key=value", "classes=10")
    p = tmpdir.mkdir("sub").join("hello.txt")
    p.write("content")
    with pytest.raises(DataLoadException) as e:
        transformer.transform(p)
    assert "The provided input data is not a valid parquet file" in str(e.value)