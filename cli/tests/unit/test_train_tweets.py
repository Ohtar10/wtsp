"""Tests related to tweet model training."""

import os.path
import pytest

from wtsp.exceptions import InvalidArgumentException
from wtsp.train.base import Trainer
from tests import common
from tests import tests_path
from wtsp.utils import parse_kwargs


def test_train_tweets_nearest_neighbors_ok():
    input_data = common.get_full_path(tests_path, common.RAW_TWEETS_PATH)
    trainer = Trainer(common.TEST_WORK_DIR_PATH, debug=True, model="nearest-neighbors")
    args = parse_kwargs("n_neighbors=2")
    result = trainer.run(input_data, **args)

    assert result == "Geo tweets nearest neighbors executed successfully. Use the report command to see the results."
    # check the output has data
    output_path = common.get_tweets_nn_dir()
    assert os.path.exists(output_path)
    # TODO check the output is the expected
    assert False


@pytest.mark.parametrize(
    "inputs,filters",
    [
        (None, None),
        ("", ""),
        ("blah", "blah"),
        (common.RAW_TWEETS_PATH, None),
        (common.RAW_TWEETS_PATH, ""),
        (common.RAW_TWEETS_PATH, "blah")
    ]
)
def test_train_tweets_nearest_neighbors_fail_on_invalid_params(inputs, filters):
    trainer = Trainer(common.TEST_WORK_DIR_PATH, debug=True, model="nearest-neighbor")
    with pytest.raises(InvalidArgumentException):
        trainer.run(inputs)
