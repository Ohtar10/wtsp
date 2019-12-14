import pytest
from tests import common
from wtsp.exceptions import InvalidArgumentException
from wtsp.train.base import Trainer


@pytest.mark.parametrize(
    "inputs,n_neighbors,location",
    [
        (None, None, None),
        ("", "", ""),
        ("blah", "blah", "blah"),
        (common.RAW_TWEETS_PATH, None, None),
        (common.RAW_TWEETS_PATH, "", ""),
        (common.RAW_TWEETS_PATH, "blah", "blah")
    ]
)
def test_trainer_fails_on_invalid_input(inputs, n_neighbors, location):
    trainer = Trainer(common.TEST_WORK_DIR_PATH,
                      debug=True,
                      domain="tweets",
                      model="nearest-neighbor")
    with pytest.raises(InvalidArgumentException):
        trainer.run(inputs, n_neighbors=n_neighbors, location=location)
