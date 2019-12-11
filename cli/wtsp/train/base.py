import os.path
from wtsp.core.base import Process
from wtsp.exceptions import InvalidArgumentException
from wtsp.train.tweets import GeoTweetsNearestNeighbors


class Trainer(Process):
    """This class orchestrates the model training modules
    invoking each according to how this class was initialized"""

    def __init__(self, work_dir, debug, model):
        super().__init__(work_dir, debug)
        self.model = model

    def run(self, input_file, **kwargs):
        self.__validate_inputs(input_file, kwargs)
        if self.model == 'nearest-neighbors':
            trainer = GeoTweetsNearestNeighbors(self.work_dir, self.debug, **kwargs)
            return trainer.run(input_file)

    def __validate_inputs(self, input_file, kwargs):
        if not input_file:
            raise InvalidArgumentException("input file not provided")
        if not os.path.exists(input_file):
            raise InvalidArgumentException("input file does not exist")
        if not kwargs:
            raise InvalidArgumentException("trainer arguments not provided")
