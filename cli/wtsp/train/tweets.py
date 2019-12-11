from wtsp.core.base import Process
from wtsp.exceptions import InvalidArgumentException


class GeoTweetsNearestNeighbors(Process):

    def __init__(self, work_dir, debug=False, **kwargs):
        super().__init__(work_dir, debug)
        params = ["n_neighbors",  "location"]
        if not all([arg in params for arg in kwargs]):
            raise InvalidArgumentException("For nearest neighbors both n_neighbors and location are required.")

        for k in kwargs.keys():
            if k in params:
                self.__setattr__(k, kwargs[k])

    def run(self, input_file):
        return "Geo tweets nearest neighbors executed successfully. Use the report command to see the results."


