
class Process:
    """Process base class containing general attributes and functionalities."""

    def __init__(self, work_dir, debug=False):
        self.work_dir = work_dir
        self.debug = debug

    def run(self, *args):
        pass
