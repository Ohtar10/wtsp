"""Base module.

Contains the base classes and functionalities for the
rest of the project.
"""
from pathlib import Path

DEFAULT_WORK_DIR = f"{str(Path.home())}/wtsp"
DEFAULT_DATA_TWEETS_DIR = "data/tweets"
DEFAULT_DATA_PRODUCTS_DIR = "data/products"


class Process:
    """Process base class containing general attributes and functionalities."""

    def __init__(self, work_dir, debug=False):
        """Create the process instance."""
        self.work_dir = work_dir
        self.debug = debug

    def run(self, *args):
        """Execute the process."""
        pass
