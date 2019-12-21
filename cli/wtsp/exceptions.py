"""WTSP custom exceptions."""


class WTSPBaseException(Exception):
    """Base class for WTSP Exceptions."""

    def __init__(self, *args: object) -> None:
        """Create a WTSPBaseException object."""
        self.offending_data = args[0]
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        return f"An error occurred with: {self.offending_data}"


class InvalidArgumentException(WTSPBaseException):
    """Raised when invalid arguments when invoking functionalities."""
    
    def _build_message(self) -> str:
        return f"An invalid argument was provided: {self.offending_data}. Please review."


class DataLoadException(WTSPBaseException):
    """Raised when there is a problem loading data."""

    def _build_message(self) -> str:
        return f"A problem occurred trying to load the data: {self.offending_data}. Please review."


class DescribeException(WTSPBaseException):
    """Raised when there is a problem describing the data."""

    def _build_message(self) -> str:
        return f"There is a problem describing the data: {self.offending_data}. Please review."


class ModelTrainingException(WTSPBaseException):
    """Raised when there is a problem training a model."""

    def _build_message(self) -> str:
        return f"There is a problem training a model: {self.offending_data}. Please review."
