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


class DataException(WTSPBaseException):
    """Raised when there is a data related error."""

    def _build_message(self) -> str:
        return f"There is a data related error: {self.offending_data}. Please review."


class DataLoadException(DataException):
    """Raised when there is a problem loading data."""

    def _build_message(self) -> str:
        return f"There is a problem loading the data: {self.offending_data}. Please review."
