class WTSPBaseException(Exception):
    """Base class for WTSP Exceptions"""

    def __init__(self, *args: object) -> None:
        self.offending_data = args[0]
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        return f"An error occurred with: {self.offending_data}"


class InvalidArgumentException(WTSPBaseException):

    def _build_message(self) -> str:
        return f"An invalid argument was provided: {self.offending_data}. Please review."
