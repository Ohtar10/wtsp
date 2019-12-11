from typing import Dict
import re


def parse_kwargs(string:str) -> Dict[str, object]:
    arg_list = string.split(",")
    kwargs = {key: infer_and_cast_to_type(extract_string(value))
              for key, value in [arg.split("=") for arg in arg_list]}
    return kwargs


def extract_string(string: str) -> str:
    matcher = re.search(r'^[\'"]?([\w\d\s\\.]+)[\'"]?$', string)
    return matcher.group(1)


def infer_and_cast_to_type(string: str) -> object:
    try:
        return float(string)
    except (ValueError, TypeError):
        # Ignore and continue with the next type
        pass

    try:
        return int(string)
    except (ValueError, TypeError):
        return string
