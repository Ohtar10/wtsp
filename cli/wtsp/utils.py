from typing import Dict


def parse_kwargs(string:str) -> Dict[str, object]:
    arg_list = string.split(",")
    kwargs = {key: cast_int_if_possible(value) for key, value in [arg.split("=") for arg in arg_list]}
    return kwargs


def cast_int_if_possible(string: str) -> object:
    try:
        return int(string)
    except (ValueError, TypeError):
        return string
