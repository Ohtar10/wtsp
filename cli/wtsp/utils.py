"""General and transversal utilities module."""

import re
import nltk
from typing import Dict
from shapely import wkt


def parse_kwargs(string: str) -> Dict[str, object]:
    """Parse kwargs strings with format.

    key1=value1,key2=1,key3=0.5,key4="hello world",key5='hello world'

    and converts it into a dictionary with each key and parsed value.

    :param string: kwargs strings
    :returns: dictionary with parameters parsing
    """
    arg_list = string.split(",")
    kwargs = {key: infer_and_cast_to_type(extract_string(value))
              for key, value in [arg.split("=") for arg in arg_list]}
    return kwargs


def extract_string(string: str) -> str:
    """Extract text from a string that is enclosed with quotes.

    e.g., converts "my string" into->  my string

    :param string: string to extract text
    :returns: string without quotes
    """
    matcher = re.search(r'^[\'"]?([-;\w\d\s\\.]+)[\'"]?$', string)
    return matcher.group(1)


def infer_and_cast_to_type(string: str) -> object:
    """Try to parse the given string into numerical types.

    float->int->str

    :param string: string value to be parsed
    :returns: float or int if it was possible to convert, str otherwise
    """
    try:
        return int(string)
    except (ValueError, TypeError):
        # Ignore and continue with the next type
        pass

    try:
        return float(string)
    except (ValueError, TypeError):
        return string


def parse_geometry(geom):
    """Parse geometry.

    Parses a WKT geometry and return the
    corresponding shapely object.
    """
    if geom:
        return wkt.loads(geom)
    else:
        return None


def ensure_nltk_resource_is_available(name: str):
    """Ensure nltk resource is available."""
    try:
        nltk.data.find(f"tokenizers/{name}")
    except LookupError:
        nltk.download(name)
