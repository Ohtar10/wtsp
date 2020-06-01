"""Core package.

It contains general structures and classes with traversal usage
in the application.
"""
import os


def get_df_engine():
    if "DATA_FRAME_ENGINE" not in os.environ or os.environ["DATA_FRAME_ENGINE"] == 'pandas':
        return "pandas"
    elif os.environ["DATA_FRAME_ENGINE"] == 'modin':
        return "modin"
