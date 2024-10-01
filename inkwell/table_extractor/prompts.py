# pylint: disable=line-too-long
# flake8: noqa: E501

TABLE_EXTRACTOR_SYSTEM_PROMPT = """You are a helpful AI assistant specialized in extracting table data from images."""

TABLE_EXTRACTOR_USER_PROMPT = """Extract information from the table image
into the following json format:

{
    "header": List[str] # list of header names
    "data": List[List[str]] # list of rows, each row is a list of strings
}

Strictly return the json output only, and nothing else."""
