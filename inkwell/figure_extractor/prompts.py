# pylint: disable=line-too-long
# flake8: noqa: E501

FIGURE_EXTRACTOR_SYSTEM_PROMPT = """You are a helpful AI assistant specialized in extracting figure data from images."""

FIGURE_EXTRACTOR_USER_PROMPT = """Extract information from the figure:

* If it is a table, extract the table data into a structured markdown format, along with table title and caption. 
* If it is a graph or chart, extract relevant information.
* If it contains text, extract all the text.

Just return the extracted information, don't include any other text.
"""
