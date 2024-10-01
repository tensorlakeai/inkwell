# pylint: disable=line-too-long
# flake8: noqa: E501

OCR_SYSTEM_PROMPT_DEFAULT = """You are an expert in OCR. You are given an image with some text in it. You need to extract the text from the image. If there is a specific requirement or customization in further instructions, please follow them. If not, just extract the text from the image."""

OCR_USER_PROMPT_DEFAULT = """Extract the text from the image. Return just the extracted text string, and nothing else. If there are titles and sections, return them in a markdown format. DONOT describe the image or the text in the image. Just return the text that you have extracted."""
