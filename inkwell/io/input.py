import logging
import urllib.error
import urllib.parse
import urllib.request
from io import BytesIO
from typing import List

import cv2
import numpy as np
import pdfplumber

from inkwell.components.document import PageImage

_logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
        AppleWebKit/537.36 (KHTML, like Gecko) \
            Chrome/58.0.3029.110 Safari/537.3"
}


def _is_url(path: str) -> bool:
    parsed_url = urllib.parse.urlparse(path)
    return parsed_url.scheme in ["http", "https"]


def read_image(image_path: str) -> np.ndarray:
    """
    Read an image from a file or a URL.

    Args:
        image_path (str): The path to the image file or URL.

    Returns:
        np.ndarray: The image as a numpy array.
    """
    _logger.debug("Reading image from %s", image_path)

    if _is_url(image_path):
        try:
            req = urllib.request.Request(image_path, headers=HEADERS)
            with urllib.request.urlopen(req) as response:
                arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
                image = cv2.imdecode(arr, -1)
                if image is None:
                    raise ValueError(f"Error reading image from {image_path}")
        except urllib.error.URLError as e:
            raise ValueError(
                f"Error downloading image from {image_path}: {e}"
            ) from e
    else:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error reading image from path: {image_path}")

    image = image[:, :, ::-1]
    return image


def read_pdf_document(pdf_path: str) -> pdfplumber.pdf.PDF:
    """
    Read a PDF document from a file or a URL.

    Args:
        pdf_path (str): The path to the PDF file or URL.

    Returns:
        pdfplumber.pdf.PDF: The PDF document.
    """
    _logger.debug("Reading PDF from %s", pdf_path)
    if _is_url(pdf_path):
        try:
            req = urllib.request.Request(pdf_path, headers=HEADERS)
            with urllib.request.urlopen(req) as response:
                pdf_content = BytesIO(response.read())
                document = pdfplumber.open(pdf_content)
        except urllib.error.URLError as e:
            raise ValueError(f"Error downloading PDF from URL: {e}") from e
    else:
        document = pdfplumber.open(pdf_path)
    return document


def convert_page_to_image(page: pdfplumber.pdf.Page) -> np.ndarray:
    image = page.to_image(resolution=512).original
    return np.array(image)


def read_pdf_as_images(pdf_path: str) -> list[np.ndarray]:
    """
    Read a PDF document as a list of images.

    Args:
        pdf_path (str): The path to the PDF file or URL.

    Returns:
        list[np.ndarray]: The list of images.
    """

    document = read_pdf_document(pdf_path)
    images = []
    for page in document.pages:
        image = convert_page_to_image(page)
        images.append(image)
    return images


def _is_native_pdf(path: str) -> bool:
    return path.endswith(".pdf")


def _preprocess_native_pdf(document, pages_to_parse: List[int] = None):
    pages = document.pages

    if pages_to_parse is not None:
        pages = [page for i, page in enumerate(pages) if i in pages_to_parse]

    pages = [(convert_page_to_image(page), page.page_number) for page in pages]

    return pages


def read_pdf_pages(document_path: str, pages_to_parse: List[int] = None):
    if _is_native_pdf(document_path):
        document = read_pdf_document(document_path)
        pages = _preprocess_native_pdf(document, pages_to_parse)
    else:
        pages = [(read_image(document_path), 1)]

    page_images = []
    for page in pages:
        page_image, page_number = page
        page_images.append(
            PageImage(
                page_image=page_image,
                page_number=page_number,
                page_layout=None,
            )
        )
    return page_images
