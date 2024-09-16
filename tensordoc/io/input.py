import cv2
import numpy as np
import pdfplumber


def read_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    image = image[:, :, ::-1]
    return image


def read_pdf_document(pdf_path: str) -> pdfplumber.pdf.PDF:
    document = pdfplumber.open(pdf_path)
    return document


def convert_page_to_image(page: pdfplumber.pdf.Page) -> np.ndarray:
    image = page.to_image(resolution=1024).original
    return np.array(image)


def read_pdf_as_images(pdf_path: str) -> list[np.ndarray]:
    document = read_pdf_document(pdf_path)
    images = []
    for page in document.pages:
        image = convert_page_to_image(page)
        images.append(image)
    return images
