# Tensordoc

Tensordoc is a Python library for extracting information from documents. 

## Installation

### Install from source

```bash
pip install git+https://github.com/tensorlakeai/tensordoc.git
```

In addition, you need to install the dependencies for the layout detector and OCR agent you want to use.

### Detectron2

```bash
pip install "detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.5#egg=detectron2"
```

### Tesseract OCR

Install Tesseract OCR from the instructions at [Tesseract OCR](https://tesseract-ocr.github.io/tessdoc/Installation.html) and then install pytesseract
```bash
pip install pytesseract
```

## Usage

```python

import cv2
from tensordoc.layout_detector import LayoutDetectorFactory, LayoutDetectorType
from tensordoc.ocr import OCRFactory, OCRType

# Read your pdf documents
image = cv2.imread("./data/sample.png")

# Initialize the layout detector
layout_detector = LayoutDetectorFactory.get_layout_detector(
    detector_name=LayoutDetectorType.FASTER_RCNN
)

# Initialize the OCR agent

ocr_agent = OCRFactory.get_ocr(
    OCRType.TESSERACT,
    lang="eng"
)

# Obtain layout components

layout = layout_detector.process(image)

# Obtain text from layout components that have text

text_blocks = Layout([l for l in layout if l.type in ["Text", "Title"]])

text_snippets = []
for layout_component in text_blocks:
    segment_image = (layout_component
                     .pad(left=5, right=5, top=5, bottom=5)
                     .crop_image(image))
    text = ocr_agent.process(segment_image)
    text_segments.append(text)

print(text_snippets)
```


