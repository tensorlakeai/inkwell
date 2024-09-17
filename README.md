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


### Basic Usage

```python
from tensordoc.pipeline import Pipeline

results = pipeline.process("/path/to/pdf")
for page in results.pages:
    images = results.get_image_fragments()
    tables = results.get_table_fragments()
    text_blocks = results.get_text_fragments()
```

### Individial Components

Refer to ```demo_detectors.ipynb``` in the ```notebooks``` folder for more details.
