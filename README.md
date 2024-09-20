# Tensordoc

Tensordoc is a modular Python library for extracting information from documents. It is designed to be flexible and easy to extend, with a focus on document layout detection, OCR, and table detection. 

You can easily swap out components of the pipeline, and add your own implement new detectors, using your own fintuned models or a cloud-based API.  


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

results = pipeline.process("/path/to/file.pdf")

for page in results.pages:

    figures = page.get_image_fragments()
    tables = page.get_table_fragments()
    text_blocks = page.get_text_fragments()

    # Check the content of the image fragments
    for figure in figures:
        figure_image = figure.content.image
        print(f"Text in figure:\n{figure.content.text}")
    
    # Check the content of the table fragments
    for table in tables:
        table_image = table.content.image
        print(f"Table detected: {table.content.data}")

    # Check the content of the text blocks
    for text_block in text_blocks:
        text_block_image = text_block.content.image
        print(f"Text block detected: {text_block.content.text}")
```

### Individial Components

Refer to ```demo_detectors.ipynb``` in the ```notebooks``` folder for more details.