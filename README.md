# Tensordoc

Tensordoc is a modular Python library for extracting information from documents. It is designed to be flexible and easy to extend, with a focus on document layout detection, OCR, and table detection. 

You can easily swap out components of the pipeline, and add your own implement new detectors, using your own fintuned models or a cloud-based API.  


## Installation

### Install from source

```bash
pip install git+https://github.com/tensorlakeai/tensordoc.git
```

In addition, you need to install the dependencies for the layout detector and OCR agent you want to use. By default we use Detectron2 based layout detectors and Tesseract OCR for OCR.

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

Use the default pipeline to extract information from a PDF file. Checkout ```demo_pipeline.ipynb``` in the ```notebooks``` folder for more details.

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


## Adding custom components

You can add your own detectors and other components to the pipeline. 
Example of adding a new table extractor is in ```demo_custom_pipeline.ipynb``` in the ```notebooks``` folder.

### Adding Layout or Table Detector

Layout and Table detectors can be subclasses of ```BaseLayoutDetector``` and ```BaseTableDetector``` respectively. You can implement your own detectors and use them in the pipeline. Example of adding a new table extractor is in ```demo_custom_pipeline.ipynb``` in the ```notebooks``` folder.


Custom layout and table detectors  need to implement the ```process``` method and return a ```Layout``` object, and added to the pipeline like this:
```python
class MyLayoutDetector(BaseLayoutDetector):
    def process(self, image: np.ndarray) -> Layout: 
        # Your detection logic here
        return Layout(blocks=blocks)

class MyTableDetector(BaseTableDetector):
    def process(self, image: np.ndarray) -> Layout: 
        # Your detection logic here
        return Layout(blocks=blocks)

pipeline = Pipeline()
pipeline.add_layout_detector(MyLayoutDetector())
pipeline.add_table_detector(MyTableDetector())
```

```python

results = model.detect() # The detection results that your model returns, assuming it returns a dictionary with keys "scores", "labels", and "boxes"

scores = results["scores"]
labels = results["labels"]
boxes = results["boxes"]

blocks = []
for score, label, box in zip(scores, labels, boxes):
    block = LayoutBlock(
        block=Rectangle(
            x_1=box[0], y_1=box[1], x_2=box[2], y_2=box[3]
        ),
        score=score.item(),
        type=class_label_map[label.item()], # This is the type of the layout element, e.g. "Table" or "Text"
    )
    blocks.append(block)

layout = Layout(blocks=blocks)
```

### Adding OCR Detector

OCR detector can be a subclass of ```BaseOCRDetector```, that has a ```process``` method, which takes an image (np.ndarray) as input and returns a ```str``` object.

```python
class MyOCRDetector(BaseOCRDetector):
    def process(self, image: np.ndarray) -> str: 
        # Your OCR logic here
        return "Extracted text"

pipeline = Pipeline()
pipeline.add_ocr_detector(MyOCRDetector())
```

### Adding Table Extractor

Table extractor can be a subclass of ```BaseTableExtractor```, that has a ```process``` method, which takes an image (np.ndarray) as input and returns a ```dict``` object.

```python
class MyTableExtractor(BaseTableExtractor):
    def process(self, image: np.ndarray) -> dict: 
        # Your table extraction logic here
        return Layout(blocks=blocks)

pipeline = Pipeline()
pipeline.add_table_extractor(MyTableExtractor())
```


### Acknowledgements

We derived inspiration from several open-source libraries in our implementation, like [Layout Parser](https://github.com/Layout-Parser/layout-parser) and [Deepdoctection](https://github.com/deepdoctection/deepdoctection). We would like to thank the contributors to these libraries for their work.
