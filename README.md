# Inkwell

Inkwell is a modular Python library for extracting information from documents. It is designed to be flexible and easy to extend, with a focus on document layout detection, OCR, and table detection. 

You can easily swap out components of the pipeline, and add your own components, using custom models or a cloud-based API.  

## Installation

```bash
pip install py-inkwell
```

In addition, install detectron2

```bash
pip install git+https://github.com/facebookresearch/detectron2.git
```


Install [Tesseract](https://tesseract-ocr.github.io/tessdoc/Installation.html) for your Operating System 

#### Ubuntu

```bash
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev
```

#### Mac OS

```bash
brew install tesseract
```

If you want to run the pipeline on GPU, install flash-attn

```bash
pip install flash-attn --no-build-isolation
```

For paddle detection based models, install PaddlePaddle and PaddleOCR

```bash
pip install paddlepaddle "paddleocr>=2.0.1"
```

### Basic Usage

```python
from inkwell.pipeline import Pipeline
from inkwell import PipelineConfig

pipeline = Pipeline()

document = pipeline.process("/path/to/file.pdf")

for page in document.pages:

    figures = page.image_fragments()
    tables = page.table_fragments()
    text_blocks = page.text_fragments()

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

## Models/Frameworks currently available



Default models: We have defined a config class [here](inkwell/pipeline/pipeline_config.py), and we use the default CPU Config in the pipeline for best results. If you want to use the default GPU pipeline, you can instantiate it with the GPU config class. 

```python
from inkwell.pipeline import DefaultGPUPipelineConfig, Pipeline
config = DefaultGPUPipelineConfig()
pipeline = Pipeline(config=config)
```

If you want to change the default models, you can replace them with models listed below by passing them in the config during pipeline initialization:

#### Layout Detection

* Faster RCNN
* LayoutLMv3

#### Table Detection

* Table Transformer

#### Table Extraction

* Table Transformer
* Phi3.5-Vision
* Qwen2 VL 2B

#### OCR

* Tesseract
* Phi 3.5-Vision
* Qwen2 VL 2B
* OpenAI GPT-4o (requires an API key)

```python
from inkwell.pipeline import PipelineConfig, Pipeline
from inkwell.layout_detector import LayoutDetectorType
from inkwell.ocr import OCRType
from inkwell.table_detector import TableDetectorType, TableExtractorType

config = PipelineConfig(
    layout_detector=LayoutDetectorType.FASTER_RCNN,
    table_extractor=TableExtractorType.PHI3_VISION,
)

pipeline = Pipeline(config=config)
```


## Advanced Customizations

You can add custom detectors and other components to the pipeline yourself - follow the instructions in the [Custom Components](notebooks/demo_pipeline_custom.ipynb) notebook

### Acknowledgements

We derived inspiration from several open-source libraries in our implementation, like [Layout Parser](https://github.com/Layout-Parser/layout-parser) and [Deepdoctection](https://github.com/deepdoctection/deepdoctection). We would like to thank the contributors to these libraries for their work.
