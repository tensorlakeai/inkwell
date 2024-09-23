# Document

Tensordoc is a Python library for extracting information from PDF documents. It is designed to be modular and easy to extend with Vision LLM models, with a focus on document layout detection, OCR, and table detection. 

You can easily swap out components of the pipeline, and add your own components, using custom models or a cloud-based API.  

## Installation

### Install from source

```bash
pip install tensordoc
```

Install Tessarect for your Operating System 
#### Linux 

#### Mac OS

### Basic Usage

```python
from tensordoc.pipeline import Pipeline
from tensordoc import PipelineConfig

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

## Document Detection Models  

Create a Grid of Models which is used by default and avaialable models by their function 

#### Table Detection

**

#### Table Extraction

**Phi3**

**Qwen**

**PaddlePaddle**

## Advanced Customizations

You can add your own detectors and other components to the pipeline. 

* [**Custom Table Extractor:**](notebooks/demo_custom_pipeline.ipynb)

### Acknowledgements

We derived inspiration from several open-source libraries in our implementation, like [Layout Parser](https://github.com/Layout-Parser/layout-parser) and [Deepdoctection](https://github.com/deepdoctection/deepdoctection). We would like to thank the contributors to these libraries for their work.
