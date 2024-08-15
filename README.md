# Image Similarity Calculation

[Calculating Image Similarity using DINO and CLIP.]

## Getting Started

These instructions will help you set up the project on your local machine for development and testing purposes.

### Prerequisites

Before running the script, you need to ensure that you have Python installed on your local machine. You can download Python from the official website: [python.org](https://www.python.org/downloads/)

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/SKDDJ/Image_Sim_Cal.git
cd Image_Sim_Cal
```

Install the required packages using pip:

```bash
pip install transformers Pillow torch
```

### Usage

To run the scoring script, execute the following command in the terminal:

```bash
python score.py
```

#### Jupyter Notebook

If you prefer to use a Jupyter Notebook, you can open `score.ipynb` and execute the cells incrementally.

```bash
jupyter notebook score.ipynb
```

### Customization

You can adjust the reference image used in the scoring process by modifying the `score.py` file and change the path to the image you would like to use.

## Contributing

Please feel free to submit pull requests to us. We appreciate your contributions!
