Note - Unzip the data folder in the root directory, it should be in the same folder as income_prediction_segmentation.ipynb/py and generate_dataset.py files.
# Income Prediction and Customer Segmentation

This repository contains code to train and evaluate:
- An income prediction model
- A customer segmentation model

## Requirements

- Python 3.9+

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Input Data
To generate output.csv run the generate_dataset.py code and place the input dataset in the project directory:
- `output.csv`

## Run the Code

### Option 1: Jupyter Notebook

Run the notebook:

```bash
jupyter notebook income_prediction_segmentation.ipynb
```

Execute all cells from top to bottom.

### Option 2: Python Script

Run the script:

```bash
python income_prediction_segmentation.py
```

## Models Generated

| Model | Filename |
|-------|----------|
| Income prediction | `model_1_predictive_model.joblib` |
| Customer segmentation | `model_2_segmentation_model.joblib` |

## Load Saved Models

```python
import joblib

# Load models
income_model = joblib.load("model_1_predictive_model.joblib")
segment_model = joblib.load("model_2_segmentation_model.joblib")
```

## Notes

- Preprocessing steps are included inside the saved models
- New data must follow the same structure as the training data
- Models are saved automatically when the code is executed
