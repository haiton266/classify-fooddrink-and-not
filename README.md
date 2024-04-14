# Model Food Drink Classification

## Structure folder

- `dataset/`
  - `train/`
    - `valid/`
    - `notvalid/`
  - `val/`
    - `valid/`
    - `notvalid/`
  - `test/`
    - `valid/`
    - `notvalid/`
- `train.py` - Script to train the model.
- `evaluate.py` - Script to evaluate the model.
- `inference.py` - Script for model inference.
- `AI_api.py` - FastAPI server process for the AI API.
- `mobilenetv2_food_classifier.h5` - The trained model file.

## Install

```
    pip install -r requirements.txt
```

## Run

To Train:

```
    python train.py
```

To evaluate: (Caculate accuracy on test dataset)

```
    python evaluate.py
```

To inference (input image, output: result)

```
    python inference.py
```

To run server AI API:

```
    python AI_api.py
```
