## Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

## Train
python train_and_predict_rf.py --train

## Predict
python train_and_predict_rf.py --predict

## Run API
uvicorn app_parent:app --reload
