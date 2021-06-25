# CCBDA Final Project - MAN-SF Implementation

## Environment Setup
- Pytorch=1.8.1
- Python=3.8
- Tensorflow=2.3.0
- `pip install -r requirements.txt`

## Run
- First, do data preprocessing, `python data_preprocess.py`
- Then, build input data for model, `python build_intput_data.py`
	- This operation might take you a while(~10 minutes)
- Finally, you can run it by `python train.py`
