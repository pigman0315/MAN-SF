# CCBDA Final Project
## Directorcy Structure
```
|- MAN-SF-master
	|- Data
		|- relation
		|- stocknet-dataset
	|- build_input_data
	|- data_preprocess.py
	|- layers.py
	|- model.py
	|- models.py
	|- README
	|- train.py
	|- utils.py
	|- valid_company.txt
	|- alignment_date_list.txt
	|- requirements.txt

```

## Environment Setup
- Pytorch=1.8.1
- Python=3.8
- Tensorflow=2.3.0
- `pip install -r requirements.txt`

## Dataset
- [Link](https://drive.google.com/file/d/1l8YTujr-Xgr49XM08iDftXHR--PT6bJI/view?fbclid=IwAR3itGn5dsqBD94kl8EVhwROlfFfv3yygdzbiPc-p9PMDv3w8DtDfBeQfFw)

## Run
- First, do data preprocessing, `python data_preprocess.py`
- Then, build input data for model, `python build_intput_data.py`
- Finally, you can run it by `python train.py`
