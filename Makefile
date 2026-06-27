install:
	pip install -r requirements.txt
train:
	python train.py
test:
	pytest -q -W ignore
app:
	streamlit run app.py
train-cv:
	python train.py --horizon 7 --windows 5 --models naive ridge hw
train-full:
	python train.py --horizon 14 --windows 8 --models naive ridge rf hw
