.PHONY: install train train-fi train-equity train-all predict upload clean test

install:
	pip install -r requirements.txt

train-fi:
	python train_fixed.py --module fi
	python train_shrinking.py --module fi

train-equity:
	python train_fixed.py --module equity
	python train_shrinking.py --module equity

train-all: train-fi train-equity

predict:
	python predict.py --module fi
	python predict.py --module equity

upload:
	python upload_to_hub.py

test:
	python -m pytest test_core.py test_integration.py

clean:
	rm -rf data/cache/* models_saved/* outputs/* __pycache__

daily-run: train-all predict upload
