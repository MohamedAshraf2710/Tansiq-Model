install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black app/*.py tests/*.py

lint:
	pylint --disable=R,C,W1203,W0718 app/*.py || true

test:
	python -m pytest -vv tests/test_main.py

all: install format lint test