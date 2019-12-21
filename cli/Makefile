.PHONY: all

build:
	pip install -r requirements.txt

benchmark:
	pytest -v -s -m benchmark tests

clean: clean-build clean-pyc clean-test

clean-build: ## remove build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -rf .coverage
	rm -rf .pytest_cache
	rm -rf .mypy_cache

install:
	pip install -e .

typing:
	pytest -v -s --mypy wtsp tests

lint:
	flake8 --statistics --count .

test:
	pytest -v -s --cov=wtsp tests

dist:
	python setup.py sdist
	python setup.py bdist_wheel --universal

all: build install typing lint test