.PHONY: all

clean-test:
	rm -rf .coverage
	rm -rf .pytest_cache
	rm -rf .mypy_cache

typing:
	pytest -v -s --mypy wtsp tests

lint:
	flake8 --statistics --count .

test:
	pytest -v -s --cov=wtsp tests