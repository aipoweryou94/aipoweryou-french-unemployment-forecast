.PHONY: check data forecast format install lint plot test test-integration typecheck

install:
	python -m pip install -e ".[dev]"

data:
	python fetch_open_data.py

test:
	python -m pytest -m "not integration" --cov=aipoweryou_forecast --cov-fail-under=80

test-integration:
	RUN_INTEGRATION=1 python -m pytest -m integration

forecast:
	python train_transformer.py

plot:
	python plot_scenarios.py

lint:
	python -m ruff check .
	python -m ruff format --check .

format:
	python -m ruff check . --fix
	python -m ruff format .

typecheck:
	python -m mypy src fetch_open_data.py plot_scenarios.py train_transformer.py

check: lint typecheck test
