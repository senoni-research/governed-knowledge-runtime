.PHONY: setup setup-mlx gate0 test lint doctor eval clean

setup:
	./scripts/bootstrap_macos.sh

setup-mlx:
	INSTALL_MLX=1 ./scripts/bootstrap_macos.sh

gate0:
	PYTHON_BIN=.venv/bin/python ./scripts/verify_gate0.sh

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff check .

doctor:
	.venv/bin/gkr doctor

eval:
	.venv/bin/gkr eval

clean:
	rm -rf artifacts .pytest_cache .ruff_cache .mypy_cache
	find src tests -type d -name __pycache__ -exec rm -rf {} +
