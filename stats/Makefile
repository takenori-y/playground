# Copyright (c) 2024 Takenori Yoshimura
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

check:
	. ../.venv/bin/activate	&& python -m mypy .

format:
	. ../.venv/bin/activate && python -m ruff check --fix .
	. ../.venv/bin/activate && python -m ruff format .

test:
	. ../.venv/bin/activate	&& python -m pytest -s

.PHONY: check format test
