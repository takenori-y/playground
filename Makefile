PYTHON_VERSION := 3.9

venv:
	test -d .venv || python$(PYTHON_VERSION) -m venv .venv
	. ./.venv/bin/activate && python -m pip install -U pip
	. ./.venv/bin/activate && python -m pip install -U -r requirements.txt
	. ./.venv/bin/activate && python -m pip install -U -r test_requirements.txt

.PHONY: venv