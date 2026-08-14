PYTHON ?= python

.PHONY: install test-python train export pipeline rtl rtl-lint rtl-xsim rtl-verilator

install:
	$(PYTHON) -m pip install -e .

test-python:
	PYTHONPATH=python $(PYTHON) -m unittest discover -s tests/python -v

train:
	$(PYTHON) python/main.py --step pretrain --run-through

export:
	$(PYTHON) python/main.py --step export

pipeline:
	$(PYTHON) python/main.py --step pretrain --run-through
	$(MAKE) rtl

rtl: rtl-lint rtl-xsim rtl-verilator

rtl-lint:
	./verification/rtl/scripts/run_verilator_lint.sh

rtl-xsim:
	./verification/rtl/scripts/run_xsim.sh

rtl-verilator:
	./verification/rtl/scripts/run_verilator.sh
