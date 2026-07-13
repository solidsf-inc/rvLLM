PY ?= python3

.PHONY: install port emit bytecode status test verify clean

install:
	$(PY) -m pip install -e .

port emit:
	$(PY) -m harness.emit_all

bytecode:
	$(PY) -m harness.emit_bytecode

status:
	$(PY) -m harness.emit_all --status

test:
	$(PY) -m unittest discover -s tests

verify:
	@$(PY) -m harness.verify_all || { \
		echo "TPU verification failed closed" >&2; \
		exit 1; \
	}

clean:
	rm -rf out/*.mlir
