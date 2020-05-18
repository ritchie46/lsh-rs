INTERPRETER = /opt/miniconda3/bin/python
SHELL = /bin/bash

test-build:
	maturin build -o wheels -i $(INTERPRETER)

build:
	maturin build --release -o wheels -i $(INTERPRETER)

install: clean uninstall build
	@-pip install wheels/floky*.whl

uninstall:
	-@pip uninstall floky -y

clean: uninstall
	@-rm wheels/*

pytest:
	@cd test && pytest
