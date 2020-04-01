

build:
	maturin build --release -o wheels -i /opt/miniconda3/bin/python

install: clean uninstall build
	pip install wheels/floky*.whl

uninstall:
	pip uninstall floky -y

clean: uninstall
	@-rm wheels/*
