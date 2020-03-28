

build:
	maturin build --release --manylinux off -o wheels -i /opt/miniconda3/bin/python

install: build
	pip install wheels/floky-0.1.0-cp37-cp37m-linux_x86_64.whl

uninstall:
	pip uninstall floky -y

clean: uninstall
	@rm wheels/*
