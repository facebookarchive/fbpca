PYTHON ?= python
NOSETESTS ?= nosetests

clean:
	$(PYTHON) setup.py clean
	rm -rf dist

inplace:
	$(PYTHON) setup.py build_ext -i
	$(PYTHON) setup.py build

test: inplace
	$(NOSETESTS) -s -v fbpca

doc:
	cd docs && make clean && make html
