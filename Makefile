
PYTHON := .venv/bin/python

.PHONY: all simulate analyze figures test clean pilot

all: simulate analyze figures

simulate:
	$(PYTHON) -m src.simulation

analyze:
	$(PYTHON) -m src.analyze

figures:
	$(PYTHON) -m src.figures

test:
	$(PYTHON) -m pytest tests/ -v

pilot:
	$(PYTHON) -m src.simulation --pilot

clean:
	rm -f results/raw/simulation_results.csv
	rm -f results/recovery_summary.csv
	rm -f results/figures/*.png
