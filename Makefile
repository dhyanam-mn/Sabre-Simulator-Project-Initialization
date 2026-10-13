.PHONY: test reproduce clean all

all: test reproduce

test:
	python -m pytest -v

reproduce:
	python -u -m sabre.experiments.run_experiment --exp all --seeds 30 --config configs/tuned.yaml

tune:
	python -u -m sabre.experiments.tune

calibrate:
	python -u -m sabre.intel.calibrate

clean:
	rm -rf results/*.csv results/*.png results/*.pdf results/*.txt
