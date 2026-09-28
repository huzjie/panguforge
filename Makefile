.PHONY: doctor pretrain sft rl serve bench test install

install:
	pip install -e .

doctor:
	python -m panguforge doctor --config config.example.yaml

pretrain:
	python -m panguforge pretrain --config examples/config_pretrain.yaml

sft:
	python -m panguforge sft --config examples/config_sft.yaml

rl:
	python -m panguforge rl --config examples/config_rl.yaml

serve:
	python -m panguforge serve --config config.example.yaml --port 8000

bench:
	python -m panguforge bench --config config.example.yaml

test:
	python -m unittest discover -s tests -t .
