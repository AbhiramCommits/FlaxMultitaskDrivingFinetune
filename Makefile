setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e .

data:
	mkdir -p data artifacts/ckpt_pretrain artifacts/logs artifacts/results artifacts/figs queries
	python -m fmdf.data.generate --n-logs 1200 --out data/raw.parquet
	python -m fmdf.data.sql --build

train:
	python -m fmdf.train.pretrain --config configs/base.yaml --steps 1500

sweep:
	XLA_FLAGS=--xla_force_host_platform_device_count=4 python -m fmdf.experiments.sweep_loss_weights

scale:
	XLA_FLAGS=--xla_force_host_platform_device_count=4 python -m fmdf.experiments.scaling

domain:
	XLA_FLAGS=--xla_force_host_platform_device_count=4 python -m fmdf.experiments.domain_shift

eval:
	python -m fmdf.eval.sensitivity
	python -m fmdf.eval.report

test:
	XLA_FLAGS=--xla_force_host_platform_device_count=2 pytest -q

all: setup data train sweep scale domain eval test
