#!/bin/bash
set -e

echo "=== Installing UV ==="
pip install uv

echo "=== Creating virtual environment ==="
uv venv --seed --python 3.11 ~/.venvs/tabarena
source ~/.venvs/tabarena/bin/activate

echo "=== Installing AutoGluon ==="
git clone https://github.com/autogluon/autogluon.git
./autogluon/full_install.sh

echo "=== Installing TabArena (original) ==="
git clone https://github.com/autogluon/tabarena.git
cd tabarena
uv pip install --prerelease=allow -e "./tabarena[benchmark]"
cd ..

echo "=== Installing Additional Libraries ==="
uv pip install wandb

cd examples/benchmarking
python run_quickstart_tabarena.py  # download dataset and artifacts

echo "=== Complete Tabarena Setup ==="

cd ..
echo "=== Cloning our experiment code ==="
git clone https://github.com/Cheol-Will/tabular-llm-adaptation.git
