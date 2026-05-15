#!/bin/bash
set -e

curl -LsSf https://astral.sh/uv/install.sh | sh

echo "=== Installing UV ==="
python3 -m pip install uv
export PATH="$HOME/.local/bin:$PATH" 

git clone https://github.com/Cheol-Will/tabular-llm-adaptation.git tabarena
cd tabarena

echo "=== Creating virtual environment ==="
uv venv --seed --python 3.11 .venv/
source .venv/bin/activate
cd ..

echo "=== Installing AutoGluon ==="
git clone https://github.com/Cheol-Will/custom_autogluon.git autogluon
./autogluon/full_install.sh

echo "=== Installing TabArena ==="
cd tabarena
uv pip install --prerelease=allow -e "./tabarena[benchmark]"
uv pip install wandb

cd examples/benchmarking
python run_quickstart_tabarena.py 

echo "=== Complete Tabarena Setup ==="
touch experiment/.env