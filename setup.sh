#!/bin/bash
set -e

# Do it yourself
# git clone https://github.com/Cheol-Will/tabular-llm-adaptation.git tabarena

echo "=== Installing UV ==="
pip install uv
export PATH="$HOME/.local/bin:$PATH" 

echo "=== Creating virtual environment ==="
uv venv --seed --python 3.11 ~/.venvs/tabarena
source ~/.venvs/tabarena/bin/activate

echo "=== Installing AutoGluon ==="
git clone https://github.com/Cheol-Will/custom_autogluon.git autogluon
./autogluon/full_install.sh

echo "=== Installing TabArena ==="
git clone https://github.com/Cheol-Will/tabular-llm-adaptation.git tabarena
cd tabarena
uv pip install --prerelease=allow -e "./tabarena[benchmark]"
cd ..

echo "=== Installing Additional Libraries ==="
uv pip install wandb

cd examples/benchmarking
python run_quickstart_tabarena.py 

echo "=== Complete Tabarena Setup ==="