#!/bin/bash

# Convenience script for running analyses
# Usage: bash scripts/run_analysis.sh

# Configuration
ANALYSIS_TYPE="hpo"  # Options: hpo, (more to be added)
MODEL="LLMBaseline"  # Options: TFMLLM, LLMBaseline, FTTransformer, etc.
EXP_NAME="260323"    # Experiment directory name

# Run analysis
python experiment/analysis.py \
    --analysis_type $ANALYSIS_TYPE \
    --model $MODEL \
    --exp_name $EXP_NAME

echo ""
echo "Analysis complete! Check evals/$EXP_NAME/ for results."
