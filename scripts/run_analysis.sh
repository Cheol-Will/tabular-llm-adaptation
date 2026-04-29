#!/bin/bash

# Convenience script for running analyses
# Usage: bash scripts/run_analysis.sh

# # Configuration
# ANALYSIS_TYPE="hpo"  # Options: hpo, (more to be added)
# MODEL="LLMBaseline"  # Options: TFMLLM, LLMBaseline, FTTransformer, etc.
# EXP_NAME="260323"    # Experiment directory name

# # Run analysis
# python experiment/analysis.py \
#     --analysis_type $ANALYSIS_TYPE \
#     --model $MODEL \
#     --exp_name $EXP_NAME

# echo ""
# echo "Analysis complete! Check evals/$EXP_NAME/ for results."


# task_id=363675
# task_id=363625
# exp_name="260320-num_emb"

analysis_type="attn-map"
# exp_name="260401-2-engineering"
exp_name="260421-3"
# exp_name="260423-bidir"
# model="LLMAdapterEngineered"
# model="LLMAdapterReg"
model="LLMAdapter"
task_id=363621

python experiment/analysis.py \
    --model $model \
    --exp_name $exp_name \
    --analysis_type $analysis_type \
    --task_id $task_id
    # --analysis_type reg-dist \
