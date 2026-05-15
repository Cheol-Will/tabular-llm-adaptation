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
# task_id=363612
model_name="Qwen/Qwen2.5-0.5B"

task_id=363612
model="LLMSlot"
# attn_type="causal"
# attn_type="bidir"
# attn_type="structured"
attn_type="structured_v2"
exp_name="260506-LS-qwen-$attn_type"
analysis_type="attn-map"

python experiment/analysis.py \
    --model $model \
    --model_name $model_name  \
    --exp_name $exp_name \
    --attn_type $attn_type \
    --task_id $task_id \
    --analysis_type $analysis_type \