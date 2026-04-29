# exp_name="260420-LLMRead-GradClip"
# exp_name="260426-BaselineBidir"
exp_name="260424-next_token_pred"
# exp_name="260421-3"
model="LLMSlot"
# model="LLMAdapter"
# model="LLMBaseline"
# model_cls_name="LLMBaselineBidirectional"
# model_cls_name="LLMRead"

num_random_configs=10 # for fast experiment
python experiment/main.py \
    --model $model \
    --exp_name $exp_name \
    --num_random_configs $num_random_configs \
    --task_ids 363707 # multi-clf 363614 363707
    # --task_ids 363675  # tmux 0 -ing
    # --task_ids 363707
    # --model_cls_name $model_cls_name \
    # --subset small
    # --task_ids 363675 
    # --task_ids 363626 363675 # tmux 0 -ing
    # --task_ids 363612 363698  # tmux todo
    # --task_ids 363621 363625 363629  # tmux 1

    # --task_ids 363626 363675
    # --task_ids 363612 363698  
    # --task_ids 
    # --prediction_method next_token_pred \
    # --task_ids 
    
    
    # --task_ids  
    # --task_ids   363629 363626 
    # --problem_type "binary"
    # --use_bidir_attn \
    # --task_ids  
    # --problem_type "binary"
    # --problem_type "reg"


    # --problem_type "reg" \
    # --use_tail_task_ids
    # --problem_type "binary" \
    # --problem_type "multi"

    
    # --problem_type "reg"
    # --problem_type "binary"
    # --task_ids 363612
    # --task_ids   363629 363626 363625
    # --task_ids 363621
    # --task_ids 363612
    # --model_cls_name $model_cls_name \
    # --problem_type "binary"
    # --problem_type "reg"

# exp_name="260413-LLMCT"
# model="LLMColumnSpecificToken"
# python experiment/main.py \
#     --model $model \
#     --exp_name $exp_name \
#     --num_random_configs $num_random_configs \
#     --problem_type "multi"

# exp_name="260402-mlp_ratio"
# exp_name="260402-mlp_ratio-tune_mlp"
# model="LLMAdapterEngineered"
# python experiment/main.py \
#     --model $model \
#     --exp_name $exp_name \
#     --num_random_configs $num_random_configs \
#     --problem_type "reg"
    # --problem_type "multi"


# num_random_configs=20 # for fast experiment
# exp_name="260402-mlp_ratio-tune_mlp"
# model="LLMAdapterReg"
# python experiment/main.py \
#     --model $model \
#     --exp_name $exp_name \
#     --num_random_configs $num_random_configs \
    # --problem_type "reg"


# model_cls_name="LLMBaselineBidirectional"
# python experiment/main.py \
#     --model $model \
#     --exp_name $exp_name \
#     --model_cls_name $model_cls_name \
#     --subset "small-large-features"
#     # --subset "small"


# model="TFMLLM"
# exp_name="260403-mlp_ratio-mlp_fine_tune"
# python experiment/main.py \
#     --model $model \
#     --exp_name $exp_name \
#     --num_random_configs $num_random_configs \
#     --problem_type "reg"
#     # --problem_type "binary"
#     # --problem_type "multi"
    # --task_ids 363707
