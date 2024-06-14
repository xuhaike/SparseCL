# GTE finetuning on HotpotQA dataset using SparseCL

export CUDA_VISIBLE_DEVICES=0

python train.py \
    --model_name our_gte \
    --model_name_or_path Alibaba-NLP/gte-large-en-v1.5 \
    --train_file data/hotpotqa_train_gpt4_final.csv \
    --eval_file data/hotpotqa_dev_gpt4_final.csv \
    --output_dir results/our-gte-hotpotqa-sparsity \
    --gradient_checkpointing True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --max_seq_length 256 \
    --pad_to_max_length True \
    --pooler_type avg \
    --overwrite_output_dir \
    --temp 0.02 \
    --loss_type sparsity \
    --dataloader_drop_last True \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"
