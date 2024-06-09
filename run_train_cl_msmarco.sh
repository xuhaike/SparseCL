# BGE finetuning on MSMARCO dataset

python train.py \
    --model_name our-bge \
    --model_name_or_path BAAI/bge-base-en-v1.5 \
    --n_gpu 1 \
    --train_file data/msmarco_train_gpt4_final.csv \
    --eval_file data/msmarco_dev_gpt4_final.csv \
    --output_dir result/our-bge-msmarco-finetune-bz64-lr2-ml256-0514 \
    --gradient_checkpointing True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --max_seq_length 256 \
    --pad_to_max_length True \
    --pooler_type avg \
    --overwrite_output_dir \
    --temp 0.02 \
    --sparsity_temp 0.02 \
    --loss_type cos \
    --dataloader_drop_last True \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"
