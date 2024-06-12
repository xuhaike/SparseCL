# BGE finetuning on Arguana dataset

export CUDA_VISIBLE_DEVICES=0

python train.py \
    --model_name our_bge \
    --model_name_or_path BAAI/bge-base-en-v1.5 \
    --train_file data/arguana_training_final.csv \
    --eval_file data/arguana_validation_final.csv \
    --output_dir results/our-bge-arguana-finetune \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --max_seq_length 512 \
    --pad_to_max_length True \
    --pooler_type avg \
    --overwrite_output_dir \
    --temp 0.02 \
    --loss_type cos \
    --dataloader_drop_last True \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"
