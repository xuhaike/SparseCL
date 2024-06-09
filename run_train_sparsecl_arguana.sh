# BGE finetuning on Arguana dataset using SparseCL

python train.py \
    --model_name our-bge \
    --model_name_or_path BAAI/bge-base-en-v1.5 \
    --n_gpu 3 \
    --train_file data/arguana_training_final.csv \
    --eval_file data/arguana_validation_final.csv \
    --output_dir result/our-bge-arguana-sparsity \
    --gradient_checkpointing True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --max_seq_length 512 \
    --pad_to_max_length True \
    --pooler_type avg \
    --overwrite_output_dir \
    --temp 0.02 \
    --sparsity_temp 0.02 \
    --loss_type sparsity \
    --dataloader_drop_last True \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"
