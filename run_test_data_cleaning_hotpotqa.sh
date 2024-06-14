python test_data_cleaning.py \
    --dataset_name hotpotqa \
    --cos_model_name gte \
    --model_name_or_path SparseCL/GTE-SparseCL-hotpotqa \
    --write_path test_results/data_cleaning_hotpotqa \
    --pooler_type avg \
    --max_seq_length 256 \
