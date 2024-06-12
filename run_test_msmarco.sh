
# first test on the dev set to select the alpha paramters

# method: CL (cosine) + SparseCL (hoyer)

python test_contradiction_faiss_final.py \
    --dataset_name msmarco \
    --split dev \
    --model_name_or_path results/our-uae-msmarco-sparsity \
    --cos_model_name_or_path results/our-uae-msmarco-finetune \
    --write_path test_results/dev_msmarco_uae_finetune_hoyer_faiss/ \
    --pooler_type avg \
    --metric both_sum \
    --max_seq_length 256 \

# method: Zeroshot (cosine) + SparseCL (hoyer)

python test_contradiction_faiss_final.py \
    --dataset_name msmarco \
    --split dev \
    --model_name_or_path results/our-uae-msmarco-sparsity \
    --cos_model_name uae \
    --write_path test_results/dev_msmarco_uae_zeroshot_hoyer_faiss/ \
    --pooler_type avg \
    --metric both_sum \
    --max_seq_length 256 \


# Then, test on the test set will automatically pick the alpha selected above.

python test_contradiction_faiss_final.py \
    --dataset_name msmarco \
    --split test \
    --model_name_or_path results/our-uae-msmarco-sparsity \
    --cos_model_name_or_path results/our-uae-msmarco-finetune \
    --write_path test_results/test_msmarco_uae_finetune_hoyer_faiss/ \
    --pooler_type avg \
    --metric both_sum \
    --max_seq_length 256 \

python test_contradiction_faiss_final.py \
    --dataset_name msmarco \
    --split test \
    --model_name_or_path results/our-uae-msmarco-sparsity \
    --cos_model_name uae \
    --write_path test_results/test_msmarco_uae_zeroshot_hoyer_faiss/ \
    --pooler_type avg \
    --metric both_sum \
    --max_seq_length 256 \
