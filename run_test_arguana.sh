
# first test on the dev set to select the alpha paramters

# method: CL (cosine) + SparseCL (hoyer)

python test_contradiction_faiss_final.py \
    --dataset_name arguana \
    --split dev \
    --model_name_or_path results/our-bge-arguana-sparsity \
    --cos_model_name_or_path results/our-bge-arguana-finetune \
    --write_path test_results/dev_arguana_bge_finetune_hoyer \
    --pooler_type avg \
    --metric both_sum \
    --max_seq_length 512 \

# method: Zeroshot (cosine) + SparseCL (hoyer)

python test_contradiction_faiss_final.py \
    --dataset_name arguana \
    --split dev \
    --model_name_or_path results/our-bge-arguana-sparsity \
    --cos_model_name bge \
    --write_path test_results/dev_arguana_bge_zeroshot_hoyer \
    --pooler_type avg \
    --metric both_sum \
    --max_seq_length 512 \

# Then, test on the test set will automatically pick the alpha selected above.

python test_contradiction_faiss_final.py \
    --dataset_name arguana \
    --split test \
    --model_name_or_path results/our-bge-arguana-sparsity \
    --cos_model_name_or_path results/our-bge-arguana-finetune \
    --write_path test_results/test_arguana_bge_finetune_hoyer \
    --pooler_type avg \
    --metric both_sum \
    --max_seq_length 512 \

python test_contradiction_faiss_final.py \
    --dataset_name arguana \
    --split test \
    --model_name_or_path results/our-bge-arguana-sparsity \
    --cos_model_name bge \
    --write_path test_results/test_arguana_bge_zeroshot_hoyer \
    --pooler_type avg \
    --metric both_sum \
    --max_seq_length 512 \
