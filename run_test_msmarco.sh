
# first test on dev set to select the alpha paramters

# method: CL (cosine) + SparseCL (hoyer)export CUDA_VISIBLE_DEVICES=2

python test_contradiction_faiss_final.py \
    --dataset_name msmarco \
    --split dev \
    --model_name_or_path result/our-bge-msmarco-sparsity-bz64-lr2-ml256-0514 \
    --cos_model_name_or_path result/our-bge-msmarco-finetune-bz64-lr2-ml256-0514 \
    --write_path test_results/dev_msmarco_bge_finetune_hoyer_bz64_lr2_ml256_faiss_0514/ \
    --pooler_type avg \
    --metric both_sum \
    --max_seq_length 256 \

# method: Zeroshot (cosine) + SparseCL (hoyer)

python test_contradiction_faiss_final.py \
    --dataset_name msmarco \
    --split dev \
    --model_name_or_path result/our-bge-msmarco-sparsity-bz64-lr2-ml256-0514 \
    --cos_model_name bge \
    --write_path test_results/dev_msmarco_bge_zeroshot_hoyer_bz64_lr2_ml256_faiss_0514/ \
    --pooler_type avg \
    --metric both_sum \
    --max_seq_length 256 \


# Then, test on the test set will automatically pick the alpha selected above.

python test_contradiction_faiss_final.py \
    --dataset_name msmarco \
    --split test \
    --model_name_or_path result/our-bge-msmarco-sparsity-bz64-lr2-ml256-0514 \
    --cos_model_name_or_path result/our-bge-msmarco-finetune-bz64-lr2-ml256-0514 \
    --write_path test_results/test_msmarco_bge_finetune_hoyer_bz64_lr2_ml256_faiss_0514/ \
    --pooler_type avg \
    --metric both_sum \
    --max_seq_length 256 \

python test_contradiction_faiss_final.py \
    --dataset_name msmarco \
    --split test \
    --model_name_or_path result/our-bge-msmarco-sparsity-bz64-lr2-ml256-0514 \
    --cos_model_name bge \
    --write_path test_results/test_msmarco_bge_zeroshot_hoyer_bz64_lr2_ml256_faiss_0514/ \
    --pooler_type avg \
    --metric both_sum \
    --max_seq_length 256 \
