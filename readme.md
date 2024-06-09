# SparseCL 
![Sentence Embedding](https://img.shields.io/badge/sentence-embedding-green) 
This repository contains the official implementation and data for "SPARSECL: Sparse Contrastive Learning for
Contradiction Retrieval".
In the supplementary material, we provide the necessary code to reproduce our main results on the Arguana dataset. All the training and test data are located in the **data** folder.

[[Webpage](https://sparsecl.github.io/)] [[Paper](preprint.pdf)] [[Huggingface Dataset](https://huggingface.co/SparseCL/all-train-test-data)] [[Twitter](TBD)] [[Model Checkpoints](https://huggingface.co/SparseCL)]

## Setup Environment

Before running the experiments, ensure that you have the correct environment set up by referring to the `requirements.txt` file. This file contains all the necessary packages.

## Running the Experiments

To conduct the experiments, use the following scripts provided in our repository:

### Standard Contrastive Learning
To perform standard contrastive learning on the Arguana dataset, run:
```bash
./run_train_cl_arguana.sh

# train on hotpotqa
./run_train_cl_hotpotqa.sh

# train on msmarco
./run_train_cl_msmarco.sh
```
If you want to use different model, please change the args: 'model_name_or_path',
e.g., if you want to use UAE as the backbone model, you can replace 'BAAI/bge-base-en-v1.5' with 'WhereIsAI/UAE-Large-V1'
```bash
python train.py \
    --model_name our-uae \
    --model_name_or_path WhereIsAI/UAE-Large-V1 \
    --n_gpu 1 \
    --train_file data/msmarco_train_gpt4_final.csv \
    --eval_file data/msmarco_dev_gpt4_final.csv \
    --output_dir result/our-uae-msmarco-finetune-bz64-lr2-ml256 \
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
    --temp 0.05 \
    --sparsity_temp 0.05 \
    --loss_type cos \
    --dataloader_drop_last True \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"
```
Currently, we support: 'BAAI/bge-base-en-v1.5', 'WhereIsAI/UAE-Large-V1', 'Alibaba-NLP/gte-large-en-v1.5'
### Sparse Contrastive Learning

Run the following script to perform SparseCL on Arguana:
```bash
# train on arguana
./run_train_sparsecl_arguana.sh

# train on hotpotqa
./run_train_sparsecl_hotpotqa.sh

# train on msmarco
./run_train_sparsecl_msmarco.sh
```
If you want to use different model, please change the args: 'model_name_or_path',
e.g., if you want to use UAE as the backbone model, you can replace 'BAAI/bge-base-en-v1.5' with 'WhereIsAI/UAE-Large-V1'
```bash
python train.py \
    --model_name our-uae \
    --model_name_or_path WhereIsAI/UAE-Large-V1 \
    --n_gpu 1 \
    --train_file data/msmarco_train_gpt4_final.csv \
    --eval_file data/msmarco_dev_gpt4_final.csv \
    --output_dir result/our-uae-msmarco-finetune-bz64-lr2-ml256 \
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
    --temp 0.05 \
    --sparsity_temp 0.05 \
    --loss_type sparsity \
    --dataloader_drop_last True \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"
```
Currently, we support: 'BAAI/bge-base-en-v1.5', 'WhereIsAI/UAE-Large-V1', 'Alibaba-NLP/gte-large-en-v1.5'
### Testing the Model

Finally, test Arguana by running the following script:
```bash
# test on arguana
./run_test_arguana.sh

# test on hotpotqa
./run_test_hotpotqa.sh

# test on msmarco
./run_test_msmarco.sh
```
Currently, we support: 'BAAI/bge-base-en-v1.5', 'WhereIsAI/UAE-Large-V1', 'Alibaba-NLP/gte-large-en-v1.5'

### Data Cleaning
If you want to run data cleaning experiments in our paper please use the following script:
```bash
python test_data_cleaning.py \
    --dataset_name hotpotqa \
    --cos_model_name gte \
    --model_name_or_path {YOUR_GTE_CHECKPOINT} \
    --write_path test_results/data_cleaning_hotpotqa \
    --pooler_type avg \
    --max_seq_length 256 \

python test_data_cleaning.py \
    --dataset_name msmarco \
    --cos_model_name gte \
    --model_name_or_path {YOUR_GTE_CHECKPOINT} \
    --write_path test_results/data_cleaning_msmarco \
    --pooler_type avg \
    --max_seq_length 256 \
```
Currently, we evaluate on 'Alibaba-NLP/gte-large-en-v1.5'.
## Source Code Acknowledgement

Part of our code is adapted from [SimCSE at Princeton NLP](https://github.com/princeton-nlp/SimCSE).


## Contact
If you have any questions about the implementation, please contact linzongy21@cs.ucla.edu or haikexu@mit.edu
