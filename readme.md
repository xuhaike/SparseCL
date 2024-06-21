# SparseCL
![Sentence Embedding](https://img.shields.io/badge/sentence-embedding-green)
This repository contains the official implementation and data for "SparseCL: Sparse Contrastive Learning for
Contradiction Retrieval".

[[Webpage](https://sparsecl.github.io/)] [[Paper](https://arxiv.org/pdf/2406.10746)] [[Huggingface Dataset](https://huggingface.co/SparseCL/all-train-dev-test-data)] [[Twitter](TBD)] [[Model Checkpoints](https://huggingface.co/SparseCL)]

## Setup Environment

Before running the experiments, ensure that you have the correct environment set up by referring to the `requirements.txt` file. This file contains all the necessary packages.

## Download Training and Test Data

Please download our training and test data from [Huggingface Dataset](https://huggingface.co/SparseCL/all-train-dev-test-data) and put them in a new folder "./data"

## Running the Experiments

To conduct the experiments, use the following scripts provided in our repository:

### Standard Contrastive Learning
To perform standard contrastive learning on Arguana/HotpotQA/MSMARCO datasets, run:
```bash
# train on arguana
./run_train_cl_arguana.sh

# train on hotpotqa
./run_train_cl_hotpotqa.sh

# train on msmarco
./run_train_cl_msmarco.sh
```
If you want to use different models, please change the args: 'model_name_or_path' and 'model_name',
e.g., if you want to use UAE as the backbone model, you can replace 'BAAI/bge-base-en-v1.5' with 'WhereIsAI/UAE-Large-V1' and 'our_bge' with 'our_uae'. Remember to adjust the hyperparameters accordingly if you change to different models. Please refer to Table 7 in our paper for our hyperparameter choices.
```bash
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
```
Currently, we support: 'BAAI/bge-base-en-v1.5', 'WhereIsAI/UAE-Large-V1', 'Alibaba-NLP/gte-large-en-v1.5'.
### Sparse Contrastive Learning

Run the following scripts to perform SparseCL on Arguana/HotpotQA/MSMARCO datasets:
```bash
# train on arguana
./run_train_sparsecl_arguana.sh

# train on hotpotqa
./run_train_sparsecl_hotpotqa.sh

# train on msmarco
./run_train_sparsecl_msmarco.sh
```
If you want to use different models, please change the args: 'model_name_or_path' and 'model_name',
e.g., if you want to use UAE as the backbone model, you can replace 'BAAI/bge-base-en-v1.5' with 'WhereIsAI/UAE-Large-V1' and 'our_bge' with 'our_uae'. Remember to adjust the hyperparameters accordingly if you change to different models. Please refer to Table 7 in our paper for our hyperparameter choices.
```bash
python train.py \
    --model_name our_bge \
    --model_name_or_path BAAI/bge-base-en-v1.5 \
    --train_file data/arguana_training_final.csv \
    --eval_file data/arguana_validation_final.csv \
    --output_dir results/our-bge-arguana-sparsity \
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
    --loss_type sparsity \
    --dataloader_drop_last True \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"
```
Currently, we support: 'BAAI/bge-base-en-v1.5', 'WhereIsAI/UAE-Large-V1', 'Alibaba-NLP/gte-large-en-v1.5'.
### Testing the Model

Finally, test contradiction retrieval on different datasets by running the following scripts:
```bash
# test on arguana
./run_test_arguana.sh

# test on hotpotqa
./run_test_hotpotqa.sh

# test on msmarco
./run_test_msmarco.sh
```

### Data Cleaning
If you want to run data cleaning experiments in our paper please use the following scripts:
```bash
# test on hotpotqa
./run_test_data_cleaning_hotpotqa.sh

# test on msmarco
./run_test_data_cleaning_msmarco.sh
```

You can use our released [Model Checkpoints](https://huggingface.co/SparseCL) or your own models. 
## Source Code Acknowledgement

Part of our code is adapted from [SimCSE at Princeton NLP](https://github.com/princeton-nlp/SimCSE).


## Contact
If you have any questions about the implementation, please contact linzongy21@cs.ucla.edu or haikexu@mit.edu
