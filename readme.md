# SparseCL 
![Sentence Embedding](https://img.shields.io/badge/sentence-embedding-green) 
This repository contains the official implementation and data for "SPARSECL: Sparse Contrastive Learning for
Contradiction Retrieval".
In the supplementary material, we provide the necessary code to reproduce our main results on the Arguana dataset. All the training and test data are located in the **data** folder.

[[Webpage](https://sparsecl.github.io/)] [[Paper](preprint.pdf)] [[Huggingface Dataset](TBD)] [[Twitter](TBD)]



## Source Code Attribution

Part of our code is adapted from [SimCSE at Princeton NLP](https://github.com/princeton-nlp/SimCSE).

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

## Contact
If you have any questions about the implementation, please contact linzongy21@cs.ucla.edu or haikexu@mit.edu
