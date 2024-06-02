# Reproducing Results on the Arguana Dataset

In the supplementary material, we provide the necessary code to reproduce our main results on the Arguana dataset. All the training and test data are located in the **data** folder.

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

### Sparse Contrastive Learning

Run the following script to perform SparseCL on Arguana:
```bash
./run_train_sparsecl_arguana.sh

### Testing the Model

Finally, test Arguana by running the following script:
```bash
./run_test_arguana.sh

