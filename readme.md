In the supplementary material, we provide the code to reproduce our main results on the Arguana dataset. All the training and test data are in the "data" folder.

Part of our code is adapted from https://github.com/princeton-nlp/SimCSE

Please refer to requirements.txt to create the environment needed to run our experiments.

Run the following script to perform standard contrastive learning on Arguana:

./run_train_cl_arguana.sh

Run the following script to perform SparseCL on Arguana:

./run_train_sparsecl_arguana.sh

Finally, test Arguana by running the following script:

./run_test_arguana.sh

