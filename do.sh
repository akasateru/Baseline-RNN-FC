set -e
time python make_dataset.py
time python RNN+FC_train.py
time python RNN+FC_test.py