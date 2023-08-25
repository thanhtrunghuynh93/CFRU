<<<<<<< HEAD
# CFRU
Repo for 'Certified Unlearning for Federated Recommendation'

## QuickStart

**First**, run the command below to get the MovieLens-1M Dataset (do the same for the case of Pinterest dataset):

```sh
# change to the ARDIS_DATASET_IV folder
cd benchmark/movielens1m/data
# unrar ARDIS_DATASET
unrar e movielens1m.rar
# return to the root folder
cd ....
```

**Second**, run the command below to get the splited dataset MovieLens-1M:

```sh
bash gen_data.sh
```
The splited data will be stored in ` ./fedtask/mnist_cnum25_dist0_skew0_seed0/attack.json`.

**Third**, run the command below to quickly run the experiment on MNIST dataset:

```sh
# all parameters are set in the file run_exp.sh
bash run_exp.sh
```
The result will be stored in ` ./fedtasksave/mnist_cnum25_dist0_skew0_seed0/R51_P0.30_alpha0.07/record/history51.pkl`.

**Finally**, run the command below to return accuracy:

```sh
# return accuracy
python test_unlearn.py
# Main accuracy: ...
# Backdoor accuracy: ...
```
