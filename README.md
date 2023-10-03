# CFRU
Repo for 'Certified Unlearning for Federated Recommendation'

## QuickStart

**First**, run the command below to get the MovieLens-1M Dataset (do the same for the case of Pinterest Dataset):

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
The splited data will be stored in ` ./fedtask/movielens1m_cnum100_dist0_skew0.0_seed0/data.json`.

**Third**, run the command below to quickly run the experiment on MovieLens-1M dataset:

```sh
# all parameters are set in the file run_exp.sh
bash run_exp.sh
```
The result will be stored in `./fedtasksave/movielens1m_cnum100_dist0_skew0_seed0/CFRU_NCF_Mu8_R40_P0.30_alpha0.5_seed0_fedAttack/record/history40.pkl`.

**Finally**, run the command below to return accuracy:

```sh
# return accuracy
python CFRU_test.py
# Hit Ratio and NDCG for top 10 recommendation!
# HR@10:    ...    || NDCG@10:    ...
```
