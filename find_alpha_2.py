import numpy as np
import math
import matplotlib.pyplot as plt

# list_alpha = np.load("list_alpha.npz")["list_alpha"] # clean hoan toan
list_alpha = np.load("1-2-3-clean-alpha.npz", allow_pickle=True)['dict_alpha'] # clean hoan toan
list_alpha_200 = np.load("list_alpha_200_not_clean.npz")["list_alpha"] # clean mot phan
dict_alpha = np.load("dict_alpha_200_da_fa.npz", allow_pickle=True)["dict_alpha"].item() # alpha cho tung layer
print(list_alpha_200)
import pdb
pdb.set_trace()