from matplotlib import pyplot as plt   
import numpy as np
import os
import sys
import csv
import aeropy.xfoil_module as xf
from aeropy.CST_2D import CST
from aeropy.CST_2D.fitting import fitting_shape_coefficients, shape_parameter_study

airfoli_fn = os.listdir(".")
data_cd = []
data_cl = []
fn_list = []
angles = 0.

for fn in airfoli_fn:
    with open(fn, 'r', encoding="utf8", errors='ignore') as f:
        try:
            coefs = xf.find_coefficients(
                fn, angles, Reynolds=100000, iteration=100, NACA=False
            )
            print(fn)
            print(coefs['CD'], coefs['CL'])
            if coefs['CD'] is not None and coefs['CL'] is not None:
                data_cd.append(coefs['CD'])
                data_cl.append(coefs['CL'])
                fn_list.append(fn)
        except:
            continue

print('Number of data:', len(data_cd))
# np.savez('../coefs.npz', data_cd, data_cl)
# with open('../file_names.csv', 'w', newline='') as f:
#     wr = csv.writer(f, quoting=csv.QUOTE_ALL)
#     for fn in fn_list:
#         wr.writerow([fn])
# print('coefs saved!')
