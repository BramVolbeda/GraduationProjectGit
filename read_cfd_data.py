import pandas as pd
import h5py

# file = pd.read_hdf('./data/pipe_ipcs_ab_cn_c0069_MCA_T_constant_ts20000_cycles3_uOrder2_curcyc_2_t_1902.00_ts=40000_up.h5')
file = h5py.File('./data/pipe_ipcs_ab_cn_c0069_MCA_T_constant_ts20000_cycles3_uOrder2_curcyc_2_t_1902.00_ts=40000_up.h5', 'r')

# print(list(file.keys()))  # 'Solution'
solution = file['Solution']
for name in file: 
    print(name)

for obj in solution.values(): 
    velocity_vector = obj
    
print(velocity_vector[0:100])

file.close()

