import pandas as pd
import os
from scipy.stats import f_oneway
from n2pc_func.set_paths import get_paths

i, _ = get_paths()
fpath = os.path.join(i, 'all_subj', 'N2pc', 'n2pc-snr', 'allsubj_snr.csv')
data = pd.read_csv(fpath)

pul_data = data[data['group'] == 'pulvinar']['SNR'].values
thal_data = data[data['group'] == 'thalamus']['SNR'].values
healthy_data = data[data['group'] == 'healthy']['SNR'].values

print(pul_data)
print(pul_data.shape)

F_obs, p_val = f_oneway(pul_data, thal_data, healthy_data)
print(f'F value : {F_obs}, P value : {p_val}')