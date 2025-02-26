import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from statannotations.Annotator import Annotator
from scipy.stats import f_oneway
from n2pc_func.set_paths import get_paths

i, _ = get_paths()
dirpath = os.path.join(i, 'all_subj', 'N2pc', 'n2pc-snr')
df = pd.read_csv(os.path.join(dirpath, 'allsubj_snr.csv'))

sns.set_context('talk')
fig, ax = plt.subplots(figsize=(7, 7))
sns.boxplot(data=df[df['group']!='young'], x='group', y='SNR', color='grey', width=0.6,
    ax=ax, showcaps=False, boxprops={'facecolor':'None'}, showfliers=False)
sns.swarmplot(data=df[df['group']!='young'], x='group', y='SNR', size=5, ax=ax)  
pairs=[("healthy", "pulvinar"), ("pulvinar", "thalamus"), ("healthy", "thalamus")]
annotator = Annotator(ax, pairs, data=df, x='group', y='SNR')
annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
annotator.apply_and_annotate()
sns.despine()
plt.tight_layout()
plt.show()
fig.savefig(os.path.join(dirpath, 'allsubj_snr.png'), dpi=300)



# pul_data = data[data['group'] == 'pulvinar']['SNR'].values
# thal_data = data[data['group'] == 'thalamus']['SNR'].values
# healthy_data = data[data['group'] == 'healthy']['SNR'].values

# print(pul_data)
# print(pul_data.shape)

# F_obs, p_val = f_oneway(pul_data, thal_data, healthy_data)
# print(f'F value : {F_obs}, P value : {p_val}')