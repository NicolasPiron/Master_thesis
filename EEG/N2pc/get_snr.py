import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from n2pc_func.set_paths import get_paths
from n2pc_func.ERP import get_snr

# subjects_list = get_subject_list()
subjects_list = ['01', '02', '03', '04', '06', '07', '12', '13', '16', '17', '18', '19', '20', '21', '22', '23',
                '51', '52', '53', '54', '55', '56', '58', '59', '60',
                '70', '71', '72', '73', '75', '76', '77', '78', '79', '80', '81', '82', '84', '85', '86', '87']
indir, outdir = get_paths()

df_list = []
failed_subjects = []
for subject in subjects_list:
    try:
        df = get_snr(subject_id=subject, input_dir=indir, output_dir=outdir)
        df_list.append(df)
    except:
        print(f'Error in subject {subject}')
        failed_subjects.append(subject)
        continue

print('===== SNR calculation done =====')
if len(failed_subjects) > 0:
    print(f'===== Computation failed for subjects: {failed_subjects} =====')
else:
    print('===== No subject skipped =====')

df = pd.concat(df_list)

grp_mapping = {'healthy':list(range(1, 24)),
              'pulvinar':[51, 53, 59, 60],
              'thalamus':[52, 54, 55, 56, 57, 58],
              'young':list(range(70, 88))}
id_to_group = {id_: group for group, ids in grp_mapping.items() for id_ in ids}
df['group'] = df['ID'].map(id_to_group)

exclude = [8, 9, 10, 11, 14, 15, 57, 74, 83]
df = df[~df['ID'].isin(exclude)]

outpath = os.path.join(outdir, 'all_subj', 'N2pc', 'n2pc-snr')
if not os.path.exists(outpath):
    os.makedirs(outpath)
df.to_csv(os.path.join(outpath, 'allsubj_snr.csv'), index=False)


fig, ax = plt.subplots(figsize=(3, 3))
sns.boxplot(data=df[df['group']!='young'], x='group', y='SNR', color='grey', width=0.5,
             ax=ax)
sns.swarmplot(data=df[df['group']!='young'], x='group', y='SNR', color='black', size=5,
                alpha=0.5, ax=ax)   
sns.despine()
plt.tight_layout()
fig.savefig(os.path.join(outpath, 'allsubj_snr.png'), dpi=300)