from n2pc_func.conn import get_hemi_df
from n2pc_func.set_paths import get_paths
from n2pc_func.params import subject_list, swp_id
import os
import pandas as pd
# import mne

# mne.set_config('MNE_MEMMAP_MIN_SIZE', '2M') 
# mne.set_config('MNE_CACHE_DIR', '/dev/shm')

def get_all_subj_hemi_df(subject_list: list, swp_id:list, i: str)-> pd.DataFrame:
    ''' Get hemisphere dataframe for all subjects.'''
    df_list = []
    for subject in subject_list:
        if subject in swp_id:
            swp = True
        else:    
            swp = False
        df = get_hemi_df(subject, swp, i)
        df_list.append(df)

    df_all = pd.concat(df_list)
    if not os.path.exists(os.path.join(o, 'all_subj', 'N2pc', 'sensor-connectivity')):
        os.makedirs(os.path.join(o, 'all_subj', 'N2pc', 'sensor-connectivity'))
    df_all.to_csv(os.path.join(o, 'all_subj', 'N2pc', 'sensor-connectivity', 'all-subjects-hemi_df.csv'))
    return df_all

i, o = get_paths()
get_all_subj_hemi_df(subject_list, swp_id, i)
# print(get_hemi_df('01', False, i))