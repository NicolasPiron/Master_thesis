from n2pc_func.conn import get_hemi_df
from n2pc_func.set_paths import get_paths
from n2pc_func.params import subject_list
import os
import pandas as pd

def get_all_subj_hemi_df(subject_list: list, i: str)-> pd.DataFrame:
    ''' Get hemisphere dataframe for all subjects.'''
    for subject in subject_list:
        df_list = []
        df = get_hemi_df(subject, i)
        df_list.append(df)

    df_all = pd.concat(df_list)
    if not os.path.exists(os.path.join(o, 'all_subj', 'N2pc', 'sensor-connectivity')):
        os.makedirs(os.path.join(o, 'all_subj', 'N2pc', 'sensor-connectivity'))
    df_all.to_csv(os.path.join(o, 'all_subj', 'N2pc', 'sensor-connectivity', 'all-subjects-hemi_df.csv'))
    return df_all

i, o = get_paths()
get_all_subj_hemi_df(subject_list, i)